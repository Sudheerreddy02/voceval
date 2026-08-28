from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import AsyncIterator

from voceval.pipeline.base import LLM, STT, TTS, VAD
from voceval.tools.registry import ToolRegistry
from voceval.tracing import timeline as tl
from voceval.tracing.timeline import Timeline
from voceval.transport.base import Channel
from voceval.types import AudioChunk, Message, Speaker, ToolCall, Transcript, Turn

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
MAX_TOOL_STEPS = 4


class Orchestrator:
    def __init__(
        self,
        *,
        vad: VAD,
        stt: STT,
        llm: LLM,
        tts: TTS,
        tools: ToolRegistry,
        system_prompt: str,
        greeting: str | None = None,
        sample_rate: int = 16000,
    ) -> None:
        self.vad = vad
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.tools = tools
        self.greeting = greeting
        self.sample_rate = sample_rate

        self.timeline = Timeline()
        self.history: list[Message] = [Message("system", system_prompt)]
        self.turns: list[Turn] = []

        self._audio_q: asyncio.Queue[AudioChunk] = asyncio.Queue()
        self._turn = 0  # turn 0 is the greeting; caller turns start at 1
        self._caller_speaking = False
        self._speak_task: asyncio.Task | None = None
        self._first_audio_done = False
        self.is_responding = False
        self.response_done = asyncio.Event()
        self.response_done.set()

    async def run(self, channel: Channel) -> None:
        reader = asyncio.create_task(self._read_inbound(channel))
        try:
            if self.greeting:
                self.is_responding = True
                self.response_done.clear()
                await self._say(channel, 0, self.greeting)
                self.history.append(Message("assistant", self.greeting))
                self.is_responding = False
                self.response_done.set()
            async for transcript in self.stt.transcribe(self._feed_stt()):
                await self._on_transcript(transcript, channel)
        finally:
            reader.cancel()
            if self._speak_task:
                self._speak_task.cancel()

    async def _read_inbound(self, channel: Channel) -> None:
        async for chunk in channel.inbound():
            speech = self.vad.is_speech(chunk)
            if speech and not self._caller_speaking:
                self._caller_speaking = True
                self._turn += 1
                self.timeline.mark(tl.CALLER_SPEECH_START, self._turn)
                if self._is_speaking():
                    await self._barge_in(channel, self._turn)
            if chunk.final_hint:
                self._caller_speaking = False
                self.timeline.mark(tl.CALLER_SPEECH_END, self._turn)
            await self._audio_q.put(chunk)

    async def _feed_stt(self) -> AsyncIterator[AudioChunk]:
        while True:
            yield await self._audio_q.get()

    async def _on_transcript(self, transcript: Transcript, channel: Channel) -> None:
        if not transcript.is_final:
            return
        self.timeline.mark(tl.STT_FINAL, self._turn, text=transcript.text)
        self.turns.append(
            Turn(Speaker.CALLER, transcript.text, self.timeline.now(), self.timeline.now())
        )
        self.history.append(Message("user", transcript.text))
        self._first_audio_done = False
        self.is_responding = True
        self.response_done.clear()
        self._speak_task = asyncio.create_task(self._respond(channel, self._turn))

    async def _respond(self, channel: Channel, turn: int) -> None:
        try:
            for _ in range(MAX_TOOL_STEPS):
                tool_call = await self._stream_reply(channel, turn)
                if tool_call is None:
                    return
                self.timeline.mark(tl.TOOL_CALL, turn, name=tool_call.name)
                result = await self.tools.call(tool_call)
                self.history.append(Message("assistant", "", tool_calls=[tool_call]))
                self.history.append(
                    Message("tool", result.content, tool_call_id=tool_call.id)
                )
        finally:
            self.is_responding = False
            self.response_done.set()

    async def _stream_reply(self, channel: Channel, turn: int) -> ToolCall | None:
        self.timeline.mark(tl.LLM_START, turn)
        buffer = ""
        said_something = False
        got_token = False
        async for delta in self.llm.complete(self.history, self.tools.schemas()):
            if delta.tool_call:
                if buffer.strip():
                    await self._say(channel, turn, buffer.strip())
                return delta.tool_call
            if delta.text and not got_token:
                got_token = True
                self.timeline.mark(tl.LLM_FIRST_TOKEN, turn)
            buffer += delta.text
            parts = _SENTENCE_END.split(buffer)
            if len(parts) > 1:
                for sentence in parts[:-1]:
                    if sentence.strip():
                        await self._say(channel, turn, sentence.strip())
                        said_something = True
                buffer = parts[-1]
        if buffer.strip():
            await self._say(channel, turn, buffer.strip())
            said_something = True
        if said_something:
            self.history.append(
                Message("assistant", self._last_agent_text(turn))
            )
        return None

    async def _say(self, channel: Channel, turn: int, text: str) -> None:
        started = self.timeline.now()
        async for chunk in self.tts.synthesize(text):
            if not self._first_audio_done:
                self._first_audio_done = True
                self.timeline.mark(tl.TTS_FIRST_AUDIO, turn)
                self.timeline.mark(tl.AGENT_SPEECH_START, turn)
            await channel.send(chunk)
        self.turns.append(
            Turn(Speaker.AGENT, text, started, self.timeline.now())
        )
        self.timeline.mark(
            tl.AGENT_SPEECH_END, turn, duration=self.timeline.now() - started, text=text
        )

    async def _barge_in(self, channel: Channel, turn: int) -> None:
        self.timeline.mark(tl.BARGE_IN, turn)
        if self._speak_task and not self._speak_task.done():
            self._speak_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._speak_task
        await channel.stop_playback()
        self.timeline.mark(tl.AGENT_STOPPED, turn)
        if self.turns and self.turns[-1].speaker == Speaker.AGENT:
            self.turns[-1].interrupted = True

    def _is_speaking(self) -> bool:
        return self._speak_task is not None and not self._speak_task.done()

    def _last_agent_text(self, turn: int) -> str:
        said = [
            e.meta.get("text", "")
            for e in self.timeline.of_kind(tl.AGENT_SPEECH_END, turn)
        ]
        return " ".join(t for t in said if t)
