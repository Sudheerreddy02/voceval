from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable

from voceval.pipeline.base import LLM, STT, TTS, VAD
from voceval.types import AudioChunk, LLMDelta, Message, ToolCall, Transcript

WORDS_PER_MINUTE = 150.0

Responder = Callable[[list[Message]], "str | ToolCall | Awaitable[str | ToolCall]"]


def speaking_duration(text: str) -> float:
    words = max(len(text.split()), 1)
    return words / WORDS_PER_MINUTE * 60.0


class MockVAD(VAD):
    def is_speech(self, chunk: AudioChunk) -> bool:
        return chunk.transcript_hint is not None or any(chunk.data)


class MockSTT(STT):
    """Rebuilds the caller utterance from the hints the simulated channel attaches,
    emitting word-level partials and one final. final_latency mimics endpointing."""

    def __init__(self, final_latency: float = 0.15, partial_interval: float = 0.25) -> None:
        self.final_latency = final_latency
        self.partial_interval = partial_interval

    async def transcribe(
        self, audio: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[Transcript]:
        spoken: list[str] = []
        last_partial = 0.0
        async for chunk in audio:
            if chunk.transcript_hint:
                spoken.append(chunk.transcript_hint)
                now = asyncio.get_event_loop().time()
                if now - last_partial >= self.partial_interval:
                    yield Transcript(" ".join(spoken), is_final=False, confidence=0.6)
                    last_partial = now
            if chunk.final_hint:
                await asyncio.sleep(self.final_latency)
                yield Transcript(" ".join(spoken), is_final=True, confidence=0.95)
                spoken = []


class MockLLM(LLM):
    def __init__(self, responder: Responder, first_token_latency: float = 0.3) -> None:
        self.responder = responder
        self.first_token_latency = first_token_latency

    async def complete(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> AsyncIterator[LLMDelta]:
        await asyncio.sleep(self.first_token_latency)
        result = self.responder(list(messages))
        if asyncio.iscoroutine(result):
            result = await result

        if isinstance(result, ToolCall):
            yield LLMDelta(tool_call=result)
            yield LLMDelta(done=True)
            return

        for word in str(result).split():
            yield LLMDelta(text=word + " ")
            await asyncio.sleep(0.02)
        yield LLMDelta(done=True)


class MockTTS(TTS):
    def __init__(self, time_to_first_audio: float = 0.12, sample_rate: int = 16000) -> None:
        self.ttfa = time_to_first_audio
        self.sample_rate = sample_rate

    async def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        total = speaking_duration(text)
        await asyncio.sleep(self.ttfa)
        emitted = 0.0
        frame = 0.1
        while emitted < total:
            step = min(frame, total - emitted)
            samples = int(step * self.sample_rate)
            yield AudioChunk(b"\x00\x00" * samples, self.sample_rate, step, transcript_hint=None)
            emitted += step
            await asyncio.sleep(step)
