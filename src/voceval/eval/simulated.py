from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from voceval.pipeline.mock import speaking_duration
from voceval.transport.base import Channel
from voceval.types import AudioChunk

SILENCE = b"\x00\x00"


class SimulatedChannel(Channel):
    """Feeds scripted caller speech to the orchestrator and watches the agent's
    outgoing audio so the driver knows when a reply has landed."""

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self._inbound: asyncio.Queue[AudioChunk | None] = asyncio.Queue()
        self._agent_audio_at = 0.0
        self._agent_bytes = 0
        self._loop = asyncio.get_event_loop()

    def inbound(self) -> AsyncIterator[AudioChunk]:
        return self._drain()

    async def _drain(self) -> AsyncIterator[AudioChunk]:
        while True:
            chunk = await self._inbound.get()
            if chunk is None:
                return
            yield chunk

    async def send(self, chunk: AudioChunk) -> None:
        self._agent_audio_at = self._loop.time()
        self._agent_bytes += len(chunk.data)

    async def stop_playback(self) -> None:
        self._agent_bytes = 0

    async def close(self) -> None:
        await self._inbound.put(None)

    def mark_turn(self) -> None:
        self._agent_bytes = 0
        self._agent_audio_at = 0.0

    async def caller_says(self, text: str, *, interrupt: bool = False) -> None:
        words = text.split()
        per_word = speaking_duration(text) / max(len(words), 1)
        for i, word in enumerate(words):
            samples = int(per_word * self.sample_rate)
            await self._inbound.put(
                AudioChunk(
                    SILENCE * samples,
                    self.sample_rate,
                    per_word,
                    transcript_hint=word,
                    final_hint=(i == len(words) - 1),
                )
            )
            await asyncio.sleep(per_word)

    async def wait_for_reply(self, quiet_gap: float = 0.4, timeout: float = 12.0) -> None:
        start = self._loop.time()
        while self._loop.time() - start < timeout:
            await asyncio.sleep(0.1)
            idle = self._loop.time() - self._agent_audio_at
            if self._agent_bytes > 0 and idle >= quiet_gap:
                return

    async def wait_for_agent_start(self, timeout: float = 8.0) -> None:
        start = self._loop.time()
        while self._loop.time() - start < timeout:
            if self._agent_bytes > 0:
                return
            await asyncio.sleep(0.05)
