from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from voceval.types import AudioChunk
from voceval.util import feed_utterance


class Channel(ABC):
    """Two-way audio between the caller and the agent. The orchestrator reads
    caller audio from inbound() and writes agent audio to send()."""

    @abstractmethod
    def inbound(self) -> AsyncIterator[AudioChunk]: ...

    @abstractmethod
    async def send(self, chunk: AudioChunk) -> None: ...

    @abstractmethod
    async def stop_playback(self) -> None:
        """Drop any agent audio still queued for the caller. Called on barge-in."""

    async def close(self) -> None:
        return None


class QueueChannel(Channel):
    """A channel backed by an inbound queue. Whatever produces caller audio
    (a script, a websocket, a phone line) pushes frames onto the queue; this
    class handles the orchestrator side and tracks the agent's outgoing audio
    so a caller can tell when the agent is talking."""

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self._inbound: asyncio.Queue[AudioChunk | None] = asyncio.Queue()
        self._agent_audio_at = 0.0
        self._agent_bytes = 0

    def inbound(self) -> AsyncIterator[AudioChunk]:
        return self._drain()

    async def _drain(self) -> AsyncIterator[AudioChunk]:
        while True:
            chunk = await self._inbound.get()
            if chunk is None:
                return
            yield chunk

    async def push(self, chunk: AudioChunk) -> None:
        await self._inbound.put(chunk)

    async def send(self, chunk: AudioChunk) -> None:
        self._agent_audio_at = asyncio.get_event_loop().time()
        self._agent_bytes += len(chunk.data)

    async def stop_playback(self) -> None:
        self._agent_bytes = 0

    async def close(self) -> None:
        await self._inbound.put(None)

    def mark_turn(self) -> None:
        self._agent_bytes = 0
        self._agent_audio_at = 0.0

    def agent_speaking(self) -> bool:
        return self._agent_bytes > 0

    async def say_as_caller(self, text: str) -> None:
        await feed_utterance(self._inbound.put, text, self.sample_rate)

    async def wait_for_reply(self, quiet_gap: float = 0.4, timeout: float = 12.0) -> None:
        loop = asyncio.get_event_loop()
        start = loop.time()
        while loop.time() - start < timeout:
            await asyncio.sleep(0.1)
            if self._agent_bytes > 0 and loop.time() - self._agent_audio_at >= quiet_gap:
                return

    async def wait_for_agent_start(self, timeout: float = 8.0) -> None:
        loop = asyncio.get_event_loop()
        start = loop.time()
        while loop.time() - start < timeout:
            if self._agent_bytes > 0:
                return
            await asyncio.sleep(0.05)
