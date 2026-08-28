from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from voceval.types import AudioChunk


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
