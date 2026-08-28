from __future__ import annotations

from collections.abc import AsyncIterator

from voceval.pipeline.base import TTS
from voceval.types import AudioChunk

DEFAULT_VOICE = "a0e99841-438c-4a64-b679-ae501e7d6091"


class CartesiaTTS(TTS):
    def __init__(self, api_key: str, voice: str | None, sample_rate: int) -> None:
        from cartesia import AsyncCartesia

        self._client = AsyncCartesia(api_key=api_key)
        self.voice = voice or DEFAULT_VOICE
        self.sample_rate = sample_rate

    async def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        stream = self._client.tts.sse(
            model_id="sonic-2",
            transcript=text,
            voice={"mode": "id", "id": self.voice},
            output_format={
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": self.sample_rate,
            },
        )
        async for chunk in stream:
            data = getattr(chunk, "audio", None) or b""
            if data:
                yield AudioChunk(data, self.sample_rate, len(data) / 2 / self.sample_rate)
