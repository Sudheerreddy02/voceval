from __future__ import annotations

from collections.abc import AsyncIterator

from voceval.pipeline.base import TTS
from voceval.types import AudioChunk
from voceval.util import aclose_stream

DEFAULT_VOICE = "JBFqnCBsd6RMkjVDRZzb"
_SUPPORTED = {8000, 16000, 22050, 24000, 44100}


class ElevenLabsTTS(TTS):
    def __init__(self, api_key: str, voice: str | None, sample_rate: int) -> None:
        from elevenlabs.client import AsyncElevenLabs

        self._client = AsyncElevenLabs(api_key=api_key)
        self.voice = voice or DEFAULT_VOICE
        self.sample_rate = sample_rate if sample_rate in _SUPPORTED else 16000

    async def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        stream = self._client.text_to_speech.convert(
            voice_id=self.voice,
            text=text,
            model_id="eleven_turbo_v2_5",
            output_format=f"pcm_{self.sample_rate}",
        )
        try:
            async for data in stream:
                if data:
                    yield AudioChunk(data, self.sample_rate, len(data) / 2 / self.sample_rate)
        finally:
            await aclose_stream(stream)
