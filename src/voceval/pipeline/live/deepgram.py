from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

from voceval.pipeline.base import STT
from voceval.types import AudioChunk, Transcript


class DeepgramSTT(STT):
    """Streaming STT over Deepgram's realtime websocket. Deepgram sends stable
    segments as is_final and marks the end of an utterance with speech_final;
    we join the segments and only end the turn on speech_final."""

    def __init__(self, api_key: str, model: str, sample_rate: int) -> None:
        self.api_key = api_key
        self.model = model
        self.sample_rate = sample_rate

    async def transcribe(
        self, audio: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[Transcript]:
        from deepgram import AsyncDeepgramClient

        client = AsyncDeepgramClient(api_key=self.api_key)
        async with client.listen.v1.connect(
            model=self.model,
            encoding="linear16",
            sample_rate=self.sample_rate,
            channels=1,
            interim_results=True,
            punctuate=True,
            endpointing=300,
        ) as socket:
            pump = asyncio.create_task(self._pump(socket, audio))
            settled: list[str] = []
            try:
                while True:
                    message = await socket.recv()
                    if getattr(message, "type", None) != "Results":
                        continue
                    alt = message.channel.alternatives[0]
                    confidence = getattr(alt, "confidence", 1.0) or 1.0

                    if message.is_final and alt.transcript:
                        settled.append(alt.transcript)
                    if message.speech_final:
                        text = " ".join(settled).strip()
                        settled = []
                        if text:
                            yield Transcript(text, is_final=True, confidence=confidence)
                    elif alt.transcript:
                        partial = " ".join([*settled, alt.transcript]).strip()
                        yield Transcript(partial, is_final=False, confidence=confidence)
            finally:
                pump.cancel()

    async def _pump(self, socket, audio: AsyncIterator[AudioChunk]) -> None:
        try:
            async for chunk in audio:
                await socket.send_media(chunk.data)
        finally:
            with contextlib.suppress(Exception):
                await socket.send_close_stream()
