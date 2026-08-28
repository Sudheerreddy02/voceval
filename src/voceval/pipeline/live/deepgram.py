from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from voceval.pipeline.base import STT
from voceval.types import AudioChunk, Transcript


class DeepgramSTT(STT):
    """Streaming STT over Deepgram's realtime websocket. A final is emitted on
    speech_final so the orchestrator ends the turn on Deepgram's endpointing."""

    def __init__(self, api_key: str, model: str, sample_rate: int) -> None:
        self.api_key = api_key
        self.model = model
        self.sample_rate = sample_rate

    async def transcribe(
        self, audio: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[Transcript]:
        from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents

        results: asyncio.Queue[Transcript | None] = asyncio.Queue()
        client = DeepgramClient(self.api_key)
        connection = client.listen.asyncwebsocket.v("1")

        async def on_transcript(_conn, result, **_kwargs) -> None:
            alt = result.channel.alternatives[0]
            if not alt.transcript:
                return
            final = bool(getattr(result, "speech_final", False) or result.is_final)
            await results.put(Transcript(alt.transcript, is_final=final, confidence=alt.confidence))

        connection.on(LiveTranscriptionEvents.Transcript, on_transcript)

        await connection.start(
            LiveOptions(
                model=self.model,
                encoding="linear16",
                sample_rate=self.sample_rate,
                channels=1,
                interim_results=True,
                punctuate=True,
                endpointing=300,
            )
        )

        async def pump() -> None:
            try:
                async for chunk in audio:
                    await connection.send(chunk.data)
            finally:
                await connection.finish()
                await results.put(None)

        pumping = asyncio.create_task(pump())
        try:
            while True:
                item = await results.get()
                if item is None:
                    return
                yield item
        finally:
            pumping.cancel()
