from __future__ import annotations

import asyncio
import base64
import contextlib
import json

from voceval.config import Settings
from voceval.eval.loader import load_agent
from voceval.transport import mulaw
from voceval.transport.base import QueueChannel
from voceval.types import AudioChunk

FRAME_MS = 20
SAMPLE_RATE = 8000

TWIML = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://{host}/twilio" />
  </Connect>
</Response>"""


class TwilioChannel(QueueChannel):
    """Bridges a Twilio Media Streams websocket. Twilio speaks 8 kHz mu-law in
    20 ms frames; audio is decoded to PCM16 on the way in and encoded back on the
    way out. `clear` is what makes barge-in audible on a real call."""

    def __init__(self, ws) -> None:
        super().__init__(sample_rate=SAMPLE_RATE)
        self._ws = ws
        self._stream_sid = ""
        self._out_buffer = bytearray()

    def bind_stream(self, stream_sid: str) -> None:
        self._stream_sid = stream_sid

    async def send(self, chunk: AudioChunk) -> None:
        await super().send(chunk)
        self._out_buffer += chunk.data
        frame_bytes = SAMPLE_RATE * FRAME_MS // 1000 * 2
        while len(self._out_buffer) >= frame_bytes:
            frame, self._out_buffer = (
                self._out_buffer[:frame_bytes],
                self._out_buffer[frame_bytes:],
            )
            payload = base64.b64encode(mulaw.encode(bytes(frame))).decode()
            await self._ws.send(
                json.dumps(
                    {"event": "media", "streamSid": self._stream_sid, "media": {"payload": payload}}
                )
            )

    async def stop_playback(self) -> None:
        await super().stop_playback()
        self._out_buffer.clear()
        await self._ws.send(json.dumps({"event": "clear", "streamSid": self._stream_sid}))


async def _handle(ws, agent_entrypoint: str, settings: Settings) -> None:
    channel = TwilioChannel(ws)
    orchestrator = load_agent(agent_entrypoint, settings).orchestrator()
    runner: asyncio.Task | None = None

    try:
        async for raw in ws:
            message = json.loads(raw)
            event = message.get("event")
            if event == "start":
                channel.bind_stream(message["start"]["streamSid"])
                runner = asyncio.create_task(orchestrator.run(channel))
            elif event == "media" and runner:
                pcm = mulaw.decode(base64.b64decode(message["media"]["payload"]))
                await channel.push(
                    AudioChunk(pcm, SAMPLE_RATE, len(pcm) / 2 / SAMPLE_RATE)
                )
            elif event == "stop":
                break
    finally:
        await channel.close()
        if runner:
            runner.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await runner


async def serve(agent_entrypoint: str, host: str = "0.0.0.0", port: int = 8770) -> None:
    from websockets.asyncio.server import serve as ws_serve

    settings = Settings.load()
    settings.sample_rate = SAMPLE_RATE

    async def handler(ws) -> None:
        await _handle(ws, agent_entrypoint, settings)

    async with ws_serve(handler, host, port):
        await asyncio.Future()
