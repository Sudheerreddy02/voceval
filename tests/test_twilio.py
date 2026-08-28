import asyncio
import base64
import json

from voceval.config import Settings
from voceval.transport import mulaw
from voceval.transport.twilio import _handle


class FakeTwilioWS:
    """Replays the Twilio Media Streams protocol and captures what we send back."""

    def __init__(self, events, hold: float = 1.2):
        self._events = events
        self._hold = hold
        self.sent: list[dict] = []

    def __aiter__(self):
        return self._play()

    async def _play(self):
        for event in self._events:
            yield json.dumps(event)
            await asyncio.sleep(0.02)
        await asyncio.sleep(self._hold)  # let the greeting stream back
        yield json.dumps({"event": "stop"})

    async def send(self, data):
        self.sent.append(json.loads(data))


async def test_greeting_streams_back_as_mulaw_frames():
    settings = Settings()
    settings.sample_rate = 8000
    ws = FakeTwilioWS([{"event": "start", "start": {"streamSid": "MZ0000000000"}}])

    await _handle(ws, "examples/restaurant_agent.py", settings)

    media = [m for m in ws.sent if m.get("event") == "media"]
    assert media, "expected the agent greeting to come back as media frames"
    assert all(m["streamSid"] == "MZ0000000000" for m in media)

    frame = base64.b64decode(media[0]["media"]["payload"])
    assert len(frame) == 160  # 20 ms of 8 kHz mu-law
    assert len(mulaw.decode(frame)) == 320


async def test_clear_is_sent_on_barge_in():
    settings = Settings()
    settings.sample_rate = 8000

    # a burst of non-silence while the agent is greeting should trigger a clear
    speech = mulaw.encode(b"\x40\x10" * 800)
    events = [
        {"event": "start", "start": {"streamSid": "MZ1"}},
        *(
            {"event": "media", "media": {"payload": base64.b64encode(speech).decode()}}
            for _ in range(6)
        ),
    ]
    ws = FakeTwilioWS(events, hold=1.5)

    await _handle(ws, "examples/restaurant_agent.py", settings)

    assert any(m.get("event") == "clear" for m in ws.sent)
