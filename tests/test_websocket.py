import json

import pytest

from voceval.config import Settings

pytest.importorskip("websockets")

from websockets.asyncio.client import connect  # noqa: E402
from websockets.asyncio.server import serve  # noqa: E402

from voceval.transport.websocket import _handle  # noqa: E402


async def _collect_turns(ws, limit: int) -> list[dict]:
    turns: list[dict] = []
    async for raw in ws:
        message = json.loads(raw)
        if message["type"] == "turn":
            turns.append(message)
        if len(turns) >= limit:
            return turns
    return turns


async def test_browser_client_gets_turns_back():
    settings = Settings()

    async def handler(ws):
        await _handle(ws, "examples/restaurant_agent.py", settings)

    async with (
        serve(handler, "127.0.0.1", 8799),
        connect("ws://127.0.0.1:8799") as ws,
    ):
        await ws.send(json.dumps({"type": "say", "text": "are you open on mondays?"}))
        turns = await _collect_turns(ws, limit=3)

    speakers = {t["speaker"] for t in turns}
    assert "agent" in speakers
    assert any("five to ten" in t["text"] or "closed" in t["text"] for t in turns)
