from __future__ import annotations

import asyncio
import contextlib
import json

from voceval.config import Settings
from voceval.eval.loader import load_agent
from voceval.pipeline.orchestrator import Orchestrator
from voceval.tracing.metrics import summarize
from voceval.transport.base import QueueChannel

# A small JSON protocol for a browser client:
#   client -> server: {"type": "say", "text": "..."}
#   server -> client: {"type": "turn", "speaker": "...", "text": "...", "interrupted": bool}
#                     {"type": "metrics", "data": {...}}  on close


async def _emit_turns(ws, orchestrator: Orchestrator) -> None:
    seen = 0
    while True:
        await asyncio.sleep(0.05)
        for turn in orchestrator.turns[seen:]:
            await ws.send(
                json.dumps(
                    {
                        "type": "turn",
                        "speaker": turn.speaker.value,
                        "text": turn.text,
                        "interrupted": turn.interrupted,
                    }
                )
            )
        seen = len(orchestrator.turns)


async def _handle(ws, agent_entrypoint: str, settings: Settings) -> None:
    channel = QueueChannel(settings.sample_rate)
    orchestrator = load_agent(agent_entrypoint, settings).orchestrator(simulated=True)
    runner = asyncio.create_task(orchestrator.run(channel))
    emitter = asyncio.create_task(_emit_turns(ws, orchestrator))

    try:
        async for raw in ws:
            message = json.loads(raw)
            if message.get("type") == "say" and message.get("text"):
                channel.mark_turn()
                await channel.say_as_caller(message["text"])
    except json.JSONDecodeError:
        pass
    finally:
        emitter.cancel()
        await channel.close()
        runner.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await runner
        with contextlib.suppress(Exception):
            await ws.send(
                json.dumps({"type": "metrics", "data": summarize(orchestrator.timeline).to_dict()})
            )


async def serve(agent_entrypoint: str, host: str = "127.0.0.1", port: int = 8765) -> None:
    from websockets.asyncio.server import serve as ws_serve

    settings = Settings.load()

    async def handler(ws) -> None:
        await _handle(ws, agent_entrypoint, settings)

    async with ws_serve(handler, host, port):
        await asyncio.Future()
