from __future__ import annotations

import asyncio

# Mock providers and the simulated caller sleep through modeled component costs
# (endpointing, time to first token, and so on). Scaling those sleeps lets the
# whole suite run faster than real time without changing the ratios between
# stages. CI keeps this at 1.0 so the latency numbers in reports are real.
_scale = 1.0


def set_scale(scale: float) -> None:
    global _scale
    _scale = max(scale, 0.0)


def get_scale() -> float:
    return _scale


async def sleep(seconds: float) -> None:
    await asyncio.sleep(seconds * _scale)
