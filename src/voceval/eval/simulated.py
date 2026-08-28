from __future__ import annotations

from voceval.transport.base import QueueChannel


class SimulatedChannel(QueueChannel):
    """A queue channel driven by a scenario script. `say_as_caller` and the
    wait helpers on the base class are everything the conversation driver needs."""

    async def caller_says(self, text: str) -> None:
        await self.say_as_caller(text)
