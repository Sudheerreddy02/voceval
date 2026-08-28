from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field

from voceval import clock
from voceval.eval.persona import PersonaCaller
from voceval.eval.scenario import CallerTurn
from voceval.eval.simulated import SimulatedChannel
from voceval.pipeline.orchestrator import Orchestrator
from voceval.tracing import timeline as tl
from voceval.tracing.timeline import Timeline
from voceval.types import Speaker, Turn


@dataclass
class Dialogue:
    scenario: str
    turns: list[Turn]
    timeline: Timeline
    tool_calls: list[str] = field(default_factory=list)

    def transcript(self) -> str:
        rows = []
        for t in self.turns:
            tag = "caller" if t.speaker == Speaker.CALLER else "agent"
            note = "  [interrupted]" if t.interrupted else ""
            rows.append(f"{tag}: {t.text}{note}")
        return "\n".join(rows)

    def agent_said(self) -> str:
        joined = " ".join(t.text for t in self.turns if t.speaker == Speaker.AGENT)
        return _normalize(joined)


_PUNCT = str.maketrans(
    {"’": "'", "‘": "'", "“": '"', "”": '"',
     "–": "-", "—": "-", "‑": "-", " ": " "}
)


def _normalize(text: str) -> str:
    return " ".join(text.translate(_PUNCT).lower().split())


class Conversation:
    def __init__(
        self,
        orchestrator: Orchestrator,
        script: list[CallerTurn] | None = None,
        *,
        caller: PersonaCaller | None = None,
        scenario_name: str = "adhoc",
        sample_rate: int = 16000,
        max_turns: int = 10,
    ) -> None:
        self.orchestrator = orchestrator
        self.script = script or []
        self.caller = caller
        self.scenario_name = scenario_name
        self.sample_rate = sample_rate
        self.max_turns = max_turns

    async def run(self) -> Dialogue:
        channel = SimulatedChannel(self.sample_rate)
        runner = asyncio.create_task(self.orchestrator.run(channel))
        try:
            await self._drive(channel)
        finally:
            await channel.close()
            runner.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await runner

        tool_calls = [
            e.meta.get("name", "") for e in self.orchestrator.timeline.of_kind(tl.TOOL_CALL)
        ]
        return Dialogue(
            scenario=self.scenario_name,
            turns=self.orchestrator.turns,
            timeline=self.orchestrator.timeline,
            tool_calls=tool_calls,
        )

    async def _drive(self, channel: SimulatedChannel) -> None:
        if self.orchestrator.greeting:
            await self._await_response()

        if self.caller is not None:
            await self._drive_persona(channel)
            return

        for i, turn in enumerate(self.script):
            following = self.script[i + 1] if i + 1 < len(self.script) else None
            channel.mark_turn()
            await channel.caller_says(turn.text)

            if following and following.interrupt:
                await channel.wait_for_agent_start()
                await clock.sleep(0.5)
            else:
                await self._await_response()

    async def _drive_persona(self, channel: SimulatedChannel) -> None:
        assert self.caller is not None
        for _ in range(self.max_turns):
            line = await self.caller.reply(list(self.orchestrator.turns))
            if line is None:
                return
            channel.mark_turn()
            await channel.caller_says(line)
            await self._await_response()

    async def _await_response(self, timeout: float = 15.0) -> None:
        for _ in range(400):
            if self.orchestrator.is_responding:
                break
            await asyncio.sleep(0.02)
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self.orchestrator.response_done.wait(), timeout)
