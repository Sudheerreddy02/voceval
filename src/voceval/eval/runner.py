from __future__ import annotations

import asyncio
from dataclasses import dataclass

from voceval.config import Settings
from voceval.eval.conversation import Conversation, Dialogue
from voceval.eval.loader import load_agent
from voceval.eval.scenario import Scenario
from voceval.eval.scorers import Score, score
from voceval.tracing.metrics import ConversationMetrics, summarize


@dataclass
class ScenarioResult:
    scenario: str
    dialogue: Dialogue
    metrics: ConversationMetrics
    scores: list[Score]

    @property
    def passed(self) -> bool:
        return all(s.passed for s in self.scores)

    def failures(self) -> list[Score]:
        return [s for s in self.scores if not s.passed]


async def run_scenario(scenario: Scenario, settings: Settings | None = None) -> ScenarioResult:
    settings = settings or Settings.load()
    agent = load_agent(scenario.agent_entrypoint, settings)
    convo = Conversation(
        agent.orchestrator(simulated=True),
        scenario.script,
        scenario_name=scenario.name,
        sample_rate=settings.sample_rate,
    )
    dialogue = await convo.run()
    metrics = summarize(dialogue.timeline)
    return ScenarioResult(scenario.name, dialogue, metrics, score(scenario, dialogue, metrics))


async def run_suite(
    scenarios: list[Scenario], settings: Settings | None = None, concurrency: int = 4
) -> list[ScenarioResult]:
    limit = asyncio.Semaphore(concurrency)

    async def one(scenario: Scenario) -> ScenarioResult:
        async with limit:
            return await run_scenario(scenario, settings)

    return list(await asyncio.gather(*(one(s) for s in scenarios)))
