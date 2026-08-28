from __future__ import annotations

import asyncio
from dataclasses import dataclass

from voceval.config import Settings
from voceval.eval.conversation import Conversation, Dialogue
from voceval.eval.loader import load_agent
from voceval.eval.persona import PersonaCaller
from voceval.eval.scenario import Scenario
from voceval.eval.scorers import Score, score
from voceval.pipeline.factory import live_llm_provider
from voceval.tracing.metrics import ConversationMetrics, summarize
from voceval.tracing.timeline import Timeline


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


async def run_scenario(
    scenario: Scenario, settings: Settings | None = None, driver: str = "script"
) -> ScenarioResult:
    settings = settings or Settings.load()
    agent = load_agent(scenario.agent_entrypoint, settings)
    convo = Conversation(
        agent.orchestrator(simulated=True),
        scenario.script,
        caller=_persona_caller(scenario, settings) if driver == "persona" else None,
        scenario_name=scenario.name,
        sample_rate=settings.sample_rate,
        max_turns=scenario.max_turns,
    )
    dialogue = await convo.run()
    metrics = summarize(dialogue.timeline)
    return ScenarioResult(scenario.name, dialogue, metrics, score(scenario, dialogue, metrics))


def _errored(scenario: Scenario, exc: Exception) -> ScenarioResult:
    dialogue = Dialogue(scenario.name, [], Timeline())
    detail = f"{type(exc).__name__}: {exc}"[:200]
    scores = [Score("run_error", False, 0.0, detail)]
    return ScenarioResult(scenario.name, dialogue, summarize(dialogue.timeline), scores)


def _persona_caller(scenario: Scenario, settings: Settings) -> PersonaCaller:
    llm = live_llm_provider(settings)
    if llm is None:
        raise RuntimeError("the persona driver needs a live LLM (set OPENAI_API_KEY)")
    if not scenario.caller_goal:
        raise RuntimeError(f"scenario {scenario.name} has no caller goal for the persona driver")
    return PersonaCaller(llm, scenario.caller_persona, scenario.caller_goal)


async def run_suite(
    scenarios: list[Scenario],
    settings: Settings | None = None,
    concurrency: int = 4,
    driver: str = "script",
) -> list[ScenarioResult]:
    if driver == "persona":
        # two live LLM streams per call; run one at a time with a breather, and
        # keep going if a scenario errors out on a rate limit
        results = []
        for i, scenario in enumerate(scenarios):
            if i:
                await asyncio.sleep(4)
            try:
                results.append(await run_scenario(scenario, settings, driver))
            except Exception as exc:  # report it, don't abort the rest of the suite
                results.append(_errored(scenario, exc))
        return results

    limit = asyncio.Semaphore(concurrency)

    async def one(scenario: Scenario) -> ScenarioResult:
        async with limit:
            return await run_scenario(scenario, settings, driver)

    return list(await asyncio.gather(*(one(s) for s in scenarios)))
