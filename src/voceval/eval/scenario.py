from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class CallerTurn:
    text: str
    interrupt: bool = False


@dataclass
class Check:
    kind: str
    params: dict = field(default_factory=dict)


@dataclass
class Expect:
    task_success: bool = True
    max_p95_response_latency: float | None = None
    max_interruption_response: float | None = None
    checks: list[Check] = field(default_factory=list)


@dataclass
class Scenario:
    name: str
    description: str
    agent_entrypoint: str
    caller_persona: str
    caller_goal: str
    script: list[CallerTurn]
    expect: Expect
    path: Path


def _parse_turn(raw: object) -> CallerTurn:
    if isinstance(raw, str):
        return CallerTurn(raw)
    if isinstance(raw, dict):
        return CallerTurn(raw["text"], bool(raw.get("interrupt", False)))
    raise ValueError(f"bad caller turn: {raw!r}")


def _parse_expect(raw: dict) -> Expect:
    checks = [
        Check(c["kind"], {k: v for k, v in c.items() if k != "kind"})
        for c in raw.get("checks", [])
    ]
    return Expect(
        task_success=bool(raw.get("task_success", True)),
        max_p95_response_latency=raw.get("max_p95_response_latency"),
        max_interruption_response=raw.get("max_interruption_response"),
        checks=checks,
    )


def load_scenario(path: str | Path) -> Scenario:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    caller = data.get("caller", {})
    return Scenario(
        name=data["name"],
        description=data.get("description", ""),
        agent_entrypoint=data["agent"]["entrypoint"],
        caller_persona=caller.get("persona", ""),
        caller_goal=caller.get("goal", ""),
        script=[_parse_turn(t) for t in caller.get("script", [])],
        expect=_parse_expect(data.get("expect", {})),
        path=path,
    )


def load_suite(directory: str | Path) -> list[Scenario]:
    directory = Path(directory)
    files = sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.yml"))
    return [load_scenario(f) for f in files]
