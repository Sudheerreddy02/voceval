from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from voceval.eval.runner import ScenarioResult

# a p95 regression has to clear both bars to count, so small jitter is ignored
LATENCY_TOLERANCE_RATIO = 0.20
LATENCY_TOLERANCE_ABS = 0.15


def build_baseline(results: list[ScenarioResult], time_scale: float = 1.0) -> dict:
    return {
        "time_scale": time_scale,
        "scenarios": {
            r.scenario: {
                "passed": r.passed,
                "p50_response_latency": round(r.metrics.p50_response_latency, 3),
                "p95_response_latency": round(r.metrics.p95_response_latency, 3),
            }
            for r in results
        }
    }


def save_baseline(
    results: list[ScenarioResult], path: str | Path, time_scale: float = 1.0
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_baseline(results, time_scale), indent=2), encoding="utf-8"
    )


@dataclass
class Regression:
    scenario: str
    kind: str
    detail: str


def compare(
    baseline_path: str | Path,
    results: list[ScenarioResult],
    check_latency: bool = True,
) -> list[Regression]:
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))["scenarios"]
    found: list[Regression] = []

    for r in results:
        base = baseline.get(r.scenario)
        if base is None:
            continue
        if base["passed"] and not r.passed:
            names = ", ".join(s.name for s in r.failures())
            found.append(Regression(r.scenario, "now_failing", names))

        if not check_latency:
            continue

        was = base["p95_response_latency"]
        now = r.metrics.p95_response_latency
        allowed = was + max(was * LATENCY_TOLERANCE_RATIO, LATENCY_TOLERANCE_ABS)
        if now > allowed:
            found.append(
                Regression(
                    r.scenario,
                    "latency",
                    f"p95 {now:.2f}s up from {was:.2f}s",
                )
            )

    return found
