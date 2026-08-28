import json
from dataclasses import dataclass

import pytest

from voceval.config import Settings
from voceval.eval.regression import compare, save_baseline
from voceval.eval.runner import run_scenario, run_suite
from voceval.eval.scenario import load_scenario, load_suite
from voceval.tracing.metrics import ConversationMetrics

SUITE = "scenarios/restaurant"


@pytest.fixture
def settings():
    return Settings()


async def test_happy_path_books_a_table(settings):
    result = await run_scenario(load_scenario(f"{SUITE}/happy_path_booking.yaml"), settings)
    assert result.passed
    assert "book_reservation" in result.dialogue.tool_calls
    assert result.metrics.p95_response_latency < 2.0


async def test_agent_does_not_confirm_when_full(settings):
    result = await run_scenario(load_scenario(f"{SUITE}/table_unavailable.yaml"), settings)
    assert "book_reservation" not in result.dialogue.tool_calls
    assert result.passed


async def test_whole_suite_passes(settings):
    results = await run_suite(load_suite(SUITE), settings)
    assert all(r.passed for r in results), [r.scenario for r in results if not r.passed]


@dataclass
class _FakeResult:
    scenario: str
    passed: bool
    metrics: ConversationMetrics

    def failures(self):
        return []


def _result(name, p95, passed=True):
    return _FakeResult(name, passed, ConversationMetrics(p95_response_latency=p95))


def test_regression_gate_flags_slowdown_and_new_failure(tmp_path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "scenarios": {
                    "a": {"passed": True, "p50_response_latency": 0.5, "p95_response_latency": 0.9},
                    "b": {"passed": True, "p50_response_latency": 0.5, "p95_response_latency": 0.9},
                }
            }
        )
    )

    regressions = compare(baseline, [_result("a", 1.4), _result("b", 0.95, passed=False)])
    kinds = {(r.scenario, r.kind) for r in regressions}
    assert ("a", "latency") in kinds
    assert ("b", "now_failing") in kinds


def test_regression_gate_ignores_small_jitter(tmp_path):
    baseline = tmp_path / "baseline.json"
    save_baseline([_result("a", 0.9)], baseline)
    assert compare(baseline, [_result("a", 1.0)]) == []
