from __future__ import annotations

import re
from dataclasses import dataclass

from voceval.eval.conversation import Dialogue
from voceval.eval.scenario import Check, Scenario
from voceval.tracing import timeline as tl
from voceval.tracing.metrics import ConversationMetrics

# a claim that a reservation was actually made, as opposed to "we're fully booked"
_CONFIRMED_A_BOOKING = re.compile(
    r"you'?re (booked|all set)"
    r"|booked (you|your)"
    r"|(table|reservation|booking)[^.]{0,40}(is|are) (booked|confirmed|set|reserved)"
    r"|confirmed[^.]{0,25}(reference|ref |bv-)"
    r"|see you (then|tonight)",
    re.I,
)


@dataclass
class Score:
    name: str
    passed: bool
    value: float
    detail: str = ""


def _check_transcript_contains(check: Check, dialogue: Dialogue, _m) -> Score:
    said = dialogue.agent_said()
    if "any" in check.params:
        options = [str(x).lower() for x in check.params["any"]]
        hit = any(o in said for o in options)
        return Score("transcript_contains", hit, 1.0 if hit else 0.0, f"any of {options}")
    wanted = [str(x).lower() for x in check.params.get("all", [])]
    missing = [w for w in wanted if w not in said]
    return Score("transcript_contains", not missing, 0.0 if missing else 1.0, f"missing {missing}")


def _check_transcript_excludes(check: Check, dialogue: Dialogue, _m) -> Score:
    said = dialogue.agent_said()
    banned = [str(x).lower() for x in check.params.get("phrases", [])]
    found = [b for b in banned if b in said]
    return Score("transcript_excludes", not found, 0.0 if found else 1.0, f"found {found}")


def _check_tool_called(check: Check, dialogue: Dialogue, _m) -> Score:
    name = check.params["name"]
    hit = name in dialogue.tool_calls
    return Score(f"tool_called:{name}", hit, 1.0 if hit else 0.0)


def _check_tool_not_called(check: Check, dialogue: Dialogue, _m) -> Score:
    name = check.params["name"]
    hit = name not in dialogue.tool_calls
    return Score(f"tool_not_called:{name}", hit, 1.0 if hit else 0.0)


def _check_no_hallucinated_confirmation(_check, dialogue: Dialogue, _m) -> Score:
    claimed = bool(_CONFIRMED_A_BOOKING.search(dialogue.agent_said()))
    actually_booked = "book_reservation" in dialogue.tool_calls
    ok = actually_booked or not claimed
    return Score(
        "no_hallucinated_confirmation",
        ok,
        1.0 if ok else 0.0,
        "confirmed without booking" if not ok else "",
    )


def _check_caller_interrupted(_check, dialogue: Dialogue, _m) -> Score:
    hit = bool(dialogue.timeline.of_kind(tl.BARGE_IN))
    return Score("caller_interrupted_agent", hit, 1.0 if hit else 0.0)


_CHECKS = {
    "transcript_contains": _check_transcript_contains,
    "transcript_excludes": _check_transcript_excludes,
    "tool_called": _check_tool_called,
    "tool_not_called": _check_tool_not_called,
    "no_hallucinated_confirmation": _check_no_hallucinated_confirmation,
    "caller_interrupted_agent": _check_caller_interrupted,
}


def score(scenario: Scenario, dialogue: Dialogue, metrics: ConversationMetrics) -> list[Score]:
    results: list[Score] = []

    for check in scenario.expect.checks:
        fn = _CHECKS.get(check.kind)
        if fn is None:
            results.append(Score(check.kind, False, 0.0, "unknown check"))
        else:
            results.append(fn(check, dialogue, metrics))

    check_scores = list(results)
    task_ok = all(s.passed for s in check_scores)
    if not scenario.expect.task_success:
        task_ok = not task_ok
    results.insert(0, Score("task_success", task_ok, 1.0 if task_ok else 0.0))

    limit = scenario.expect.max_p95_response_latency
    if limit is not None:
        ok = metrics.p95_response_latency <= limit
        results.append(
            Score(
                "p95_response_latency",
                ok,
                1.0 if ok else 0.0,
                f"{metrics.p95_response_latency:.2f}s <= {limit}s",
            )
        )

    limit = scenario.expect.max_interruption_response
    if limit is not None:
        ok = metrics.max_interruption_response <= limit
        results.append(
            Score(
                "interruption_response",
                ok,
                1.0 if ok else 0.0,
                f"{metrics.max_interruption_response:.2f}s <= {limit}s",
            )
        )

    return results
