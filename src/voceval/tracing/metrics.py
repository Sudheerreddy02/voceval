from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean

from voceval.tracing import timeline as tl
from voceval.tracing.timeline import Timeline


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


@dataclass
class TurnMetrics:
    turn: int
    response_latency: float | None = None  # caller stopped -> agent first audio
    llm_ttft: float | None = None  # llm start -> first token
    interruption_response: float | None = None  # barge-in -> agent stopped
    talk_over: float = 0.0  # agent speech that overlapped the caller


@dataclass
class ConversationMetrics:
    turns: list[TurnMetrics] = field(default_factory=list)
    p50_response_latency: float = 0.0
    p95_response_latency: float = 0.0
    mean_llm_ttft: float = 0.0
    max_interruption_response: float = 0.0
    talk_over_ratio: float = 0.0

    def to_dict(self) -> dict:
        return {
            "p50_response_latency": round(self.p50_response_latency, 3),
            "p95_response_latency": round(self.p95_response_latency, 3),
            "mean_llm_ttft": round(self.mean_llm_ttft, 3),
            "max_interruption_response": round(self.max_interruption_response, 3),
            "talk_over_ratio": round(self.talk_over_ratio, 3),
            "turns": [
                {
                    "turn": t.turn,
                    "response_latency": _r(t.response_latency),
                    "llm_ttft": _r(t.llm_ttft),
                    "interruption_response": _r(t.interruption_response),
                    "talk_over": round(t.talk_over, 3),
                }
                for t in self.turns
            ],
        }


def _r(v: float | None) -> float | None:
    return None if v is None else round(v, 3)


def _turn_metrics(timeline: Timeline, turn: int) -> TurnMetrics:
    m = TurnMetrics(turn=turn)

    caller_end = timeline.first(tl.CALLER_SPEECH_END, turn)
    first_audio = timeline.first(tl.TTS_FIRST_AUDIO, turn)
    if caller_end and first_audio:
        m.response_latency = first_audio.at - caller_end.at

    llm_start = timeline.first(tl.LLM_START, turn)
    first_token = timeline.first(tl.LLM_FIRST_TOKEN, turn)
    if llm_start and first_token:
        m.llm_ttft = first_token.at - llm_start.at

    barge = timeline.first(tl.BARGE_IN, turn)
    stopped = timeline.first(tl.AGENT_STOPPED, turn)
    if barge and stopped:
        m.interruption_response = stopped.at - barge.at
        m.talk_over = max(0.0, stopped.at - barge.at)

    return m


def summarize(timeline: Timeline) -> ConversationMetrics:
    per_turn = [_turn_metrics(timeline, t) for t in timeline.turns()]
    latencies = [t.response_latency for t in per_turn if t.response_latency is not None]
    ttfts = [t.llm_ttft for t in per_turn if t.llm_ttft is not None]
    interrupts = [
        t.interruption_response for t in per_turn if t.interruption_response is not None
    ]

    agent_speech = sum(
        e.meta.get("duration", 0.0) for e in timeline.of_kind(tl.AGENT_SPEECH_END)
    )
    talk_over = sum(t.talk_over for t in per_turn)

    return ConversationMetrics(
        turns=per_turn,
        p50_response_latency=percentile(latencies, 0.5),
        p95_response_latency=percentile(latencies, 0.95),
        mean_llm_ttft=mean(ttfts) if ttfts else 0.0,
        max_interruption_response=max(interrupts) if interrupts else 0.0,
        talk_over_ratio=(talk_over / agent_speech) if agent_speech else 0.0,
    )
