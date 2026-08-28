from voceval.tracing import timeline as tl
from voceval.tracing.metrics import percentile, summarize
from voceval.tracing.timeline import Timeline


def test_percentile_interpolates():
    assert percentile([10, 20, 30, 40], 0.5) == 25
    assert percentile([], 0.9) == 0.0


def build_timeline() -> Timeline:
    t = Timeline()
    t.mark(tl.CALLER_SPEECH_START, 1, at=0.0)
    t.mark(tl.CALLER_SPEECH_END, 1, at=1.0)
    t.mark(tl.LLM_START, 1, at=1.05)
    t.mark(tl.LLM_FIRST_TOKEN, 1, at=1.35)
    t.mark(tl.TTS_FIRST_AUDIO, 1, at=1.6)
    t.mark(tl.AGENT_SPEECH_START, 1, at=1.6)
    t.mark(tl.AGENT_SPEECH_END, 1, at=4.6, duration=3.0)
    t.mark(tl.CALLER_SPEECH_START, 2, at=3.0)
    t.mark(tl.BARGE_IN, 2, at=3.0)
    t.mark(tl.AGENT_STOPPED, 2, at=3.2)
    return t


def test_summarize_reads_latency_and_interruption():
    m = summarize(build_timeline())
    assert round(m.p50_response_latency, 2) == 0.6
    assert round(m.mean_llm_ttft, 2) == 0.3
    assert round(m.max_interruption_response, 2) == 0.2
    assert 0 < m.talk_over_ratio < 1
