from voceval.config import Settings
from voceval.eval.conversation import Conversation
from voceval.eval.scenario import CallerTurn
from voceval.pipeline.responders import scripted
from voceval.tracing import timeline as tl
from voceval.tracing.metrics import summarize
from voceval.types import Speaker


def build_orchestrator():
    from voceval.agent import VoiceAgent

    agent = VoiceAgent(
        system_prompt="You are a test agent.",
        responder=scripted(["Sure, I can help with that.", "All set, anything else?"]),
        greeting="Thanks for calling, how can I help?",
        settings=Settings(),
    )
    return agent.orchestrator()


async def test_agent_answers_two_turns():
    convo = Conversation(
        build_orchestrator(),
        [CallerTurn("I need a hand with something."), CallerTurn("Great, that's all.")],
    )
    dialogue = await convo.run()

    agent_turns = [t for t in dialogue.turns if t.speaker == Speaker.AGENT]
    assert len(agent_turns) >= 2
    assert dialogue.timeline.of_kind(tl.STT_FINAL)

    metrics = summarize(dialogue.timeline)
    assert metrics.p50_response_latency > 0
