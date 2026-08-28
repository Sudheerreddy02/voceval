from voceval.config import Settings
from voceval.eval.conversation import Conversation
from voceval.eval.persona import PersonaCaller
from voceval.pipeline.mock import MockLLM
from voceval.pipeline.responders import scripted
from voceval.types import Speaker, Turn


def caller_with(lines):
    return PersonaCaller(MockLLM(scripted(lines)), "a caller", "book a table")


async def test_reply_returns_the_line():
    caller = caller_with(["Hi, do you have a table for two tonight?"])
    assert await caller.reply([]) == "Hi, do you have a table for two tonight?"


async def test_reply_ends_on_the_sentinel():
    caller = caller_with(["[end]"])
    assert await caller.reply([Turn(Speaker.AGENT, "Anything else?", 0.0, 1.0)]) is None


async def test_persona_driven_booking_runs_end_to_end():
    from voceval.eval.loader import load_agent

    agent = load_agent("examples/restaurant_agent.py", Settings())

    caller = caller_with(
        [
            "Hi, I'd like to book a table for two at 7pm tonight.",
            "It's under Rivera.",
            "Great, thanks. [end]",
        ]
    )
    convo = Conversation(
        agent.orchestrator(simulated=True), caller=caller, scenario_name="persona", max_turns=6
    )
    dialogue = await convo.run()

    assert "book_reservation" in dialogue.tool_calls
    assert any(t.speaker == Speaker.CALLER for t in dialogue.turns)
