from __future__ import annotations

from voceval.pipeline.base import LLM
from voceval.types import Message, Speaker, Turn

_SYSTEM = """You are a person phoning a business. Stay in character.

Persona: {persona}
What you want: {goal}

Rules:
- Reply with only what you would say out loud, one or two short sentences.
- No narration, no quotation marks, no stage directions.
- Once you have what you wanted, or it is clear the agent cannot help, reply with
  exactly [end] and nothing else."""


class PersonaCaller:
    """Plays the caller side of an eval by asking an LLM what to say next, given
    the conversation so far. Live only: needs a real model behind `llm`."""

    def __init__(self, llm: LLM, persona: str, goal: str) -> None:
        self.llm = llm
        self.persona = persona
        self.goal = goal

    async def reply(self, turns: list[Turn]) -> str | None:
        history = "\n".join(
            f"{'You' if t.speaker == Speaker.CALLER else 'Agent'}: {t.text}" for t in turns
        )
        messages = [
            Message("system", _SYSTEM.format(persona=self.persona, goal=self.goal)),
            Message("user", (history or "The agent just picked up.") + "\n\nYour next line:"),
        ]

        text = ""
        async for delta in self.llm.complete(messages):
            text += delta.text
        text = text.strip().strip('"').strip()

        if not text or text.lower() == "[end]" or "[end]" in text.lower():
            return None
        return text
