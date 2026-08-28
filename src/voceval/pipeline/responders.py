from __future__ import annotations

from collections.abc import Iterator

from voceval.types import Message


def echo(messages: list[Message]) -> str:
    last = next((m.content for m in reversed(messages) if m.role == "user"), "")
    return f"You said: {last}" if last else "I'm listening."


def scripted(lines: list[str]):
    it: Iterator[str] = iter(lines)

    def responder(_messages: list[Message]) -> str:
        return next(it, "Sorry, I don't have anything else to add.")

    return responder
