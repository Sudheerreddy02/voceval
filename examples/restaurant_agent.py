"""Bella Vista, a phone host for a small restaurant.

Live mode drives this with a real LLM. Mock mode uses the rule-based responder
below so the eval suite stays deterministic and needs no keys.
"""

from __future__ import annotations

import json
import re

from voceval import VoiceAgent
from voceval.config import Settings
from voceval.tools.registry import Tool, ToolRegistry
from voceval.types import Message, ToolCall

SYSTEM_PROMPT = """You are the phone host for Bella Vista, an Italian restaurant.
Take reservations, answer questions about hours, and hand off anything else to a
human host. Confirm a booking only after the booking tool succeeds. Keep replies
to one or two sentences."""

GREETING = "Thanks for calling Bella Vista, this is the host. How can I help?"

HOURS = "We're open five to ten, Tuesday through Sunday, closed Mondays."

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "a couple": 2,
}


def check_availability(date: str, time: str, party_size: int) -> dict:
    taken = party_size > 8 or time in {"20:00", "8pm"}
    return {
        "available": not taken,
        "alternatives": ["17:30", "21:00"] if taken else [],
    }


def book_reservation(name: str, date: str, time: str, party_size: int) -> dict:
    ref = "BV-" + str(abs(hash((name, date, time))) % 10000).zfill(4)
    return {"confirmed": True, "reference": ref, "name": name, "time": time}


def restaurant_hours() -> str:
    return HOURS


def transfer_to_host() -> str:
    return "Connecting you to a host now."


def tools() -> ToolRegistry:
    return ToolRegistry(
        [
            Tool(
                "check_availability",
                "Check whether a table is free for a date, time and party size.",
                {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string"},
                        "time": {"type": "string"},
                        "party_size": {"type": "integer"},
                    },
                    "required": ["date", "time", "party_size"],
                },
                check_availability,
            ),
            Tool(
                "book_reservation",
                "Book a table once availability is confirmed.",
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "date": {"type": "string"},
                        "time": {"type": "string"},
                        "party_size": {"type": "integer"},
                    },
                    "required": ["name", "date", "time", "party_size"],
                },
                book_reservation,
            ),
            Tool(
                "restaurant_hours",
                "Return opening hours.",
                {"type": "object", "properties": {}},
                restaurant_hours,
            ),
            Tool(
                "transfer_to_host",
                "Hand the call to a human host.",
                {"type": "object", "properties": {}},
                transfer_to_host,
            ),
        ]
    )


def _find_party_size(text: str) -> int | None:
    for word, value in _NUMBER_WORDS.items():
        if word in text:
            return value
    m = re.search(r"\b(\d{1,2})\b(?!\s*(?:pm|am|:))", text)
    return int(m.group(1)) if m and int(m.group(1)) <= 20 else None


def _find_time(text: str) -> str | None:
    m = re.search(r"\b(\d{1,2})\s*(pm|am)\b", text)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    for word, value in _NUMBER_WORDS.items():
        if word in text and ("pm" in text or "evening" in text or "tonight" in text):
            return f"{value}pm"
    return None


def _find_name(text: str) -> str | None:
    m = re.search(r"\b(?:name is|it's|it is|under|for)\s+([A-Z][a-z]+)", text)
    if m:
        return m.group(1)
    m = re.search(r"(?:\b[A-Za-z][-\s]){2,}\b[A-Za-z]\b", text)
    if m:
        return re.sub(r"[-\s]", "", m.group(0)).capitalize()
    return None


class RestaurantResponder:
    """A tiny state machine that stands in for the LLM in mock runs."""

    def __call__(self, messages: list[Message]) -> str | ToolCall:
        last = messages[-1]
        if last.role == "tool":
            try:
                payload = json.loads(last.content)
            except ValueError:
                return f"{last.content} Anything else?"
            return self._after_tool(messages, payload)

        if _already_booked(messages):
            return "You're all set. Have a good evening."

        user = _recent_user(messages).lower()
        if any(w in user for w in ("hour", "open", "close")):
            return ToolCall("c_hours", "restaurant_hours", {})
        if any(w in user for w in ("gift card", "job", "catering", "complaint", "manager")):
            return ToolCall("c_transfer", "transfer_to_host", {})

        state = _collect(messages)
        if state["party_size"] is None:
            return "Happy to help. How many people, and what time?"
        if state["time"] is None:
            return "Got it. What time would you like?"
        if state["name"] is None:
            return "And what name should I put it under?"
        return ToolCall(
            "c_check",
            "check_availability",
            {"date": "today", "time": state["time"], "party_size": state["party_size"]},
        )

    def _after_tool(self, messages: list[Message], result: dict) -> str | ToolCall:
        if "available" in result:
            if not result["available"]:
                alts = ", ".join(result.get("alternatives", []))
                return f"That slot is full. I could do {alts} instead, would either work?"
            state = _collect(messages)
            return ToolCall(
                "c_book",
                "book_reservation",
                {
                    "name": state["name"],
                    "date": "today",
                    "time": state["time"],
                    "party_size": state["party_size"],
                },
            )
        if result.get("confirmed"):
            return (
                f"You're booked for {result['time']} under {result['name']}. "
                f"The reference is {result['reference']}. See you then."
            )
        return "All set. Anything else?"


def _recent_user(messages: list[Message]) -> str:
    return next((m.content for m in reversed(messages) if m.role == "user"), "")


def _already_booked(messages: list[Message]) -> bool:
    return any(m.role == "tool" and '"confirmed": true' in m.content for m in messages)


def _collect(messages: list[Message]) -> dict:
    users = [m.content for m in messages if m.role == "user"]
    state = {"party_size": None, "time": None, "name": None}
    for text in users:
        low = text.lower()
        state["party_size"] = state["party_size"] or _find_party_size(low)
        state["time"] = state["time"] or _find_time(low)
    for text in users[1:]:
        name = _find_name(text)
        if name:
            state["name"] = state["name"] or name
    return state


def build_agent(settings: Settings | None = None) -> VoiceAgent:
    settings = settings or Settings.load()
    return VoiceAgent(
        system_prompt=SYSTEM_PROMPT,
        tools=tools(),
        responder=RestaurantResponder(),
        greeting=GREETING,
        settings=settings,
    )
