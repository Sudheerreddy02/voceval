from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# event kinds recorded during a conversation
CALLER_SPEECH_START = "caller_speech_start"
CALLER_SPEECH_END = "caller_speech_end"
STT_FINAL = "stt_final"
LLM_START = "llm_start"
LLM_FIRST_TOKEN = "llm_first_token"
LLM_END = "llm_end"
TOOL_CALL = "tool_call"
TTS_FIRST_AUDIO = "tts_first_audio"
AGENT_SPEECH_START = "agent_speech_start"
AGENT_SPEECH_END = "agent_speech_end"
BARGE_IN = "barge_in"
AGENT_STOPPED = "agent_stopped"


@dataclass
class Event:
    kind: str
    at: float
    turn: int
    meta: dict[str, Any] = field(default_factory=dict)


class Timeline:
    def __init__(self) -> None:
        self.events: list[Event] = []
        self._t0 = time.monotonic()

    def now(self) -> float:
        return time.monotonic() - self._t0

    def mark(self, kind: str, turn: int, at: float | None = None, **meta: Any) -> None:
        self.events.append(Event(kind, self.now() if at is None else at, turn, meta))

    def of_kind(self, kind: str, turn: int | None = None) -> list[Event]:
        return [
            e for e in self.events if e.kind == kind and (turn is None or e.turn == turn)
        ]

    def first(self, kind: str, turn: int | None = None) -> Event | None:
        found = self.of_kind(kind, turn)
        return found[0] if found else None

    def turns(self) -> list[int]:
        return sorted({e.turn for e in self.events})

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [
                {"kind": e.kind, "at": round(e.at, 4), "turn": e.turn, "meta": e.meta}
                for e in self.events
            ]
        }
