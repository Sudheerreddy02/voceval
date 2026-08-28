from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Speaker(str, Enum):
    CALLER = "caller"
    AGENT = "agent"


@dataclass
class AudioChunk:
    """A slice of PCM audio. In simulation the bytes are silence and the real
    content travels in transcript_hint, which live STT providers ignore."""

    data: bytes
    sample_rate: int
    duration: float
    transcript_hint: str | None = None
    final_hint: bool = False


@dataclass
class Transcript:
    text: str
    is_final: bool
    confidence: float = 1.0
    at: float = field(default_factory=time.monotonic)


@dataclass
class Message:
    role: str  # system | user | assistant | tool
    content: str
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    tool_call_id: str
    content: str


@dataclass
class Turn:
    speaker: Speaker
    text: str
    started_at: float
    ended_at: float
    interrupted: bool = False

    @property
    def duration(self) -> float:
        return self.ended_at - self.started_at


@dataclass
class LLMDelta:
    text: str = ""
    tool_call: ToolCall | None = None
    done: bool = False
