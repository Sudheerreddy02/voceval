from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from voceval.types import AudioChunk, LLMDelta, Message, ToolCall, Transcript


class VAD(ABC):
    @abstractmethod
    def is_speech(self, chunk: AudioChunk) -> bool: ...


class STT(ABC):
    @abstractmethod
    def transcribe(self, audio: AsyncIterator[AudioChunk]) -> AsyncIterator[Transcript]:
        """Consume caller audio, yield partial transcripts and one final per utterance."""


class LLM(ABC):
    @abstractmethod
    def complete(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> AsyncIterator[LLMDelta]:
        """Yield response text token by token, plus tool calls as they resolve."""


class TTS(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        """Yield audio for one utterance. First yield is time-to-first-audio."""


def tool_schema(name: str, description: str, parameters: dict) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
    }


__all__ = ["VAD", "STT", "LLM", "TTS", "ToolCall", "tool_schema"]
