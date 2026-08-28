from __future__ import annotations

from dataclasses import dataclass, field

from voceval.config import Settings
from voceval.pipeline.factory import build_providers
from voceval.pipeline.mock import Responder
from voceval.pipeline.orchestrator import Orchestrator
from voceval.tools.registry import ToolRegistry


@dataclass
class VoiceAgent:
    system_prompt: str
    tools: ToolRegistry = field(default_factory=ToolRegistry)
    responder: Responder | None = None
    greeting: str | None = None
    settings: Settings = field(default_factory=Settings.load)

    def orchestrator(self) -> Orchestrator:
        vad, stt, llm, tts = build_providers(self.settings, self.responder)
        return Orchestrator(
            vad=vad,
            stt=stt,
            llm=llm,
            tts=tts,
            tools=self.tools,
            system_prompt=self.system_prompt,
            greeting=self.greeting,
            sample_rate=self.settings.sample_rate,
        )
