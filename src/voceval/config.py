from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from voceval import clock


def _load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


@dataclass
class Settings:
    provider_profile: str = "mock"  # mock | live
    sample_rate: int = 16000
    time_scale: float = 1.0

    openai_api_key: str | None = None
    openai_base_url: str | None = None
    llm_model: str = "gpt-4o-mini"

    deepgram_api_key: str | None = None
    stt_model: str = "nova-2"

    cartesia_api_key: str | None = None
    elevenlabs_api_key: str | None = None
    tts_voice: str | None = None

    @classmethod
    def load(cls) -> Settings:
        _load_dotenv()
        settings = cls(
            provider_profile=os.getenv("VOCEVAL_PROVIDER_PROFILE", "mock"),
            sample_rate=int(os.getenv("VOCEVAL_SAMPLE_RATE", "16000")),
            time_scale=float(os.getenv("VOCEVAL_TIME_SCALE", "1.0")),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
            llm_model=os.getenv("VOCEVAL_LLM_MODEL", "gpt-4o-mini"),
            deepgram_api_key=os.getenv("DEEPGRAM_API_KEY") or None,
            stt_model=os.getenv("VOCEVAL_STT_MODEL", "nova-2"),
            cartesia_api_key=os.getenv("CARTESIA_API_KEY") or None,
            elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY") or None,
            tts_voice=os.getenv("VOCEVAL_TTS_VOICE") or None,
        )
        clock.set_scale(settings.time_scale)
        return settings

    @property
    def live(self) -> bool:
        return self.provider_profile == "live"
