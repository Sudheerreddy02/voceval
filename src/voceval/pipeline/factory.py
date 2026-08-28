from __future__ import annotations

from voceval.config import Settings
from voceval.pipeline.base import LLM, STT, TTS, VAD
from voceval.pipeline.mock import MockLLM, MockSTT, MockTTS, MockVAD, Responder
from voceval.pipeline.responders import echo


def build_providers(
    settings: Settings, responder: Responder | None = None
) -> tuple[VAD, STT, LLM, TTS]:
    if not settings.live:
        return _mock_stack(settings, responder)

    return (
        MockVAD(),
        _live_stt(settings),
        _live_llm(settings),
        _live_tts(settings),
    )


def _mock_stack(
    settings: Settings, responder: Responder | None
) -> tuple[VAD, STT, LLM, TTS]:
    return (
        MockVAD(),
        MockSTT(),
        MockLLM(responder or echo),
        MockTTS(sample_rate=settings.sample_rate),
    )


def _live_stt(settings: Settings) -> STT:
    if not settings.deepgram_api_key:
        raise RuntimeError("live profile needs DEEPGRAM_API_KEY (or set profile to mock)")
    from voceval.pipeline.live.deepgram import DeepgramSTT

    return DeepgramSTT(settings.deepgram_api_key, settings.stt_model, settings.sample_rate)


def _live_llm(settings: Settings) -> LLM:
    if not settings.openai_api_key:
        raise RuntimeError("live profile needs OPENAI_API_KEY (or set profile to mock)")
    from voceval.pipeline.live.openai_llm import OpenAILLM

    return OpenAILLM(settings.openai_api_key, settings.llm_model)


def _live_tts(settings: Settings) -> TTS:
    if settings.cartesia_api_key:
        from voceval.pipeline.live.cartesia import CartesiaTTS

        return CartesiaTTS(settings.cartesia_api_key, settings.tts_voice, settings.sample_rate)
    if settings.elevenlabs_api_key:
        from voceval.pipeline.live.elevenlabs import ElevenLabsTTS

        return ElevenLabsTTS(settings.elevenlabs_api_key, settings.tts_voice, settings.sample_rate)
    raise RuntimeError("live profile needs CARTESIA_API_KEY or ELEVENLABS_API_KEY")
