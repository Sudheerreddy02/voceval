from __future__ import annotations

from voceval.config import Settings
from voceval.pipeline.base import LLM, STT, TTS, VAD
from voceval.pipeline.mock import MockLLM, MockSTT, MockTTS, MockVAD, Responder
from voceval.pipeline.responders import echo
from voceval.pipeline.vad import EnergyVAD


def build_providers(
    settings: Settings, responder: Responder | None = None, *, simulated: bool = False
) -> tuple[VAD, STT, LLM, TTS]:
    """In the mock profile everything is mocked. In the live profile each stage
    is live if its key is set and mocked otherwise, so a partial key set still
    runs. A simulated run always keeps the mock VAD and STT: the caller feeds
    text through the channel, there is no real audio to transcribe."""
    if not settings.live:
        return (
            MockVAD(),
            MockSTT(),
            MockLLM(responder or echo),
            MockTTS(sample_rate=settings.sample_rate),
        )

    llm = _live_llm(settings) or MockLLM(responder or echo)
    tts = _live_tts(settings) or MockTTS(sample_rate=settings.sample_rate)

    if simulated:
        return (MockVAD(), MockSTT(), llm, tts)

    live_stt = _live_stt(settings)
    vad: VAD = EnergyVAD() if live_stt else MockVAD()
    return (vad, live_stt or MockSTT(), llm, tts)


def _live_stt(settings: Settings) -> STT | None:
    if not settings.deepgram_api_key:
        return None
    from voceval.pipeline.live.deepgram import DeepgramSTT

    return DeepgramSTT(settings.deepgram_api_key, settings.stt_model, settings.sample_rate)


def _live_llm(settings: Settings) -> LLM | None:
    if not settings.openai_api_key:
        return None
    from voceval.pipeline.live.openai_llm import OpenAILLM

    return OpenAILLM(settings.openai_api_key, settings.llm_model, settings.openai_base_url)


def _live_tts(settings: Settings) -> TTS | None:
    if settings.cartesia_api_key:
        from voceval.pipeline.live.cartesia import CartesiaTTS

        return CartesiaTTS(settings.cartesia_api_key, settings.tts_voice, settings.sample_rate)
    if settings.elevenlabs_api_key:
        from voceval.pipeline.live.elevenlabs import ElevenLabsTTS

        return ElevenLabsTTS(settings.elevenlabs_api_key, settings.tts_voice, settings.sample_rate)
    return None
