import array
import math

from voceval.pipeline.vad import EnergyVAD
from voceval.types import AudioChunk


def tone(amplitude: int, sample_rate: int = 16000, ms: int = 20) -> AudioChunk:
    n = sample_rate * ms // 1000
    samples = array.array(
        "h", (int(amplitude * math.sin(i / 8)) for i in range(n))
    )
    return AudioChunk(samples.tobytes(), sample_rate, ms / 1000)


def test_silence_is_not_speech():
    assert EnergyVAD().is_speech(tone(0)) is False
    assert EnergyVAD().is_speech(AudioChunk(b"", 16000, 0.02)) is False


def test_loud_frame_is_speech():
    assert EnergyVAD().is_speech(tone(6000)) is True


def test_threshold_is_configurable():
    quiet = tone(300)
    assert EnergyVAD(threshold=100).is_speech(quiet) is True
    assert EnergyVAD(threshold=5000).is_speech(quiet) is False
