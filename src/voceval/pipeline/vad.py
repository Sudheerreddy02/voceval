from __future__ import annotations

import array

from voceval.pipeline.base import VAD
from voceval.types import AudioChunk


class EnergyVAD(VAD):
    """RMS energy gate for 16-bit PCM. Cheap and good enough to drive turn
    taking; swap in a model-based VAD through the same interface if needed."""

    def __init__(self, threshold: float = 500.0) -> None:
        self.threshold = threshold

    def is_speech(self, chunk: AudioChunk) -> bool:
        if not chunk.data:
            return False
        samples = array.array("h")
        samples.frombytes(chunk.data[: len(chunk.data) // 2 * 2])
        if not samples:
            return False
        rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
        return rms > self.threshold
