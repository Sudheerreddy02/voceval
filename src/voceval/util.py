from __future__ import annotations

from voceval import clock
from voceval.types import AudioChunk

WORDS_PER_MINUTE = 150.0
SILENCE = b"\x00\x00"


def speaking_duration(text: str) -> float:
    words = max(len(text.split()), 1)
    return words / WORDS_PER_MINUTE * 60.0


async def feed_utterance(put, text: str, sample_rate: int) -> None:
    """Turn a line of text into paced audio frames and hand them to `put`, one
    word at a time, sleeping for each word's share of the utterance."""
    words = text.split()
    per_word = speaking_duration(text) / max(len(words), 1)
    for i, word in enumerate(words):
        samples = int(per_word * sample_rate)
        await put(
            AudioChunk(
                SILENCE * samples,
                sample_rate,
                per_word,
                transcript_hint=word,
                final_hint=(i == len(words) - 1),
            )
        )
        await clock.sleep(per_word)
