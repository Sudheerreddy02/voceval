# Latency

The number that matters for a phone agent is the gap between the caller
finishing a sentence and the agent starting to talk. voceval calls this
`response_latency` and measures it per turn as `tts_first_audio - caller_speech_end`.

## What goes into it

For a turn with no tool call:

| stage | mock cost | where it's set |
| --- | --- | --- |
| endpointing (silence -> final transcript) | 150 ms | `MockSTT.final_latency` |
| LLM time to first token | 300 ms | `MockLLM.first_token_latency` |
| first sentence forming | a few tokens | sentence split in the orchestrator |
| TTS time to first audio | 120 ms | `MockTTS.time_to_first_audio` |

A turn that calls a tool pays the LLM round trip twice plus the tool's own time,
which is why the booking turns in the restaurant suite sit higher than the
question-answering ones.

## Baseline (real time, mock providers)

From `voceval eval --update-baseline` at `VOCEVAL_TIME_SCALE=1.0`:

| scenario | p50 | p95 |
| --- | --- | --- |
| asks_about_hours | ~0.90s | ~1.14s |
| happy_path_booking | ~0.73s | ~1.26s |
| caller_interrupts | ~0.64s | ~1.27s |
| table_unavailable | ~0.83s | ~0.98s |
| out_of_scope | ~1.06s | ~1.06s |

The regression gate flags a scenario when its p95 goes more than 20% (and at
least 150 ms) above the baseline, or when a scenario that used to pass its checks
starts failing.

## Live providers

Swap in real STT/LLM/TTS with a `.env` and `VOCEVAL_PROVIDER_PROFILE=live`. The
model choice dominates: a small fast model keeps first-token under 400 ms, a
larger one can push the whole turn past 1.5s. The eval suite runs the same way
against either.
