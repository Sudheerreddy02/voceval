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
eval suite runs the same way; `simulate` and `eval` feed the agent scripted text
so they exercise the live LLM and TTS but keep the mock STT.

A run against Groq (`openai/gpt-oss-20b`) for the LLM and Cartesia (`sonic-2`)
for TTS:

| scenario | p50 | p95 |
| --- | --- | --- |
| asks_about_hours | ~1.6s | ~2.4s |
| happy_path_booking | ~1.3s | ~2.8s |
| caller_interrupts | ~1.6s | ~1.8s |
| table_unavailable | ~1.3s | ~1.3s |

Every check passes except the 1.8s p95 limit on `happy_path_booking`, which the
live stack misses at ~2.8s. That is the harness doing its job: LLM time to first
token is the cost, and the booking turn pays it twice for the two tool calls.
A faster first-token model (for example `gpt-4o-mini`) brings it back under.
CI runs the mock stack, so the badge reflects correctness, not a given
provider's latency.
