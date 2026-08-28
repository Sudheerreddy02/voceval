# How it fits together

## The runtime

A call is a loop between two coroutines that share an `Orchestrator`.

`_read_inbound` owns the caller's audio. It runs every incoming frame through the
VAD, decides when a caller turn starts and ends, and forwards frames to the STT
queue. If the caller starts talking while the agent is still speaking, this is
where the barge-in fires: the current speak task is cancelled, the channel is
told to drop whatever audio it still has buffered, and the timeline gets a
`barge_in` / `agent_stopped` pair so the interruption can be measured later.

The main coroutine consumes STT results. Partial transcripts are ignored for
control flow; a final transcript ends the caller's turn, gets appended to the
message history, and starts a response task.

The response task streams tokens from the LLM. Tool calls are run inline and fed
back to the model, up to a small step limit. Text is buffered until a sentence
boundary and then handed to the TTS a sentence at a time, so the agent starts
talking before the model has finished writing.

Everything a metric needs is a timestamped event on the `Timeline`:
`caller_speech_end`, `llm_start`, `llm_first_token`, `tts_first_audio`,
`agent_speech_end`, `barge_in`, `agent_stopped`. The runtime never computes a
latency itself; `tracing/metrics.py` derives all of them from the event list
after the call.

## Providers

`VAD`, `STT`, `LLM` and `TTS` are four small interfaces in `pipeline/base.py`.
Each has a mock implementation that sleeps through a modeled cost (endpointing
delay, time to first token, time to first audio) and a live implementation that
calls a real service. `pipeline/factory.py` picks the set based on
`VOCEVAL_PROVIDER_PROFILE` and which keys are present.

The mock STT is the one oddity: in a simulated call the audio frames carry the
caller's words in `transcript_hint`, and the mock reads them back. Live STT
ignores that field.

## The eval harness

`Channel` is the seam. A real call uses a transport channel (local audio, a
WebSocket, Twilio). An eval run uses `SimulatedChannel`, which turns scripted
caller lines into paced audio frames and watches the agent's outgoing audio so
the driver knows when a reply has landed.

`Conversation` drives a scenario one of two ways. With the script it greets, then
for each caller line speaks it and waits for the response, except when the next
line is marked as an interruption, in which case it starts talking partway
through the agent's turn. With `PersonaCaller` it asks a live LLM for the next
caller line given the transcript so far, and stops when that model replies
`[end]` or the turn budget runs out.

`scorers.py` runs the scenario's checks against the transcript, the tool calls
and the metrics. `regression.py` compares a run to a saved baseline and reports
scenarios that started failing or got materially slower. The CLI wires these into
`voceval eval`, which is what CI runs.

## Time

Mock sleeps go through `clock.sleep`, which multiplies by a scale factor. CI runs
at 1.0 so report latencies are real wall-clock numbers. Local iteration can set
`VOCEVAL_TIME_SCALE=0.1` to run the whole suite in a few seconds.
