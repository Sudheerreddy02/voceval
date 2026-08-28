# Phone calls with Twilio

`voceval twilio` serves the agent over Twilio Media Streams. Twilio opens a
WebSocket to it, streams the caller's audio as 8 kHz mu-law in 20 ms frames, and
plays back whatever audio voceval sends. Barge-in sends a `clear` so the caller
actually hears the agent stop.

## What you need

- A Twilio number.
- A Deepgram key in `.env` (the phone path needs real STT; without it the agent
  greets but can't hear the caller).
- A public HTTPS/WSS URL that reaches your machine. In development, a tunnel:

  ```bash
  ngrok http 8770
  ```

## Wire it up

```bash
voceval twilio --agent examples/restaurant_agent.py --public-host <your-ngrok-host>
```

It prints the TwiML the number should return:

```xml
<Response>
  <Connect>
    <Stream url="wss://<your-ngrok-host>/twilio" />
  </Connect>
</Response>
```

In the Twilio console, set the number's **A call comes in** webhook to a URL that
serves that TwiML (a small static endpoint, or Twilio's TwiML Bins). Call the
number and you are talking to the agent.

## Notes

- Audio is decoded from mu-law to PCM16 on the way in and re-encoded on the way
  out; the codec is in `transport/mulaw.py` and has a round-trip test.
- Trial Twilio accounts can only call verified numbers and play a short trial
  notice first.
- `test_twilio.py` replays the Media Streams protocol through the handler with a
  fake socket, so the framing and barge-in path are covered without a real call.
