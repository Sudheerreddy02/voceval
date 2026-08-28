import struct

from voceval.transport import mulaw


def test_frame_sizes():
    pcm = b"\x00\x00" * 160  # 20 ms at 8 kHz
    encoded = mulaw.encode(pcm)
    assert len(encoded) == 160
    assert len(mulaw.decode(encoded)) == 320


def test_round_trip_stays_close():
    samples = [0, 1000, -1000, 8000, -8000, 20000, -20000]
    pcm = struct.pack(f"<{len(samples)}h", *samples)

    restored = struct.unpack(f"<{len(samples)}h", mulaw.decode(mulaw.encode(pcm)))
    for original, back in zip(samples, restored, strict=True):
        assert abs(original - back) <= max(400, abs(original) * 0.1)


def test_silence_maps_to_the_mulaw_zero_code():
    assert mulaw.encode(b"\x00\x00") == b"\xff"
