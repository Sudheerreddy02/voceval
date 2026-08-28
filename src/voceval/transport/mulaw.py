from __future__ import annotations

# G.711 mu-law, the encoding Twilio Media Streams uses (8 kHz, 8-bit).

_BIAS = 0x84
_CLIP = 32635


def encode(pcm16: bytes) -> bytes:
    out = bytearray(len(pcm16) // 2)
    for i in range(len(out)):
        sample = int.from_bytes(pcm16[2 * i : 2 * i + 2], "little", signed=True)
        sign = 0x80 if sample < 0 else 0
        if sample < 0:
            sample = -sample
        sample = min(sample + _BIAS, _CLIP)

        exponent = 7
        mask = 0x4000
        while exponent > 0 and not sample & mask:
            exponent -= 1
            mask >>= 1

        mantissa = (sample >> (exponent + 3)) & 0x0F
        out[i] = ~(sign | (exponent << 4) | mantissa) & 0xFF
    return bytes(out)


def decode(ulaw: bytes) -> bytes:
    out = bytearray(len(ulaw) * 2)
    for i, byte in enumerate(ulaw):
        byte = ~byte & 0xFF
        sign = byte & 0x80
        exponent = (byte >> 4) & 0x07
        mantissa = byte & 0x0F
        sample = ((mantissa << 3) + _BIAS) << exponent
        sample -= _BIAS
        if sign:
            sample = -sample
        out[2 * i : 2 * i + 2] = int(sample).to_bytes(2, "little", signed=True)
    return bytes(out)
