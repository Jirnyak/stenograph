#!/usr/bin/env python3
"""
STENOGRAPH — Universal Matrix Cipher

Everything is an image. Images are matrices. Matrices multiply.

    encrypt:  X · K = Y
    decrypt:  Y · K⁻¹ = X     (K orthogonal → K⁻¹ = Kᵀ)

Key is a SEPARATE file. User generates it once, shares it however they want.
Input = one PNG. Output = one PNG. Nothing else. No metadata. No headers.
Pure matrix multiplication.

Usage:
    python steno.py keygen                          # → key.npy + key.png
    python steno.py encrypt  image.png [output]     # image × K → encrypted.png
    python steno.py decrypt  image.png [output]     # image × Kᵀ → decrypted.png
    python steno.py audio2img music.mp3 [output]    # audio → image
    python steno.py img2audio image.png [out.wav]   # image → audio
    python steno.py text2img "hello" [output]       # text → image
    python steno.py img2text image.png              # image → text
"""

import sys, os, struct, zlib
import numpy as np
from PIL import Image

# ─── Config ──────────────────────────────────────────────────────────────────

BLOCK       = 32
KEY_FILE    = "key.npy"
KEY_VIS     = "key.png"
SAFETY      = 0.85
IMG_SIZE    = 1024          # standard image size for text/audio encoding
LOG_MAG_MAX = 20.0

TEXT_MAGIC  = b"TX"         # in-pixel marker for text content
AUDIO_MAGIC = b"AU"         # in-pixel marker for audio content
BIT_ZERO    = 88            # pixel value for bit=0 (centered ±40 from 128)
BIT_ONE     = 168           # pixel value for bit=1 (survives JPEG + cipher)
TEXT_REPS   = 25            # spatial repetition factor for noise resilience

# Primes for spatial interleaving (all coprime to IMG_SIZE² // TEXT_REPS)
_TEXT_PRIMES = [104729, 7919, 65537, 15485863, 324517, 49979687, 611953,
                2097169, 3145739, 4194319, 5242907, 6291469, 7340033,
                8388617, 9437189, 10485767, 11534351, 12582917, 13631489,
                14680067, 15728641, 16777259, 17825801, 18874379, 19922947]

# JPEG YCbCr luminance weights — G dominates luminance, JPEG preserves it best
_LUM_W = np.array([0.299, 0.587, 0.114])

# ─── Image I/O (no metadata, ever) ──────────────────────────────────────────

def save_png(path, arr):
    Image.fromarray(arr).save(path)

def load_png(path):
    return np.array(Image.open(path).convert("RGB"))

# ─── Fixed-range float ↔ uint8 ──────────────────────────────────────────────

def _map_to_u8(m, lo, hi):
    rng = hi - lo
    if rng < 1e-15:
        return np.full(m.shape, 128, np.uint8)
    return np.clip((m - lo) / rng * 247 + 4, 0, 255).astype(np.uint8)

def _map_from_u8(img, lo, hi):
    return (img.astype(np.float64) - 4) / 247 * (hi - lo) + lo

# ─── Key ─────────────────────────────────────────────────────────────────────

def _make_ortho(n):
    Q, R = np.linalg.qr(np.random.randn(n, n))
    Q *= np.sign(np.diag(R))
    return Q

def keygen(size=BLOCK):
    """Generate orthogonal key: key.npy (exact math) + key.png (visual)."""
    K = np.stack([_make_ortho(size) for _ in range(3)])  # (3, B, B)
    np.save(KEY_FILE, K)

    # Visual: tile to 1024×1024
    r = IMG_SIZE // size
    chs = []
    for c in range(3):
        tile = np.tile(_map_to_u8(K[c], K[c].min(), K[c].max()), (r, r))[:IMG_SIZE, :IMG_SIZE]
        chs.append(tile)
    save_png(KEY_VIS, np.stack(chs, -1))

    for c in range(3):
        err = np.max(np.abs(K[c] @ K[c].T - np.eye(size)))
        print(f"  ch{c}: ||K·Kᵀ - I||∞ = {err:.2e}")
    print(f"Key: {KEY_FILE} ({size}×{size}×3)  Visual: {KEY_VIS}")

def load_key():
    return np.load(KEY_FILE)

# ─── Block cipher (pure matrix multiply, nothing else) ──────────────────────

def _ensure_rgb(img):
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    return img[:, :, :3]

def _pad(img, B):
    h, w = img.shape[:2]
    ph, pw = (B - h % B) % B, (B - w % B) % B
    if ph or pw:
        img = np.pad(img, ((0, ph), (0, pw), (0, 0)), constant_values=128)
    return img

def _block_mul(X, K, B):
    """X × K per block. Pure matrix multiplication."""
    H, W = X.shape
    nH, nW = H // B, W // B
    blocks = X.astype(np.float64).reshape(nH, B, nW, B).transpose(0, 2, 1, 3).reshape(-1, B, B)
    mu = blocks.mean(axis=(1, 2), keepdims=True)
    out = (blocks - mu) * SAFETY @ K + mu
    return out.reshape(nH, nW, B, B).transpose(0, 2, 1, 3).reshape(H, W)

def _block_mul_inv(Y, Kt, B):
    """Y × Kᵀ per block. Inverse of _block_mul."""
    H, W = Y.shape
    nH, nW = H // B, W // B
    blocks = Y.astype(np.float64).reshape(nH, B, nW, B).transpose(0, 2, 1, 3).reshape(-1, B, B)
    mu = blocks.mean(axis=(1, 2), keepdims=True)
    out = (blocks - mu) @ Kt / SAFETY + mu
    return out.reshape(nH, nW, B, B).transpose(0, 2, 1, 3).reshape(H, W)

# ─── Encrypt / Decrypt ──────────────────────────────────────────────────────
#  Pure: image in → matrix multiply → image out. Same dimensions. Nothing added.

def encrypt_image(path, output=None):
    img = _ensure_rgb(load_png(path))
    K = load_key()
    B = K.shape[1]
    padded = _pad(img, B)

    enc = np.stack([np.clip(np.round(
        _block_mul(padded[:, :, c], K[c], B)), 0, 255).astype(np.uint8)
        for c in range(3)], -1)

    out = output or _outname(path, "_enc")
    save_png(out, enc)
    print(f"Encrypted: {out} ({enc.shape[1]}×{enc.shape[0]})")

def decrypt_image(path, output=None):
    img = _ensure_rgb(load_png(path))
    K = load_key()
    B = K.shape[1]
    padded = _pad(img, B)

    dec = np.stack([np.clip(np.round(
        _block_mul_inv(padded[:, :, c], K[c].T, B)), 0, 255).astype(np.uint8)
        for c in range(3)], -1)

    out = output or _outname(path, "_dec")
    save_png(out, dec)
    print(f"Decrypted: {out} ({dec.shape[1]}×{dec.shape[0]})")

# ─── Audio ↔ Image ──────────────────────────────────────────────────────────

def _make_fmap(N, nf):
    """Deterministic log-frequency mapping."""
    fmap = np.unique(np.logspace(0, np.log10(nf), N, dtype=int).clip(0, nf - 1))
    return fmap[np.linspace(0, len(fmap) - 1, N).astype(int)]

def _bytes_to_pixels(data):
    """Bytes → pixels: 1 bit per subpixel, values 0 or 255."""
    bits = np.unpackbits(np.frombuffer(data, np.uint8))
    pad = (3 - len(bits) % 3) % 3
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, np.uint8)])
    return (bits.reshape(-1, 3) * 255).astype(np.uint8)

def _pixels_to_bytes(row, n_bytes):
    """Pixels → bytes: threshold at 128."""
    flat = row.reshape(-1)
    bits = (flat[:n_bytes * 8] > 128).astype(np.uint8)
    return np.packbits(bits).tobytes()[:n_bytes]

def load_audio(path, sr=22050, max_sec=60):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.wav':
        import wave
        with wave.open(path) as wf:
            nch, sw, osr = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
            raw = wf.readframes(min(wf.getnframes(), max_sec * osr))
        dt = {1: np.int8, 2: np.int16, 4: np.int32}.get(sw, np.int16)
        d = np.frombuffer(raw, dt).astype(np.float64) / (2 ** (sw * 8 - 1))
        if nch > 1:
            d = d.reshape(-1, nch).mean(1)
        if osr != sr:
            d = np.interp(np.linspace(0, len(d) - 1, int(len(d) * sr / osr)),
                          np.arange(len(d)), d)
    elif ext == '.mp3':
        from pydub import AudioSegment
        a = AudioSegment.from_mp3(path).set_channels(1).set_frame_rate(sr)
        if len(a) > max_sec * 1000:
            a = a[:max_sec * 1000]
        d = np.array(a.get_array_of_samples(), np.float64) / 32768
    else:
        raise ValueError(f"Unsupported: {ext}")
    return d[:max_sec * sr], sr

def save_wav(path, data, sr=22050):
    import wave
    pcm = (np.clip(data, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())

def audio_to_image(audio_path, output=None, N=IMG_SIZE):
    """Audio → image. Row 0 = binary-encoded params (pixel values, not metadata).
    Rows 1+ = spectral data with fixed scaling ranges."""
    data, sr = load_audio(audio_path)
    n_samp = len(data)
    rows = N - 1
    cols = N
    audio_len = rows * cols

    if len(data) < audio_len:
        data = np.pad(data, (0, audio_len - len(data)))
    else:
        data = np.interp(np.linspace(0, len(data) - 1, audio_len),
                         np.arange(len(data)), data)

    frames = data.reshape(rows, cols)
    spec = np.fft.rfft(frames, axis=1)
    nf = spec.shape[1]
    fmap = _make_fmap(cols, nf)

    R = _map_to_u8(np.log1p(np.abs(spec[:, fmap])), 0, LOG_MAG_MAX)
    G = _map_to_u8(np.angle(spec[:, fmap]), -np.pi, np.pi)
    rms = np.sqrt(np.mean(frames ** 2, axis=1, keepdims=True))
    Bc = _map_to_u8(np.broadcast_to(rms, (rows, cols)), 0, 1)

    # Row 0: binary-encoded sample rate + sample count (these are PIXEL VALUES)
    hdr = _bytes_to_pixels(AUDIO_MAGIC + struct.pack('>II', sr, n_samp))
    row0 = np.full((1, cols, 3), 128, np.uint8)
    row0[0, :len(hdr)] = hdr

    out = output or _outname(audio_path, "_spec", ".png")
    save_png(out, np.concatenate([row0, np.stack([R, G, Bc], -1)], axis=0))
    print(f"Audio → Image: {out} ({N}×{N})")

def image_to_audio(img_path, output=None):
    """Image → audio via iFFT. Reads binary pixel header if present."""
    img = _ensure_rgb(load_png(img_path))
    N, W = img.shape[:2]

    hdr = _pixels_to_bytes(img[0], 10)
    if hdr[:2] == AUDIO_MAGIC:
        sr, n_samp = struct.unpack('>II', hdr[2:10])
        spectral = img[1:]
    else:
        sr, n_samp = 22050, N * W
        spectral = img

    rows = spectral.shape[0]
    nf = W // 2 + 1
    fmap = _make_fmap(W, nf)

    log_mag = _map_from_u8(spectral[:, :, 0], 0, LOG_MAG_MAX)
    phase   = _map_from_u8(spectral[:, :, 1], -np.pi, np.pi)

    mag_f = np.zeros((rows, nf))
    pha_f = np.zeros((rows, nf))
    for i in range(rows):
        mag_f[i] = np.interp(np.arange(nf), fmap, log_mag[i, :len(fmap)])
        pha_f[i] = np.interp(np.arange(nf), fmap, phase[i, :len(fmap)])

    spec = np.expm1(mag_f.clip(0, 50)) * np.exp(1j * pha_f)
    audio = np.fft.irfft(spec, n=W, axis=1).ravel()[:n_samp]
    pk = np.abs(audio).max()
    if pk > 0:
        audio *= 0.95 / pk

    out = output or _outname(img_path, "_audio", ".wav")
    save_wav(out, audio, sr)
    print(f"Image → Audio: {out} ({len(audio)} samples, {sr}Hz)")

# ─── Text ↔ Image ────────────────────────────────────────────────────────────

def text_to_image(text, output="text.png", N=IMG_SIZE):
    """Text → image: zlib compressed, 15× spatially interleaved, R=G=B.
    Each bit stored at 15 different positions × 3 channels = 45 copies.
    Survives JPEG compression + matrix cipher round-trip."""
    raw = text.encode('utf-8')
    comp = zlib.compress(raw, 9)
    crc = zlib.crc32(comp) & 0xFFFFFFFF

    # Header: TX(2) + comp_len(4) + raw_len(4) + crc32(4) = 14 bytes
    payload = TEXT_MAGIC + struct.pack('<III', len(comp), len(raw), crc) + comp

    n_pixels = N * N
    seg_size = n_pixels // TEXT_REPS
    n_bits = len(payload) * 8

    if n_bits > seg_size:
        raise ValueError(f"Text too long: {n_bits} bits > {seg_size} segment capacity")

    # Random noise fill
    np.random.seed(42)
    pixel_vals = np.where(
        np.random.randint(0, 2, n_pixels), BIT_ONE, BIT_ZERO
    ).astype(np.uint8)

    # Write each bit at TEXT_REPS spatially interleaved positions
    bits = np.unpackbits(np.frombuffer(payload, np.uint8))[:n_bits]
    bit_vals = np.where(bits, BIT_ONE, BIT_ZERO).astype(np.uint8)
    indices = np.arange(n_bits, dtype=np.int64)

    for r in range(TEXT_REPS):
        positions = r * seg_size + (indices * _TEXT_PRIMES[r]) % seg_size
        pixel_vals[positions] = bit_vals

    # R=G=B: 3 independent cipher channels carry same data
    img = np.stack([pixel_vals.reshape(N, N)] * 3, axis=-1)
    save_png(output, img)
    print(f"Text → Image: {output} ({N}×{N}, {len(raw)}B → {len(comp)}B, {TEXT_REPS}× redundant)")

def image_to_text(path):
    """Extract text using soft voting: sum(pixel - 128) across 3 channels × 15 positions."""
    img = _ensure_rgb(load_png(path))
    N = img.shape[0]
    n_pixels = N * N
    seg_size = n_pixels // TEXT_REPS
    channels = img.reshape(n_pixels, 3).astype(np.float64)

    # Decode header first (14 bytes = 112 bits)
    hdr_bits = _soft_decode(channels, seg_size, 112)
    hdr = np.packbits(hdr_bits).tobytes()[:14]

    if hdr[:2] != TEXT_MAGIC:
        print("Image → Text: no TX magic found")
        return _raw_image_to_text(img)

    comp_len, raw_len, stored_crc = struct.unpack('<III', hdr[2:14])
    if comp_len < 1 or comp_len > seg_size // 8:
        return _raw_image_to_text(img)

    # Decode full payload
    total_bits = (14 + comp_len) * 8
    if total_bits > seg_size:
        return _raw_image_to_text(img)

    all_bits = _soft_decode(channels, seg_size, total_bits)
    data = np.packbits(all_bits).tobytes()

    comp_data = data[14:14 + comp_len]
    if zlib.crc32(comp_data) & 0xFFFFFFFF != stored_crc:
        print("Image → Text: CRC mismatch")
        return _raw_image_to_text(img)

    try:
        text = zlib.decompress(comp_data).decode('utf-8')
    except Exception:
        text = _raw_image_to_text(img)

    print(f"Image → Text: {len(text)} chars")
    return text

def _soft_decode(channels, seg_size, n_bits):
    """Soft-decision voting: luminance-weighted sum across all spatial copies.
    Uses Y = 0.299R + 0.587G + 0.114B because JPEG preserves luminance best."""
    indices = np.arange(n_bits, dtype=np.int64)
    votes = np.zeros(n_bits, np.float64)

    for r in range(TEXT_REPS):
        positions = r * seg_size + (indices * _TEXT_PRIMES[r]) % seg_size
        # Luminance-weighted vote: G channel dominates (JPEG preserves Y best)
        votes += _LUM_W[0] * (channels[positions, 0] - 128.0)
        votes += _LUM_W[1] * (channels[positions, 1] - 128.0)
        votes += _LUM_W[2] * (channels[positions, 2] - 128.0)

    return (votes > 0).astype(np.uint8)

def _raw_image_to_text(img):
    return ''.join(chr(b) if 32 <= b <= 126 else ('\n' if b == 10 else '')
                   for b in img.ravel()[:10000])

# ─── Util / CLI ──────────────────────────────────────────────────────────────

def _outname(p, suffix, ext=None):
    b, e = os.path.splitext(p)
    return b + suffix + (ext or e)

CMDS = {
    "keygen":    "Generate orthogonal key pair",
    "encrypt":   "Encrypt image:  encrypt <img> [out]",
    "decrypt":   "Decrypt image:  decrypt <img> [out]",
    "audio2img": "Audio → Image:  audio2img <audio> [out]",
    "img2audio": "Image → Audio:  img2audio <img> [out.wav]",
    "text2img":  "Text → Image:   text2img <text|file> [out]",
    "img2text":  "Image → Text:   img2text <img>",
}

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        print("STENOGRAPH — Universal Matrix Cipher\n")
        for c, d in CMDS.items():
            print(f"  {c:12s} {d}")
        return

    cmd, args = sys.argv[1], sys.argv[2:]

    if cmd == "keygen":
        keygen(int(args[0]) if args else BLOCK)
    elif cmd == "encrypt":
        encrypt_image(args[0], args[1] if len(args) > 1 else None)
    elif cmd == "decrypt":
        decrypt_image(args[0], args[1] if len(args) > 1 else None)
    elif cmd == "audio2img":
        audio_to_image(args[0], args[1] if len(args) > 1 else None)
    elif cmd == "img2audio":
        image_to_audio(args[0], args[1] if len(args) > 1 else None)
    elif cmd == "text2img":
        t = args[0]
        if os.path.isfile(t):
            t = open(t, encoding='utf-8').read()
        text_to_image(t, args[1] if len(args) > 1 else "text.png")
    elif cmd == "img2text":
        print("─" * 40)
        print(image_to_text(args[0]))

if __name__ == "__main__":
    main()
