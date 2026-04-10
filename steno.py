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
    """Text → image: magic + lengths + zlib, all as binary 0/255 pixels."""
    raw = text.encode('utf-8')
    comp = zlib.compress(raw, 9)
    payload = TEXT_MAGIC + struct.pack('<II', len(comp), len(raw)) + comp

    total_bits = N * N * 3
    if len(payload) > total_bits // 8:
        raise ValueError(f"Text too long: {len(payload)}B > {total_bits // 8}B")

    bits = np.unpackbits(np.frombuffer(payload, np.uint8))
    pixels = np.zeros(total_bits, np.uint8)
    pixels[:len(bits)] = bits * 255

    n_fill = total_bits - len(bits)
    if n_fill > 0:
        np.random.seed(zlib.crc32(raw))
        pixels[len(bits):] = (np.random.randint(0, 2, n_fill) * 255).astype(np.uint8)

    save_png(output, pixels.reshape(N, N, 3))
    print(f"Text → Image: {output} ({N}×{N}, {len(raw)}B → {len(comp)}B)")

def image_to_text(path):
    """Read text from image. Checks for TX magic in pixel data."""
    img = _ensure_rgb(load_png(path))
    flat = img.reshape(-1)

    # Check magic: first 2 bytes = 16 bits
    magic = np.packbits((flat[:16] > 128).astype(np.uint8)).tobytes()[:2]

    if magic == TEXT_MAGIC:
        # Header: 2 (magic) + 4 (comp_len) + 4 (raw_len) = 10 bytes = 80 bits
        hdr = np.packbits((flat[:80] > 128).astype(np.uint8)).tobytes()[:10]
        comp_len = struct.unpack('<I', hdr[2:6])[0]
        total_bits = (10 + comp_len) * 8
        data = np.packbits((flat[:total_bits] > 128).astype(np.uint8)).tobytes()
        try:
            text = zlib.decompress(data[10:10 + comp_len]).decode('utf-8')
        except Exception:
            text = _raw_image_to_text(img)
    else:
        text = _raw_image_to_text(img)

    print(f"Image → Text: {len(text)} chars")
    return text

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
