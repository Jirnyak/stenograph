# STENOGRAPH — Architecture

## Core Principle

Everything is an image. Images are matrices. Matrices multiply.

```
Any file ──→ Image ──×K──→ Encrypted Image ──×Kᵀ──→ Image ──→ Any file
```

ONE input file → ONE output PNG. Key is SEPARATE. No metadata. No headers in files.
User manages keys. User decides what to encrypt/decrypt. The system just multiplies.

## Mathematical Foundation

### Orthogonal Matrix Cipher

Key `K` is a random orthogonal matrix: `K · Kᵀ = I` (identity).

- **Encrypt:** `Y = (X - μ) · 0.85 · K + μ` per block
- **Decrypt:** `X = (Y - μ) · Kᵀ / 0.85 + μ` per block

Properties:
- `K⁻¹ = Kᵀ` — no separate inverse needed, just transpose
- Generated via QR decomposition of random Gaussian matrix
- 3 independent matrices, one per RGB channel
- Stored as `key.npy` — float64, shape `(3, B, B)`

### Block-Based Processing

Image split into `B×B` blocks (default B=32). Each block transformed independently.

- **Why blocks:** Full N×N multiply amplifies quantization error by √N.
  Block-based limits error to √B per block.
- **Per-block centering:** subtract block mean μ before transform, add back after.
  Keeps pixel values near valid range [0, 255].
- **Safety factor 0.85:** scales centered values to prevent clipping after
  matrix multiply. Trade-off: slight quality loss vs. no overflow.

### Padding

Images padded to block-multiple dimensions with value 128 (neutral gray).
Output has padded dimensions. This is intentional — no original-size info stored.

## Pipelines

### Image → Image (Encrypt/Decrypt)

```
image.png → load RGB → pad to B multiples → block multiply × K → clip [0,255] → save PNG
```

Pure multiplication. Same dimensions in, same dimensions out (after padding).
No headers, no magic bytes, nothing added to the pixel data.

### Text → Image

```
text → UTF-8 → zlib compress → [TX magic + lengths + CRC32 + compressed data]
     → 25× spatial repetition: each bit stored at 25 different pixel positions
     → per pixel: R=G=B = 88 (bit=0) or 168 (bit=1)
     → save N×N PNG
```

- Magic bytes `TX` (0x54, 0x58) at start of payload
- Header: TX(2) + comp_len(4) + raw_len(4) + CRC32(4) = 14 bytes
- CRC32 on compressed data for integrity verification
- Capacity: `N² / 25 / 8` bytes payload at N×N
- Default N=1024 → ~5.2 KB compressed (~15 KB text at 3:1)
- **Pixel values 88/168 (±40 from 128):** survives matrix cipher clipping
  AND has enough contrast for encrypted image to resist JPEG quantization.
- **25× spatial repetition:** each data bit stored at 25 different positions
  across the image using coprime strides. Spreads JPEG 8×8 block errors
  across many independent data bits.
- **R=G=B per pixel:** 3 independent cipher channels carry same bit.
  After encrypt/decrypt, luminance-weighted soft voting across all
  25 positions × 3 channels = 75 observations per bit.
- **Luminance-weighted voting (Y = 0.299R + 0.587G + 0.114B):**
  JPEG preserves Y channel much better than chrominance. G channel
  after decrypt has ~8% BER vs R ~30% and B ~40%. Luminance weighting
  gives optimal SNR through the JPEG round-trip.
- Survives: encrypt → JPEG q≥35 → decrypt → perfect text recovery
- Remaining pixels filled with random 88/168 noise (R=G=B)

### Image → Text

```
image → for each bit: sum luminance(pixel - 128) across 25 spatial copies
      → sign of sum = decoded bit → check TX magic → verify CRC32
      → zlib decompress → UTF-8 decode → text
```

Falls back to raw ASCII extraction if no magic found.

### Audio → Image (PCM Exact Mode)

```
audio → load/resample to 22050 Hz mono
      → normalize → 16-bit PCM bytes
      → spatially interleave bytes after a redundant length header
      → save 1024×1024 PNG
```

- Web default mode.
- Good for PNG-only round trips where the goal is recognizable source audio.
- Header stores original sample count and stored sample count in repeated pixels.
- This is a byte carrier, not a visual music representation.
- Lossy image edits, JPEG, strong noise, or resizing corrupt samples directly.

### Audio → Fourier Image (Robust Mode)

```
audio → load/resample to the selected grid rate → 1024 Hann-windowed frames
      → FFT per frame → log-magnitude on a linear FFT-bin image grid
      → R = magnitude, G/B = phase as cos/sin
      → rows 0..3: redundant AF header in pixels
      → save N×N PNG
```

- Alternative web mode selected as `Fourier robust`.
- 1024 image columns are time frames. The 0..500s span slider computes a
  discrete grid and stores it in the `AF` header. Short spans use 1024-sample
  FFT frames at 22050 Hz. Longer spans use 2048-sample frames and drop the
  stored sample rate when needed: about 60s stays at 22050 Hz, about 180s uses
  8000 Hz, and 500s uses about 4000 Hz.
- Longer audio is fitted into the selected time grid and expanded back on decode.
  Choose a longer span to reduce this time-warping.
- Magnitude uses a fixed `log1p` range. No per-file metadata is needed for scale.
- Phase is stored as a vector instead of a wrapped angle, so PNG round trips keep
  musical timing and JPEG/noise errors degrade more gradually.
- JPEG/noise/resize damage becomes spectral blur instead of byte corruption.
- Drawing on the red magnitude channel creates tones and attacks. If no `AF`
  header is present, the decoder ignores phase channels and treats the image as
  a freehand log-frequency magnitude map.

### Fourier Image → Audio

```
image → read AF header if present, otherwise use defaults
      → read R magnitude and G/B phase vectors
      → map log-frequency image rows back to FFT bins
      → inverse FFT + overlap-add → normalize → WAV
```

Works on any image. Without an `AF` header the whole image is treated as a
drawable frequency map.

The Python CLI `audio2img` / `img2audio` path is the older phase-preserving
spectral codec and uses an `AU` row header. The web Fourier mode uses `AF` and
phase-vector reconstruction for `AF` images, with a magnitude-only fallback for
freehand images.

### Image → Image-In-Image

```
cover.png + secret.png → fit both into 1024×1024 RGB
                        → replace cover low bits with secret high bits
                        → write SIMG footer in final low-bit cells
                        → hidden-in-cover.png
```

- Default depth is 4 low bits per channel.
- More bits give a clearer recovered image and more visible carrier changes.
- The output must stay PNG or another lossless format; JPEG destroys the low bits.
- A 12-byte in-pixel `SIMG` footer stores version, bit depth, and CRC32.
- Old no-footer images still reveal with an explicit bit depth or the 4-bit fallback.

### Image-In-Image → Image

```
hidden-in-cover.png → read SIMG footer → choose bit depth
                    → read low bits → verify CRC32 → revealed.png
```

The recovered image is quantized by the chosen bit depth. At 4 bits it is
recognizable and useful for visual payloads, not a perfect copy.

## Key Management

Key is a **completely separate file**. Not embedded, not transmitted with data.

```
python steno.py keygen     →  key.npy (exact float64) + key.png (visual)
```

- User generates key once
- User shares key through any channel they trust
- User is responsible for key security
- Same key encrypts and decrypts (just transposed)
- Universal: one key works for any content type or image size

## Project Structure

```
stenograph/
├── steno.py              # Python CLI — full implementation
├── architecture.md       # This file
├── key.npy              # Generated key (user's file)
├── key.png              # Visual key representation
│
└── stenograph-web/      # Web version (standalone, deployable)
    ├── package.json
    ├── vite.config.js
    ├── index.html
    └── src/
        ├── steno.js     # Core math (JS port)
        ├── audio-fourier.js # Robust editable Fourier audio codec
        ├── main.js      # UI logic
        └── style.css    # Minimal dark theme
```

## Data Format Summary

| What | Format | Magic | Storage |
|------|--------|-------|---------|
| Key | `.npy` float64 `(3,B,B)` | numpy header | Separate file |
| Encrypted image | PNG, RGB | None | Pure pixel data |
| Text-as-image | PNG, binary 88/168 | `TX` in pixels | 25× redundant, CRC32 |
| Audio-as-image | PNG, 16-bit PCM bytes | length header | Exact PNG-only web mode |
| Fourier audio image | PNG, magnitude + phase vector | `AF` in pixels | Robust/editable web mode |
| CLI spectral audio | PNG, magnitude+phase | `AU` in row 0 | Python phase-preserving mode |
| Image-in-image | PNG, LSB RGB | `SIMG` footer | 1-4 low bits/channel |

**Zero metadata in any output PNG.** All information lives in pixel values.
