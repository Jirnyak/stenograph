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
text → UTF-8 → zlib compress → [TX magic + lengths + compressed data]
     → binary encoding: each bit = one subpixel (0 or 255)
     → reshape to N×N×3 → save PNG
```

- Magic bytes `TX` (0x54, 0x58) at start of pixel data
- Capacity: `N² × 3 / 8` bytes (compressed) at N×N
- Default N=1024 → ~393 KB compressed text capacity
- Binary encoding survives ±100 pixel error via threshold at 128
- Remaining pixels filled with seeded random 0/255 (noise-like)

### Image → Text

```
image → flatten pixels → check first 16 subpixels for TX magic
      → threshold at 128 → unpack bits → read lengths
      → zlib decompress → UTF-8 decode → text
```

Falls back to raw ASCII extraction if no magic found.

### Audio → Image

```
audio → load/resample to 22050 Hz mono → reshape to (N-1) × N frames
      → FFT per frame → log-magnitude (R), phase (G), RMS envelope (B)
      → log-frequency mapping (even distribution across pixels)
      → fixed scaling ranges (no per-image params needed)
      → row 0: binary-encoded [AU magic + sample_rate + sample_count]
      → save N×N PNG
```

Fixed scaling ranges (no stored parameters):
- Magnitude: `[0, 20]` (log1p scale)
- Phase: `[-π, π]`
- RMS: `[0, 1]`

### Image → Audio

```
image → check row 0 for AU magic → extract sr, n_samples
      → rows 1+: decode R→magnitude, G→phase using fixed ranges
      → log-frequency map interpolation to full spectrum
      → inverse FFT per row → concatenate → normalize → WAV
```

Works on ANY image (uses defaults if no audio magic found).

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
        ├── main.js      # UI logic
        └── style.css    # Minimal dark theme
```

## Data Format Summary

| What | Format | Magic | Storage |
|------|--------|-------|---------|
| Key | `.npy` float64 `(3,B,B)` | numpy header | Separate file |
| Encrypted image | PNG, RGB | None | Pure pixel data |
| Text-as-image | PNG, binary 0/255 | `TX` in pixels | Self-contained |
| Audio-as-image | PNG, spectral | `AU` in row 0 | Self-contained |

**Zero metadata in any output PNG.** All information lives in pixel values.
