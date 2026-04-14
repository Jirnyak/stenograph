/**
 * STENOGRAPH — Core
 *
 * Everything is a 1024×1024 image. Every algorithm is dumb and universal.
 * No magic bytes. No detection. No memory of what anything was.
 * You give it an image, it runs the algorithm. That's it.
 */

const N = 1024;
const T = N * N * 3;          // 3,145,728 total bytes
const PIX = N * N;            // 1,048,576 pixels (= bytes per channel)

// Primes for spatial interleaving (all odd → coprime to PIX=2^20)
const S1 = 104729;
const S2 = 7919;
const S3 = 65537;

// Safe binary pixel values (avoid clipping during matrix multiply)
const BIT_ZERO = 88;
const BIT_ONE  = 168;

// 25× spatial repetition for JPEG resilience
const TEXT_REPS = 25;
const TEXT_PRIMES = [
  104729, 7919, 65537, 15485863, 324517, 49979687, 611953,
  2097169, 3145739, 4194319, 5242907, 6291469, 7340033,
  8388617, 9437189, 10485767, 11534351, 12582917, 13631489,
  14680067, 15728641, 16777259, 17825801, 18874379, 19922947
];
const SEG_SIZE = (PIX / TEXT_REPS) | 0;

// JPEG YCbCr luminance weights
const LUM_R = 0.299, LUM_G = 0.587, LUM_B = 0.114;

// ─── CRC-32 ─────────────────────────────────────────────────────────────────

const CRC_T = new Uint32Array(256);
for (let i = 0; i < 256; i++) {
  let c = i;
  for (let j = 0; j < 8; j++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
  CRC_T[i] = c;
}
function crc32(data) {
  let c = 0xFFFFFFFF;
  for (let i = 0; i < data.length; i++) c = CRC_T[(c ^ data[i]) & 0xFF] ^ (c >>> 8);
  return (c ^ 0xFFFFFFFF) >>> 0;
}

// ─── Canvas / RGB helpers ───────────────────────────────────────────────────

function makeCanvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  return { canvas: c, ctx: c.getContext('2d') };
}

/** Any RGBA + dimensions → 1024×1024 RGB (Uint8Array) */
function toN(rgba, w, h) {
  const { canvas: src, ctx: sctx } = makeCanvas(w, h);
  sctx.putImageData(new ImageData(new Uint8ClampedArray(rgba), w, h), 0, 0);
  const { canvas: dst, ctx: dctx } = makeCanvas(N, N);
  dctx.drawImage(src, 0, 0, N, N);
  const d = dctx.getImageData(0, 0, N, N).data;
  const rgb = new Uint8Array(N * N * 3);
  for (let i = 0; i < N * N; i++) {
    rgb[i * 3] = d[i * 4];
    rgb[i * 3 + 1] = d[i * 4 + 1];
    rgb[i * 3 + 2] = d[i * 4 + 2];
  }
  return rgb;
}

/** RGB → RGBA for ImageData */
function toRGBA(rgb) {
  const rgba = new Uint8ClampedArray(N * N * 4);
  for (let i = 0; i < N * N; i++) {
    rgba[i * 4] = rgb[i * 3];
    rgba[i * 4 + 1] = rgb[i * 3 + 1];
    rgba[i * 4 + 2] = rgb[i * 3 + 2];
    rgba[i * 4 + 3] = 255;
  }
  return rgba;
}

/** RGB → PNG Blob */
export function rgbToBlob(rgb) {
  const { canvas, ctx } = makeCanvas(N, N);
  ctx.putImageData(new ImageData(toRGBA(rgb), N, N), 0, 0);
  return new Promise(r => canvas.toBlob(r, 'image/png'));
}

/** File (image) → 1024×1024 RGB */
export function loadImageFile(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const { canvas, ctx } = makeCanvas(img.width, img.height);
      ctx.drawImage(img, 0, 0);
      const d = ctx.getImageData(0, 0, img.width, img.height).data;
      resolve(toN(d, img.width, img.height));
      URL.revokeObjectURL(img.src);
    };
    img.onerror = () => reject(new Error('Cannot load image'));
    img.src = URL.createObjectURL(file);
  });
}

/** File (audio) → mono Float32Array at 22050 Hz. Max 600 seconds. */
export function loadAudioFile(file) {
  return file.arrayBuffer().then(buf => {
    const actx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 22050 });
    return actx.decodeAudioData(buf).then(ab => {
      const samples = ab.getChannelData(0);
      actx.close();
      const maxSamples = 22050 * 600;
      if (samples.length > maxSamples) return samples.subarray(0, maxSamples);
      return samples;
    });
  });
}

// ─── GPU Matrix Multiply (WebGL2) ───────────────────────────────────────────

let _gpuReady = false, _gpuCtx = null;

function initGPU() {
  if (_gpuReady) return _gpuCtx;
  _gpuReady = true;
  try {
    const c = document.createElement('canvas');
    c.width = c.height = 1;
    const gl = c.getContext('webgl2');
    if (!gl) return null;

    const vs = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vs, '#version 300 es\nin vec2 p;\nvoid main(){gl_Position=vec4(p,0,1);}');
    gl.compileShader(vs);

    const fs = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fs, `#version 300 es
precision highp int;
precision highp isampler2D;
uniform isampler2D uA, uB;
out ivec4 C;
void main(){
  ivec2 pos=ivec2(gl_FragCoord.xy);
  int s=0;
  for(int k=0;k<1024;k++)
    s+=texelFetch(uA,ivec2(k,pos.y),0).r*texelFetch(uB,ivec2(k,pos.x),0).r;
  C=ivec4(s,0,0,0);
}`);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) return null;

    const pr = gl.createProgram();
    gl.attachShader(pr, vs); gl.attachShader(pr, fs); gl.linkProgram(pr);
    if (!gl.getProgramParameter(pr, gl.LINK_STATUS)) return null;

    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const vb = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vb);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,1,-1,-1,1,1,1]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(pr, 'p');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

    _gpuCtx = { gl, pr, vao, uA: gl.getUniformLocation(pr, 'uA'), uB: gl.getUniformLocation(pr, 'uB') };
    return _gpuCtx;
  } catch (e) { return null; }
}

function gpuMul(A, B, n) {
  const g = initGPU();
  if (!g) return null;
  const { gl, pr, vao, uA, uB } = g;

  try {
    const BT = new Int32Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) BT[j * n + i] = B[i * n + j];

    gl.useProgram(pr);
    gl.bindVertexArray(vao);

    function uploadTex(unit, data) {
      const t = gl.createTexture();
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32I, n, n, 0, gl.RED_INTEGER, gl.INT, data);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      return t;
    }
    const tA = uploadTex(0, A);
    const tB = uploadTex(1, BT);
    gl.uniform1i(uA, 0);
    gl.uniform1i(uB, 1);

    const tC = gl.createTexture();
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, tC);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32I, n, n, 0, gl.RGBA_INTEGER, gl.INT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tC, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
      gl.deleteTexture(tA); gl.deleteTexture(tB); gl.deleteTexture(tC);
      gl.deleteFramebuffer(fb); gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      _gpuCtx = null; return null;
    }

    gl.viewport(0, 0, n, n);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    const buf = new Int32Array(n * n * 4);
    gl.readPixels(0, 0, n, n, gl.RGBA_INTEGER, gl.INT, buf);

    const C = new Int32Array(n * n);
    for (let i = 0; i < n * n; i++) C[i] = buf[i * 4];

    gl.deleteTexture(tA); gl.deleteTexture(tB); gl.deleteTexture(tC);
    gl.deleteFramebuffer(fb); gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return C;
  } catch (e) {
    _gpuCtx = null;
    return null;
  }
}

// ─── CPU Matrix Multiply (fallback) ─────────────────────────────────────────

function cpuMul(A, B, n) {
  const BT = new Int32Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) BT[j * n + i] = B[i * n + j];
  const C = new Int32Array(n * n);
  for (let i = 0; i < n; i++) {
    const ai = i * n;
    for (let j = 0; j < n; j++) {
      const bj = j * n;
      let s = 0;
      for (let k = 0; k < n; k++) s += A[ai + k] * BT[bj + k];
      C[ai + j] = s;
    }
  }
  return C;
}

/** C = A × B (raw integer). Tries GPU first, falls back to CPU. */
function matMul(A, B, n) {
  const gpu = gpuMul(A, B, n);
  if (gpu) return gpu;
  return cpuMul(A, B, n);
}

// ─── Hadamard key generation (orthogonal, binary ±1) ────────────────────────

/** Build Sylvester Hadamard matrix H_n, entries ±1, H×Hᵀ = n×I */
function buildHadamard(n) {
  const H = new Int8Array(n * n);
  H[0] = 1;
  for (let half = 1; half < n; half *= 2)
    for (let i = 0; i < half; i++)
      for (let j = 0; j < half; j++) {
        const v = H[i * n + j];
        H[i * n + half + j] = v;
        H[(half + i) * n + j] = v;
        H[(half + i) * n + half + j] = -v;
      }
  return H;
}

function fisherYates(n) {
  const a = new Uint32Array(n);
  for (let i = 0; i < n; i++) a[i] = i;
  const r = new Uint32Array(n);
  crypto.getRandomValues(r);
  for (let i = n - 1; i > 0; i--) {
    const j = r[i] % (i + 1);
    const t = a[i]; a[i] = a[j]; a[j] = t;
  }
  return a;
}

/**
 * Generate key pair: K and Kᵀ (randomized Hadamard).
 * K/√n is orthogonal → K × Kᵀ = n × I.
 * Keys stored as binary 0/255 → survive any JPEG compression.
 * Independent matrix per RGB channel for maximum diffusion.
 */
export function generateKeyPair() {
  const H = buildHadamard(N);
  const enc = new Uint8Array(N * N * 3);
  const dec = new Uint8Array(N * N * 3);

  for (let ch = 0; ch < 3; ch++) {
    const rp = fisherYates(N), cp = fisherYates(N);
    const sb = new Uint8Array(N * 2);
    crypto.getRandomValues(sb);

    for (let i = 0; i < N; i++) {
      const rs = (sb[i] & 1) * 2 - 1;
      for (let j = 0; j < N; j++) {
        const cs = (sb[N + j] & 1) * 2 - 1;
        const v = rs * cs * H[rp[i] * N + cp[j]];
        const px = v > 0 ? 255 : 0;
        enc[(i * N + j) * 3 + ch] = px;
        dec[(j * N + i) * 3 + ch] = px;
      }
    }
  }

  return { encrypt: enc, decrypt: dec };
}

// ─── Matrix multiply (float, orthogonal, compression-resilient) ─────────────

/**
 * image × key. Both are 1024×1024 RGB images.
 * Key interpreted as binary ±1 (threshold 128). Image centred at 128.
 * Y = K × (X − 128) / √n + 128.
 * Orthogonal keys don't amplify errors → survives JPEG, resize, etc.
 * ~8% clipping at extremes for high-variance images — acceptable trade.
 */
export function multiply(imageRGB, keyRGB) {
  const out = new Uint8Array(N * N * 3);
  const scale = 1.0 / Math.sqrt(N);

  for (let ch = 0; ch < 3; ch++) {
    const K = new Int32Array(N * N);
    for (let i = 0; i < N * N; i++)
      K[i] = keyRGB[i * 3 + ch] > 128 ? 1 : -1;

    const X = new Int32Array(N * N);
    for (let i = 0; i < N * N; i++)
      X[i] = imageRGB[i * 3 + ch] - 128;

    const Y = matMul(K, X, N);

    for (let i = 0; i < N * N; i++)
      out[i * 3 + ch] = Math.max(0, Math.min(255, Math.round(Y[i] * scale + 128)));
  }

  return out;
}

// ─── Text → Image (dumb, no magic) ─────────────────────────────────────────

async function zDeflate(data) {
  const cs = new CompressionStream('deflate');
  const w = cs.writable.getWriter();
  w.write(data); w.close();
  const chunks = []; const r = cs.readable.getReader();
  while (true) { const { done, value } = await r.read(); if (done) break; chunks.push(value); }
  let len = 0; for (const c of chunks) len += c.length;
  const out = new Uint8Array(len); let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}

async function zInflate(data) {
  const ds = new DecompressionStream('deflate');
  const w = ds.writable.getWriter();
  w.write(data); w.close();
  const chunks = []; const r = ds.readable.getReader();
  while (true) { const { done, value } = await r.read(); if (done) break; chunks.push(value); }
  let len = 0; for (const c of chunks) len += c.length;
  const out = new Uint8Array(len); let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}

/**
 * Text → 1024×1024 RGB.
 * 25× spatial repetition + R=G=B per pixel = 75 copies per bit.
 * Luminance-weighted soft voting on decode survives JPEG q≥35.
 * [compLen:4][rawLen:4][CRC32:4][zlibData] → bits → interleaved pixels.
 * Capacity: ~5KB compressed text.
 */
export async function textToRGB(text) {
  const raw = new TextEncoder().encode(text);
  const comp = await zDeflate(raw);
  const crc = crc32(comp);

  const payload = new Uint8Array(12 + comp.length);
  const dv = new DataView(payload.buffer);
  dv.setUint32(0, comp.length, true);
  dv.setUint32(4, raw.length, true);
  dv.setUint32(8, crc, true);
  payload.set(comp, 12);

  const numBits = payload.length * 8;
  if (numBits > SEG_SIZE) throw new Error('Text too long');

  const rgb = new Uint8Array(T);

  // Fill with random binary noise (R=G=B safe pixel values)
  const CHUNK = 65536;
  for (let pos = 0; pos < T; pos += CHUNK) {
    const end = Math.min(pos + CHUNK, T);
    const noise = new Uint8Array(end - pos);
    crypto.getRandomValues(noise);
    for (let k = 0; k < noise.length; k++) {
      const v = noise[k] < 128 ? BIT_ZERO : BIT_ONE;
      rgb[pos + k] = v;
    }
  }

  // Overwrite all R=G=B to match per-pixel (noise fill above is per subpixel)
  for (let p = 0; p < PIX; p++) {
    rgb[p * 3 + 1] = rgb[p * 3];
    rgb[p * 3 + 2] = rgb[p * 3];
  }

  // Write each bit at TEXT_REPS spatially interleaved positions, R=G=B
  for (let i = 0; i < numBits; i++) {
    const v = ((payload[i >> 3] >> (7 - (i & 7))) & 1) ? BIT_ONE : BIT_ZERO;
    for (let r = 0; r < TEXT_REPS; r++) {
      const pos = r * SEG_SIZE + ((i * TEXT_PRIMES[r]) % SEG_SIZE);
      rgb[pos * 3]     = v;
      rgb[pos * 3 + 1] = v;
      rgb[pos * 3 + 2] = v;
    }
  }

  return rgb;
}

/**
 * Image → text. Luminance-weighted soft voting across 25 spatial copies × 3 channels.
 * CRC32 verified. Returns string or null.
 */
export async function rgbToText(rgb) {
  try {
    const hdr = softDecode(rgb, 96);
    const dv = new DataView(hdr.buffer);
    const compLen = dv.getUint32(0, true);
    const rawLen = dv.getUint32(4, true);
    const storedCrc = dv.getUint32(8, true);

    if (compLen < 1 || compLen > SEG_SIZE / 8 || rawLen < 1 || rawLen > 1000000) return null;
    if ((12 + compLen) * 8 > SEG_SIZE) return null;

    const full = softDecode(rgb, (12 + compLen) * 8);
    const comp = full.subarray(12);
    if (crc32(comp) !== storedCrc) return null;

    const dec = await zInflate(comp);
    if (dec.length !== rawLen) return null;
    return new TextDecoder().decode(dec);
  } catch { return null; }
}

/** Soft-decision voting: luminance-weighted sum across 25 spatial copies */
function softDecode(rgb, numBits) {
  const out = new Uint8Array((numBits + 7) >> 3);
  for (let i = 0; i < numBits; i++) {
    let vote = 0;
    for (let r = 0; r < TEXT_REPS; r++) {
      const pos = r * SEG_SIZE + ((i * TEXT_PRIMES[r]) % SEG_SIZE);
      const idx = pos * 3;
      vote += LUM_R * (rgb[idx] - 128) + LUM_G * (rgb[idx + 1] - 128) + LUM_B * (rgb[idx + 2] - 128);
    }
    if (vote > 0) out[i >> 3] |= (1 << (7 - (i & 7)));
  }
  return out;
}

/** Raw pixel values as ASCII (always works, fallback) */
export function rgbToRawText(rgb) {
  let s = '';
  const limit = Math.min(10000, rgb.length / 3);
  for (let i = 0; i < limit; i++) {
    const b = rgb[i * 3];
    if (b >= 32 && b <= 126) s += String.fromCharCode(b);
    else if (b === 10) s += '\n';
  }
  return s;
}

// ─── Audio ↔ Image (16-bit PCM + robust binary header) ──────────────────────

/**
 * Audio → 1024×1024 RGB.
 * Header: 64 bits (origLen + storedLen), each bit stored 9× as binary 0/255.
 *   Copies interleaved across 3 bands (top/middle/bottom of image)
 *   → survives JPEG, resize, partial crop.
 * Data: 16-bit PCM, spatially interleaved from byte 576 onwards.
 * Capacity: ~1,572,576 samples = ~71.3s at 22050 Hz. 96dB SNR.
 */
export function audioToRGB(samples, sr = 22050) {
  const origLen = samples.length;
  const HDR = 576;
  const dataZone = T - HDR;
  const maxSamples = dataZone >> 1;
  let data;

  if (origLen <= maxSamples) {
    data = samples;
  } else {
    data = new Float32Array(maxSamples);
    for (let i = 0; i < maxSamples; i++) {
      const si = i * (origLen - 1) / (maxSamples - 1);
      const lo = Math.floor(si), hi = Math.min(lo + 1, origLen - 1);
      data[i] = samples[lo] + (samples[hi] - samples[lo]) * (si - lo);
    }
  }

  let peak = 0;
  for (let i = 0; i < data.length; i++) peak = Math.max(peak, Math.abs(data[i]));
  const scale = peak > 1e-10 ? 32767 / peak : 1;

  const stored = data.length;
  const hdr = new Uint8Array(8);
  const hdv = new DataView(hdr.buffer);
  hdv.setUint32(0, origLen, true);
  hdv.setUint32(4, stored, true);

  const sampleBytes = new Uint8Array(stored * 2);
  const sdv = new DataView(sampleBytes.buffer);
  for (let i = 0; i < stored; i++)
    sdv.setInt16(i * 2, Math.max(-32768, Math.min(32767, Math.round(data[i] * scale))), true);

  const rgb = new Uint8Array(T);
  rgb.fill(128);

  // Data: interleaved from byte 576 onwards
  for (let i = 0; i < sampleBytes.length; i++)
    rgb[HDR + ((i * S1) % dataZone)] = sampleBytes[i];

  // Header LAST: 64 bits × 9 copies, binary 0/255
  // 3 bands (top/mid/bottom) × 3 copies per band = 9 total
  for (let i = 0; i < 64; i++) {
    const v = ((hdr[i >> 3] >> (7 - (i & 7))) & 1) * 255;
    for (let band = 0; band < 3; band++)
      for (let c = 0; c < 3; c++)
        rgb[band * PIX + i + c * 64] = v;
  }

  return rgb;
}

/**
 * Image → audio. Binary header (9× majority vote), 16-bit PCM.
 * Always works on any image.
 */
export function rgbToAudio(rgb) {
  const HDR = 576;
  const dataZone = T - HDR;

  // Header: 9 copies per bit → majority vote (≥5/9)
  const hdr = new Uint8Array(8);
  for (let i = 0; i < 64; i++) {
    let votes = 0;
    for (let band = 0; band < 3; band++)
      for (let c = 0; c < 3; c++)
        if (rgb[band * PIX + i + c * 64] > 128) votes++;
    if (votes >= 5) hdr[i >> 3] |= (1 << (7 - (i & 7)));
  }

  const hdv = new DataView(hdr.buffer);
  const origLen = hdv.getUint32(0, true);
  const storedLen = hdv.getUint32(4, true);

  const maxSamples = dataZone >> 1;
  const count = (storedLen > 0 && storedLen <= maxSamples) ? storedLen : maxSamples;

  const sampleBytes = new Uint8Array(count * 2);
  for (let i = 0; i < sampleBytes.length; i++)
    sampleBytes[i] = rgb[HDR + ((i * S1) % dataZone)];

  const sdv = new DataView(sampleBytes.buffer);
  const decoded = new Float64Array(count);
  for (let i = 0; i < count; i++)
    decoded[i] = sdv.getInt16(i * 2, true) / 32768;

  const maxOrig = 22050 * 600;
  let out;
  if (origLen > 0 && origLen <= maxOrig && origLen !== count) {
    out = new Float64Array(origLen);
    for (let i = 0; i < origLen; i++) {
      const si = i * (count - 1) / (origLen - 1);
      const lo = Math.floor(si), hi = Math.min(lo + 1, count - 1);
      out[i] = decoded[lo] + (decoded[hi] - decoded[lo]) * (si - lo);
    }
  } else {
    out = decoded;
  }

  return { samples: out, sampleRate: 22050 };
}

/** Samples → WAV Blob */
export function toWavBlob(samples, sr) {
  const n = samples.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const v = new DataView(buf);
  const u = new Uint8Array(buf);
  u.set([0x52,0x49,0x46,0x46], 0); v.setUint32(4, 36+n*2, true);
  u.set([0x57,0x41,0x56,0x45], 8);
  u.set([0x66,0x6D,0x74,0x20], 12); v.setUint32(16, 16, true);
  v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr*2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  u.set([0x64,0x61,0x74,0x61], 36); v.setUint32(40, n*2, true);
  for (let i = 0; i < n; i++)
    v.setInt16(44+i*2, Math.max(-32768, Math.min(32767, Math.round(samples[i]*32767))), true);
  return new Blob([buf], { type: 'audio/wav' });
}

export { N };
