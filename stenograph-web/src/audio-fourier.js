const N = 1024;
const T = N * N * 3;
const PIX = N * N;

const SAMPLE_RATE = 22050;
const FRAME = 1024;
const HOP = 512;
const FRAMES = 1024;
const SYNTH_LEN = (FRAMES - 1) * HOP + FRAME;

const HEADER_ROWS = 4;
const HEADER_BYTES = 16;
const HEADER_BITS = HEADER_BYTES * 8;
const HEADER_REPS = 25;
const HEADER_SEG = ((N * HEADER_ROWS) / HEADER_REPS) | 0;
const HEADER_PRIME = 109;
const BIT_ZERO = 88;
const BIT_ONE = 168;

const MIN_FREQ = 40;
const MAX_FREQ = 10000;
const MAG_MAX = 6.0;
const MAP_LO = 4;
const MAP_SCALE = 247;
const PHASE_CENTER = 128;
const PHASE_RADIUS = 123;

const LUM_R = 0.299;
const LUM_G = 0.587;
const LUM_B = 0.114;

const window1024 = makeHann(FRAME);
const imageBins = makeImageBinMap(N - HEADER_ROWS);
const synthYs = makeSynthYMap(N - HEADER_ROWS);
const synthPhases = makeInitialPhases();

function clamp(n, lo, hi) {
  return n < lo ? lo : n > hi ? hi : n;
}

function makeHann(n) {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1));
  return w;
}

function fitSamples(samples) {
  const out = new Float64Array(SYNTH_LEN);
  if (!samples.length) return out;

  let peak = 0;
  for (let i = 0; i < samples.length; i++) {
    const v = Math.abs(samples[i]);
    if (v > peak) peak = v;
  }
  const gain = peak > 1e-10 ? 0.95 / peak : 1;

  if (samples.length <= SYNTH_LEN) {
    for (let i = 0; i < samples.length; i++) out[i] = samples[i] * gain;
    return out;
  }

  for (let i = 0; i < SYNTH_LEN; i++) {
    const p = i * (samples.length - 1) / (SYNTH_LEN - 1);
    const lo = Math.floor(p);
    const hi = Math.min(lo + 1, samples.length - 1);
    out[i] = (samples[lo] + (samples[hi] - samples[lo]) * (p - lo)) * gain;
  }
  return out;
}

function makeImageBinMap(rows) {
  const map = new Array(rows);
  const logRange = Math.log(MAX_FREQ / MIN_FREQ);
  for (let y = 0; y < rows; y++) {
    const t = 1 - y / (rows - 1);
    const freq = MIN_FREQ * Math.exp(t * logRange);
    const bin = clamp(freq * FRAME / SAMPLE_RATE, 1, FRAME / 2);
    const lo = Math.floor(bin);
    map[y] = { lo, hi: Math.min(lo + 1, FRAME / 2), f: bin - lo };
  }
  return map;
}

function makeSynthYMap(rows) {
  const map = new Float64Array(FRAME / 2 + 1);
  const logRange = Math.log(MAX_FREQ / MIN_FREQ);
  for (let k = 1; k <= FRAME / 2; k++) {
    const freq = k * SAMPLE_RATE / FRAME;
    const t = Math.log(clamp(freq, MIN_FREQ, MAX_FREQ) / MIN_FREQ) / logRange;
    map[k] = (1 - t) * (rows - 1);
  }
  return map;
}

function makeInitialPhases() {
  const phases = new Float64Array(FRAME / 2 + 1);
  for (let k = 1; k < phases.length; k++) {
    let x = (k * 2654435761) >>> 0;
    x ^= x << 13; x ^= x >>> 17; x ^= x << 5;
    phases[k] = ((x >>> 0) / 4294967296) * Math.PI * 2;
  }
  return phases;
}

function fft(re, im, inverse = false) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i]; re[i] = re[j]; re[j] = tr;
      const ti = im[i]; im[i] = im[j]; im[j] = ti;
    }
  }

  for (let len = 2; len <= n; len <<= 1) {
    const ang = (inverse ? 2 : -2) * Math.PI / len;
    const wlr = Math.cos(ang);
    const wli = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let wr = 1;
      let wi = 0;
      const half = len >> 1;
      for (let j = 0; j < half; j++) {
        const a = i + j;
        const b = a + half;
        const br = re[b] * wr - im[b] * wi;
        const bi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - br;
        im[b] = im[a] - bi;
        re[a] += br;
        im[a] += bi;
        const nwr = wr * wlr - wi * wli;
        wi = wr * wli + wi * wlr;
        wr = nwr;
      }
    }
  }

  if (inverse) {
    for (let i = 0; i < n; i++) {
      re[i] /= n;
      im[i] /= n;
    }
  }
}

function mapToByte(v) {
  return clamp(Math.round(MAP_LO + clamp(v / MAG_MAX, 0, 1) * MAP_SCALE), 0, 255);
}

function mapFromByte(v) {
  return clamp((v - MAP_LO) / MAP_SCALE * MAG_MAX, 0, MAG_MAX);
}

function makeHeader(sampleRate, sampleCount) {
  const h = new Uint8Array(HEADER_BYTES);
  h[0] = 0x41; // A
  h[1] = 0x46; // F
  h[2] = 1;
  h[3] = 2; // log-magnitude + phase-vector format
  const dv = new DataView(h.buffer);
  dv.setUint32(4, sampleRate >>> 0, true);
  dv.setUint32(8, sampleCount >>> 0, true);
  dv.setUint16(12, FRAME, true);
  dv.setUint16(14, HOP, true);
  return h;
}

function writeHeader(rgb, sampleRate, sampleCount) {
  const h = makeHeader(sampleRate, sampleCount);
  for (let bit = 0; bit < HEADER_BITS; bit++) {
    const v = ((h[bit >> 3] >> (7 - (bit & 7))) & 1) ? BIT_ONE : BIT_ZERO;
    for (let r = 0; r < HEADER_REPS; r++) {
      const p = r * HEADER_SEG + ((bit * HEADER_PRIME) % HEADER_SEG);
      const i = p * 3;
      rgb[i] = v;
      rgb[i + 1] = v;
      rgb[i + 2] = v;
    }
  }
}

function readHeader(rgb) {
  const h = new Uint8Array(HEADER_BYTES);
  for (let bit = 0; bit < HEADER_BITS; bit++) {
    let vote = 0;
    for (let r = 0; r < HEADER_REPS; r++) {
      const p = r * HEADER_SEG + ((bit * HEADER_PRIME) % HEADER_SEG);
      const i = p * 3;
      vote += LUM_R * (rgb[i] - 128) + LUM_G * (rgb[i + 1] - 128) + LUM_B * (rgb[i + 2] - 128);
    }
    if (vote > 0) h[bit >> 3] |= 1 << (7 - (bit & 7));
  }

  if (h[0] !== 0x41 || h[1] !== 0x46 || h[2] !== 1) return null;
  const dv = new DataView(h.buffer);
  const sampleRate = dv.getUint32(4, true);
  const sampleCount = dv.getUint32(8, true);
  const frame = dv.getUint16(12, true);
  const hop = dv.getUint16(14, true);
  if (sampleRate < 8000 || sampleRate > 96000 || frame !== FRAME || hop !== HOP) return null;
  return { format: h[3], sampleRate, sampleCount, frame, hop };
}

function resample(samples, count) {
  const out = new Float64Array(count);
  if (!samples.length || !count) return out;
  if (count === samples.length) {
    out.set(samples);
    return out;
  }
  if (count === 1) {
    out[0] = samples[0];
    return out;
  }
  for (let i = 0; i < count; i++) {
    const p = i * (samples.length - 1) / (count - 1);
    const lo = Math.floor(p);
    const hi = Math.min(lo + 1, samples.length - 1);
    out[i] = samples[lo] + (samples[hi] - samples[lo]) * (p - lo);
  }
  return out;
}

function readLogAt(rgb, frame, y, y0, rows) {
  const yi = Math.floor(y);
  let sum = 0;
  let n = 0;
  for (let dy = -1; dy <= 1; dy++) {
    const sy = clamp(yi + dy, 0, rows - 1) + y0;
    for (let dx = -1; dx <= 1; dx++) {
      const sx = clamp(frame + dx, 0, N - 1);
      const p = (sy * N + sx) * 3;
      sum += mapFromByte((rgb[p] + rgb[p + 1]) * 0.5);
      n++;
    }
  }
  return sum / n;
}

function phaseToByte(v) {
  return clamp(Math.round(PHASE_CENTER + clamp(v, -1, 1) * PHASE_RADIUS), 0, 255);
}

function phaseFromByte(v) {
  return clamp((v - PHASE_CENTER) / PHASE_RADIUS, -1, 1);
}

function readPhaseComplexAt(rgb, frame, y, y0, rows) {
  const y1 = clamp(Math.floor(y), 0, rows - 1);
  const y2 = Math.min(y1 + 1, rows - 1);
  const f = y - y1;
  const p1 = ((y1 + y0) * N + frame) * 3;
  const p2 = ((y2 + y0) * N + frame) * 3;
  const m1 = Math.expm1(mapFromByte(rgb[p1]));
  const m2 = Math.expm1(mapFromByte(rgb[p2]));
  const r1 = m1 * phaseFromByte(rgb[p1 + 1]);
  const i1 = m1 * phaseFromByte(rgb[p1 + 2]);
  const r2 = m2 * phaseFromByte(rgb[p2 + 1]);
  const i2 = m2 * phaseFromByte(rgb[p2 + 2]);
  return {
    re: r1 + (r2 - r1) * f,
    im: i1 + (i2 - i1) * f,
  };
}

/**
 * Audio -> editable Fourier image.
 * R carries log-frequency magnitude. G/B carry phase as cos/sin.
 * The header is redundant in pixels; no PNG metadata is used.
 */
export function audioToFourierRGB(samples, sampleRate = SAMPLE_RATE) {
  const data = fitSamples(samples);
  const rgb = new Uint8Array(T);
  rgb.fill(128);

  const re = new Float64Array(FRAME);
  const im = new Float64Array(FRAME);
  const mags = new Float64Array(FRAME / 2 + 1);
  const rows = N - HEADER_ROWS;

  for (let x = 0; x < FRAMES; x++) {
    const start = x * HOP;
    for (let i = 0; i < FRAME; i++) {
      const v = data[start + i] * window1024[i];
      re[i] = v;
      im[i] = 0;
    }
    fft(re, im);

    for (let k = 0; k <= FRAME / 2; k++) mags[k] = Math.hypot(re[k], im[k]);

    for (let y = 0; y < rows; y++) {
      const m = imageBins[y];
      const rr = re[m.lo] + (re[m.hi] - re[m.lo]) * m.f;
      const ii = im[m.lo] + (im[m.hi] - im[m.lo]) * m.f;
      const mag = mags[m.lo] + (mags[m.hi] - mags[m.lo]) * m.f;
      const len = Math.hypot(rr, ii);
      const p = ((y + HEADER_ROWS) * N + x) * 3;
      rgb[p] = mapToByte(Math.log1p(mag));
      rgb[p + 1] = phaseToByte(len > 1e-12 ? rr / len : 1);
      rgb[p + 2] = phaseToByte(len > 1e-12 ? ii / len : 0);
    }
  }

  writeHeader(rgb, sampleRate, samples.length);
  return rgb;
}

/**
 * Fourier image -> audio.
 * Works on AF images and on arbitrary drawings by treating brightness as a
 * log-frequency magnitude map.
 */
export function fourierRGBToAudio(rgb) {
  if (!rgb || rgb.length !== T) throw new Error('Image must be 1024x1024 RGB');

  const header = readHeader(rgb);
  const sampleRate = header ? header.sampleRate : SAMPLE_RATE;
  const y0 = header ? HEADER_ROWS : 0;
  const rows = N - y0;
  const out = new Float64Array(SYNTH_LEN);
  const norm = new Float64Array(SYNTH_LEN);
  const re = new Float64Array(FRAME);
  const im = new Float64Array(FRAME);
  const phases = new Float64Array(synthPhases);
  const phaseVector = header && header.format >= 2;

  for (let x = 0; x < FRAMES; x++) {
    re.fill(0);
    im.fill(0);

    for (let k = 1; k <= FRAME / 2; k++) {
      const freq = k * sampleRate / FRAME;
      if (freq > MAX_FREQ) continue;
      const y = synthYs[k] * rows / (N - HEADER_ROWS);
      if (phaseVector) {
        const c = readPhaseComplexAt(rgb, x, y, y0, rows);
        re[k] = c.re;
        im[k] = c.im;
      } else {
        const logMag = readLogAt(rgb, x, y, y0, rows);
        const mag = Math.expm1(logMag);
        const ph = phases[k];
        re[k] = mag * Math.cos(ph);
        im[k] = mag * Math.sin(ph);
        phases[k] += 2 * Math.PI * k * HOP / FRAME;
        if (phases[k] > Math.PI * 2) phases[k] %= Math.PI * 2;
      }
      if (k < FRAME / 2) {
        re[FRAME - k] = re[k];
        im[FRAME - k] = -im[k];
      } else {
        im[k] = 0;
      }
    }

    fft(re, im, true);

    const start = x * HOP;
    for (let i = 0; i < FRAME; i++) {
      const w = window1024[i];
      out[start + i] += re[i] * w;
      norm[start + i] += w * w;
    }
  }

  for (let i = 0; i < out.length; i++) {
    if (norm[i] > 1e-8) out[i] /= norm[i];
  }

  const wanted = header && header.sampleCount > 0 && header.sampleCount <= SAMPLE_RATE * 600
    ? header.sampleCount
    : SYNTH_LEN;
  let samples = out;
  if (wanted < out.length) samples = out.slice(0, wanted);
  else if (wanted > out.length) samples = resample(out, wanted);

  let peak = 0;
  for (let i = 0; i < samples.length; i++) {
    const v = Math.abs(samples[i]);
    if (v > peak) peak = v;
  }
  if (peak > 1e-10) {
    const gain = 0.95 / peak;
    for (let i = 0; i < samples.length; i++) samples[i] *= gain;
  }

  return { samples, sampleRate, detected: !!header };
}
