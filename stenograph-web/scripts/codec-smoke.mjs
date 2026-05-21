import { audioToFourierRGB, fourierRGBToAudio, isFourierAudioRGB } from '../src/audio-fourier.js';
import { audioToRGB, rgbToAudio } from '../src/steno.js';

const SR = 22050;
const EPS = 1e-9;

function assert(ok, msg) {
  if (!ok) throw new Error(msg);
}

function peak(samples) {
  let p = 0;
  for (const v of samples) p = Math.max(p, Math.abs(v));
  return p;
}

function corr(a, b) {
  const n = Math.min(a.length, b.length);
  let aa = 0, bb = 0, ab = 0;
  for (let i = 0; i < n; i++) {
    aa += a[i] * a[i];
    bb += b[i] * b[i];
    ab += a[i] * b[i];
  }
  return ab / Math.sqrt(Math.max(aa * bb, EPS));
}

function tone(freq, seconds = 2) {
  const n = SR * seconds;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = 0.7 * Math.sin(2 * Math.PI * freq * i / SR);
  return out;
}

function chord(seconds = 2) {
  const n = SR * seconds;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / SR;
    out[i] = 0.35 * Math.sin(2 * Math.PI * 220 * t)
      + 0.25 * Math.sin(2 * Math.PI * 440 * t)
      + 0.20 * Math.sin(2 * Math.PI * 880 * t);
  }
  return out;
}

function checkFourier(name, samples, minCorr) {
  const rgb = audioToFourierRGB(samples, SR);
  const decoded = fourierRGBToAudio(rgb);
  const c = corr(samples, decoded.samples);
  assert(decoded.detected, `${name}: AF header was not detected`);
  assert(isFourierAudioRGB(rgb), `${name}: AF image was not auto-detectable`);
  assert(decoded.sampleRate === SR, `${name}: sample rate changed`);
  assert(decoded.samples.length === samples.length, `${name}: sample length changed`);
  assert(peak(decoded.samples) > 0.5, `${name}: decoded signal is too quiet`);
  assert(c >= minCorr, `${name}: Fourier correlation ${c.toFixed(4)} < ${minCorr}`);
  console.log(`${name}: Fourier corr=${c.toFixed(4)} peak=${peak(decoded.samples).toFixed(3)}`);
  return rgb;
}

function checkPCM(name, samples) {
  const decoded = rgbToAudio(audioToRGB(samples, SR));
  const c = corr(samples, decoded.samples);
  assert(decoded.sampleRate === SR, `${name}: PCM sample rate changed`);
  assert(decoded.samples.length === samples.length, `${name}: PCM sample length changed`);
  assert(c >= 0.99, `${name}: PCM correlation ${c.toFixed(4)} < 0.99`);
  console.log(`${name}: PCM corr=${c.toFixed(4)} peak=${peak(decoded.samples).toFixed(3)}`);
}

const sine = tone(440);
const triad = chord();

const sineRGB = checkFourier('sine 440Hz', sine, 0.95);
checkFourier('three-tone chord', triad, 0.95);
checkPCM('sine 440Hz', sine);
checkPCM('three-tone chord', triad);

let quietMax = 0;
for (let y = 4; y < 1024; y++) {
  const p = (y * 1024 + 1023) * 3;
  quietMax = Math.max(quietMax, sineRGB[p], sineRGB[p + 1], sineRGB[p + 2]);
}
assert(quietMax <= 4, `silent Fourier tail is not black, max=${quietMax}`);
console.log(`silent tail: max=${quietMax}`);
