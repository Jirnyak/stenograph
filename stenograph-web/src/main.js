import {
  generateKeyPair, loadImageFile, loadAudioFile, rgbToBlob, multiply,
  textToRGB, rgbToText, rgbToRawText, audioToRGB, rgbToAudio, toWavBlob,
  hideImageInImage, revealImageFromImageInfo,
} from './steno.js';
import { audioToFourierRGB, fourierRGBToAudio, getFourierProfileInfo, isFourierAudioRGB } from './audio-fourier.js';
import { extractFromPhoto } from './extract.js';

const $ = s => document.querySelector(s);
const tick = () => new Promise(r => setTimeout(r, 10));

// ─── DOM ────────────────────────────────────────────────────────────────────

const dropFile   = $('#drop-file');
const dropKey    = $('#drop-key');
const dropCover  = $('#drop-cover');
const fileInput  = $('#file-input');
const keyInput   = $('#key-input');
const coverInput = $('#cover-input');
const textInput  = $('#text-input');
const fileLbl    = $('#file-label');
const fileHint   = $('#file-hint');
const keyLbl     = $('#key-label');
const keyHint    = $('#key-hint');
const coverLbl   = $('#cover-label');
const coverHint  = $('#cover-hint');
const hideBits   = $('#hide-bits');
const audioMode  = $('#audio-mode');
const fourierSpan = $('#fourier-span');
const fourierAuto = $('#fourier-auto');
const fourierSpanLabel = $('#fourier-span-label');
const btnToImg   = $('#btn-to-img');
const btnFromImg = $('#btn-from-img');
const btnMul     = $('#btn-multiply');
const btnHide    = $('#btn-hide-image');
const btnReveal  = $('#btn-reveal-image');
const btnExtract = $('#btn-extract');
const btnKeygen  = $('#btn-keygen');
const statusEl   = $('#status');
const output     = $('#output');
const previewCv  = $('#preview');
const outImage   = $('#out-image');
const outExtract = $('#out-extract');
const extractCv  = $('#extract-preview');
const outText    = $('#out-text');
const outAudio   = $('#out-audio');
const dlImg      = $('#dl-img');
const dlTxt      = $('#dl-txt');
const dlWav      = $('#dl-wav');
const textOut    = $('#text-content');
const audioEl    = $('#audio-player');

// ─── State ──────────────────────────────────────────────────────────────────

let fileRGB = null;
let keyRGB  = null;
let coverRGB = null;
let rawSourceImg = null; // original Image element for extract
let sourceAudio = null;

function setStatus(msg, type = '') {
  statusEl.textContent = msg;
  statusEl.className = 'status ' + type;
}

function clearOutput() {
  output.classList.add('hidden');
  outImage.classList.add('hidden');
  outExtract.classList.add('hidden');
  outText.classList.add('hidden');
  outAudio.classList.add('hidden');
}

async function showRGB(rgb, name = 'stenograph.png') {
  previewCv.width = 1024; previewCv.height = 1024;
  const ctx = previewCv.getContext('2d');
  const rgba = new Uint8ClampedArray(1024 * 1024 * 4);
  for (let i = 0; i < 1024 * 1024; i++) {
    rgba[i*4] = rgb[i*3]; rgba[i*4+1] = rgb[i*3+1]; rgba[i*4+2] = rgb[i*3+2]; rgba[i*4+3] = 255;
  }
  ctx.putImageData(new ImageData(rgba, 1024, 1024), 0, 0);
  const blob = await rgbToBlob(rgb);
  dlImg.href = URL.createObjectURL(blob);
  dlImg.download = name;
  outImage.classList.remove('hidden');
  output.classList.remove('hidden');
}

function showText(text) {
  textOut.textContent = text || '(empty)';
  const blob = new Blob([text || ''], { type: 'text/plain' });
  dlTxt.href = URL.createObjectURL(blob);
  outText.classList.remove('hidden');
  output.classList.remove('hidden');
}

function showAudio(samples, sr) {
  const blob = toWavBlob(samples, sr);
  const url = URL.createObjectURL(blob);
  audioEl.src = url;
  dlWav.href = url;
  outAudio.classList.remove('hidden');
  output.classList.remove('hidden');
}

function download(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}

function getHideBits() {
  if (hideBits.value === 'auto') return 4;
  const n = Number(hideBits.value) | 0;
  return Math.max(1, Math.min(4, n || 4));
}

function getRevealBits() {
  if (hideBits.value === 'auto') return 0;
  return getHideBits();
}

async function ensureSourceRGB(message = 'Drop a file or type text') {
  const txt = textInput.value.trim();
  if (txt && !fileRGB) {
    sourceAudio = null;
    setStatus('Encoding text...', '');
    await tick();
    fileRGB = await textToRGB(txt);
    fileLbl.textContent = 'typed text';
    fileHint.textContent = 'text -> 1024x1024';
    dropFile.classList.add('loaded');
  }
  if (!fileRGB) {
    setStatus(message, 'err');
    return false;
  }
  return true;
}

// ─── File classification & loading ──────────────────────────────────────────

function classifyFile(file) {
  const n = file.name.toLowerCase();
  if (file.type.startsWith('audio/') || /\.(wav|mp3|ogg|flac|m4a|aac)$/.test(n)) return 'audio';
  if (file.type.startsWith('image/') || /\.(png|jpg|jpeg|bmp|webp|gif)$/.test(n)) return 'image';
  return 'text';
}

function getAudioModeName() {
  return audioMode.value === 'fourier' ? 'Fourier' : 'PCM';
}

function getFourierSpanTarget() {
  const n = Number(fourierSpan.value);
  return Number.isFinite(n) ? n : 60;
}

function formatSeconds(seconds) {
  return seconds >= 100 ? `${seconds.toFixed(0)}s` : `${seconds.toFixed(1)}s`;
}

function updateFourierSpanLabel() {
  const info = getFourierProfileInfo(getFourierSpanTarget());
  fourierSpanLabel.textContent = `${formatSeconds(info.seconds)} · ${info.sampleRate}Hz/${info.frame}/${info.hop}`;
  return info;
}

function setFourierSpanTarget(seconds) {
  const min = Number(fourierSpan.min);
  const max = Number(fourierSpan.max);
  const target = Math.max(min, Math.min(max, seconds));
  fourierSpan.value = target.toFixed(1);
  return updateFourierSpanLabel();
}

async function encodeSourceAudio() {
  if (!sourceAudio) return;
  const fourier = audioMode.value === 'fourier';
  const spanTarget = getFourierSpanTarget();
  const spanInfo = updateFourierSpanLabel();
  setStatus(fourier ? 'Drawing Fourier audio...' : 'Encoding PCM audio...', '');
  await tick();
  fileRGB = fourier
    ? audioToFourierRGB(sourceAudio.samples, sourceAudio.sampleRate, spanTarget)
    : audioToRGB(sourceAudio.samples, sourceAudio.sampleRate);
  const seconds = sourceAudio.samples.length / sourceAudio.sampleRate;
  fileHint.textContent = fourier && seconds > spanInfo.seconds
    ? `audio/fourier fit ${formatSeconds(spanInfo.seconds)}`
    : `audio/${audioMode.value} -> 1024x1024`;
}

async function onFile(file) {
  const type = classifyFile(file);
  try {
    rawSourceImg = null;
    sourceAudio = null;
    if (type === 'image') {
      setStatus('Loading image...', '');
      fileRGB = await loadImageFile(file);
      // Keep original image for extract
      rawSourceImg = await loadRawImage(file);
    } else if (type === 'audio') {
      setStatus('Decoding audio...', '');
      const samples = await loadAudioFile(file);
      sourceAudio = { samples, sampleRate: 22050 };
      if (fourierAuto.checked) setFourierSpanTarget(samples.length / sourceAudio.sampleRate);
      await encodeSourceAudio();
    } else {
      setStatus('Encoding text...', '');
      const text = await file.text();
      fileRGB = await textToRGB(text);
    }
    fileLbl.textContent = file.name;
    if (type !== 'audio') fileHint.textContent = `${type} -> 1024x1024`;
    dropFile.classList.add('loaded');
    if (type === 'audio' && audioMode.value === 'fourier') {
      const spanInfo = updateFourierSpanLabel();
      if (sourceAudio.samples.length / sourceAudio.sampleRate > spanInfo.seconds)
        setStatus(`Loaded: ${file.name} (${getAudioModeName()}, fit ${formatSeconds(spanInfo.seconds)})`, 'ok');
      else
        setStatus(`Loaded: ${file.name} (${getAudioModeName()}, ${formatSeconds(spanInfo.seconds)} grid)`, 'ok');
    } else {
      setStatus(type === 'audio' ? `Loaded: ${file.name} (${getAudioModeName()})` : `Loaded: ${file.name}`, 'ok');
    }
  } catch (e) {
    setStatus(`Error: ${e.message}`, 'err');
    console.error(e);
  }
}

async function onKey(file) {
  try {
    keyRGB = await loadImageFile(file, 'stretch');
    keyLbl.textContent = file.name;
    keyHint.textContent = '1024×1024';
    dropKey.classList.add('loaded');
    btnMul.disabled = false;
    setStatus(`Key: ${file.name}`, 'ok');
  } catch (e) {
    setStatus(`Error: ${e.message}`, 'err');
  }
}

async function onCover(file) {
  try {
    setStatus('Loading cover image...', '');
    coverRGB = await loadImageFile(file);
    coverLbl.textContent = file.name;
    coverHint.textContent = 'cover -> 1024x1024';
    dropCover.classList.add('loaded');
    setStatus(`Cover: ${file.name}`, 'ok');
  } catch (e) {
    setStatus(`Error: ${e.message}`, 'err');
    console.error(e);
  }
}

// ─── Actions ────────────────────────────────────────────────────────────────

/** → Image: any input → 1024×1024 PNG. That's it. */
async function doToImage() {
  clearOutput();
  if (!(await ensureSourceRGB())) return;

  await showRGB(fileRGB, 'encoded.png');
  setStatus('To image: 1024x1024', 'ok');
}

/** ← Image: take image, blindly run ALL decoders. Show all 3 outputs. */
async function doFromImage() {
  clearOutput();
  if (!fileRGB) { setStatus('Drop an image', 'err'); return; }

  try {
    // 1. Image — always show
    await showRGB(fileRGB, 'image.png');

    // 2. Text — blindly try, always show something
    setStatus('Decoding text...', '');
    const text = await rgbToText(fileRGB);
    showText(text || rgbToRawText(fileRGB));

    // 3. Audio — blindly run selected decoder, but respect AF Fourier images.
    const useFourier = audioMode.value === 'fourier' || isFourierAudioRGB(fileRGB);
    if (useFourier && audioMode.value !== 'fourier') audioMode.value = 'fourier';
    setStatus(`Decoding ${useFourier ? 'Fourier' : 'PCM'} audio...`, '');
    await tick();
    const audio = useFourier
      ? fourierRGBToAudio(fileRGB)
      : rgbToAudio(fileRGB);
    showAudio(audio.samples, audio.sampleRate);

    setStatus(`From image: all outputs (${useFourier ? 'Fourier' : 'PCM'} audio)`, 'ok');
  } catch (e) {
    setStatus(`Error: ${e.message}`, 'err');
    console.error(e);
  }
}

/** × Key: image × key → image. Pure multiply. */
async function doMultiply() {
  clearOutput();
  if (!(await ensureSourceRGB('Drop a file'))) return;
  if (!keyRGB) { setStatus('Drop a key image', 'err'); return; }

  try {
    setStatus('Multiplying...', '');
    await tick();
    const result = multiply(fileRGB, keyRGB);
    await showRGB(result, 'multiplied.png');
    setStatus('× Key: done', 'ok');
  } catch (e) {
    setStatus(`Error: ${e.message}`, 'err');
    console.error(e);
  }
}

/** Hide current image/file image inside cover image. */
async function doHideImage() {
  clearOutput();
  if (!(await ensureSourceRGB('Drop a secret image/file or type text'))) return;
  if (!coverRGB) { setStatus('Drop a cover image', 'err'); return; }

  try {
    const bits = getHideBits();
    setStatus('Hiding image in cover...', '');
    await tick();
    const result = hideImageInImage(coverRGB, fileRGB, bits);
    await showRGB(result, 'hidden-in-cover.png');
    setStatus(`Hidden in cover with ${bits} low bits + SIMG footer. Keep PNG.`, 'ok');
  } catch (e) {
    setStatus(`Error: ${e.message}`, 'err');
    console.error(e);
  }
}

/** Reveal image hidden in the low bits of the loaded image. */
async function doRevealImage() {
  clearOutput();
  if (!fileRGB) { setStatus('Drop a stego PNG first', 'err'); return; }

  try {
    const requestedBits = getRevealBits();
    setStatus('Revealing hidden image...', '');
    await tick();
    const info = revealImageFromImageInfo(fileRGB, requestedBits);
    const result = info.rgb;
    fileRGB = result;
    rawSourceImg = null;
    fileLbl.textContent = 'revealed image';
    fileHint.textContent = 'hidden -> 1024x1024';
    dropFile.classList.add('loaded');
    await showRGB(result, 'revealed.png');
    if (info.detected && info.verified)
      setStatus(`Hidden image revealed: ${info.bits} bits, checksum ok`, 'ok');
    else if (info.detected)
      setStatus(`Hidden image revealed: ${info.bits} bits, checksum mismatch`, 'err');
    else
      setStatus(`Hidden image revealed: ${info.bits} bits, legacy/no footer`, 'ok');
  } catch (e) {
    setStatus(`Error: ${e.message}`, 'err');
    console.error(e);
  }
}

/** Generate key pair → download 2 PNGs */
async function doKeygen() {
  clearOutput();
  setStatus('Generating key pair...', '');
  await tick();

  const { encrypt: enc, decrypt: dec } = generateKeyPair();
  await showRGB(enc, 'key_encrypt.png');

  const encBlob = await rgbToBlob(enc);
  const decBlob = await rgbToBlob(dec);
  download(encBlob, 'key_encrypt.png');
  await tick();
  download(decBlob, 'key_decrypt.png');

  setStatus('Key pair → 2 PNGs downloaded', 'ok');
}

/** Load file as an Image element (full resolution, for extract) */
function loadRawImage(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Cannot load image'));
    img.src = URL.createObjectURL(file);
  });
}

/** Extract: find stenograph square in raw photo → 1024×1024 */
async function doExtract() {
  clearOutput();
  if (!rawSourceImg) { setStatus('Drop a photo first', 'err'); return; }

  try {
    setStatus('Extracting...', '');
    await tick();

    const { rgb, preview } = extractFromPhoto(rawSourceImg);

    // Show detection preview
    extractCv.width = preview.width;
    extractCv.height = preview.height;
    extractCv.getContext('2d').drawImage(preview, 0, 0);
    outExtract.classList.remove('hidden');
    output.classList.remove('hidden');

    // Replace fileRGB with extracted result
    fileRGB = rgb;
    await showRGB(rgb, 'extracted.png');
    setStatus('Extract: found and corrected → 1024×1024', 'ok');
  } catch (e) {
    setStatus(`Extract error: ${e.message}`, 'err');
    console.error(e);
  }
}

// ─── Events ─────────────────────────────────────────────────────────────────

btnToImg.onclick = doToImage;
btnFromImg.onclick = doFromImage;
btnMul.onclick = doMultiply;
btnHide.onclick = doHideImage;
btnReveal.onclick = doRevealImage;
btnExtract.onclick = doExtract;
btnKeygen.onclick = doKeygen;

function setupDrop(zone, input, handler) {
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag'));
  zone.addEventListener('drop', e => {
    e.preventDefault(); zone.classList.remove('drag');
    if (e.dataTransfer.files.length) handler(e.dataTransfer.files[0]);
  });
  input.addEventListener('change', () => { if (input.files.length) handler(input.files[0]); });
}

setupDrop(dropFile, fileInput, onFile);
setupDrop(dropKey, keyInput, onKey);
setupDrop(dropCover, coverInput, onCover);
updateFourierSpanLabel();
audioMode.addEventListener('change', async () => {
  if (!sourceAudio) return;
  try {
    await encodeSourceAudio();
    setStatus(`Audio mode: ${getAudioModeName()}`, 'ok');
  } catch (e) {
    setStatus(`Error: ${e.message}`, 'err');
    console.error(e);
  }
});
fourierSpan.addEventListener('input', () => {
  fourierAuto.checked = false;
  updateFourierSpanLabel();
});
fourierAuto.addEventListener('change', async () => {
  if (!fourierAuto.checked || !sourceAudio) return;
  try {
    const spanInfo = setFourierSpanTarget(sourceAudio.samples.length / sourceAudio.sampleRate);
    if (audioMode.value === 'fourier') await encodeSourceAudio();
    setStatus(`Fourier auto: ${formatSeconds(spanInfo.seconds)}`, 'ok');
  } catch (e) {
    setStatus(`Error: ${e.message}`, 'err');
    console.error(e);
  }
});
fourierSpan.addEventListener('change', async () => {
  if (!sourceAudio || audioMode.value !== 'fourier') return;
  try {
    await encodeSourceAudio();
    const spanInfo = updateFourierSpanLabel();
    setStatus(`Fourier span: ${formatSeconds(spanInfo.seconds)}`, 'ok');
  } catch (e) {
    setStatus(`Error: ${e.message}`, 'err');
    console.error(e);
  }
});
textInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && e.ctrlKey) { e.preventDefault(); doToImage(); }
});
