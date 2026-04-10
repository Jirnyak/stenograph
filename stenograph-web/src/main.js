import {
  generateKeyPair, loadImageFile, loadAudioFile, rgbToBlob, multiply,
  textToRGB, rgbToText, rgbToRawText, audioToRGB, rgbToAudio, toWavBlob,
} from './steno.js';
import { extractFromPhoto } from './extract.js';

const $ = s => document.querySelector(s);
const tick = () => new Promise(r => setTimeout(r, 10));

// ─── DOM ────────────────────────────────────────────────────────────────────

const dropFile   = $('#drop-file');
const dropKey    = $('#drop-key');
const fileInput  = $('#file-input');
const keyInput   = $('#key-input');
const textInput  = $('#text-input');
const fileLbl    = $('#file-label');
const fileHint   = $('#file-hint');
const keyLbl     = $('#key-label');
const keyHint    = $('#key-hint');
const btnToImg   = $('#btn-to-img');
const btnFromImg = $('#btn-from-img');
const btnMul     = $('#btn-multiply');
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
let rawSourceImg = null; // original Image element for extract

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

// ─── File classification & loading ──────────────────────────────────────────

function classifyFile(file) {
  const n = file.name.toLowerCase();
  if (file.type.startsWith('audio/') || /\.(wav|mp3|ogg|flac|m4a|aac)$/.test(n)) return 'audio';
  if (file.type.startsWith('image/') || /\.(png|jpg|jpeg|bmp|webp|gif)$/.test(n)) return 'image';
  return 'text';
}

async function onFile(file) {
  const type = classifyFile(file);
  try {
    if (type === 'image') {
      setStatus('Loading image...', '');
      fileRGB = await loadImageFile(file);
      // Keep original image for extract
      rawSourceImg = await loadRawImage(file);
    } else if (type === 'audio') {
      setStatus('Decoding audio...', '');
      const samples = await loadAudioFile(file);
      setStatus('FFT...', '');
      await tick();
      fileRGB = audioToRGB(samples);
    } else {
      setStatus('Encoding text...', '');
      const text = await file.text();
      fileRGB = await textToRGB(text);
    }
    fileLbl.textContent = file.name;
    fileHint.textContent = `${type} → 1024×1024`;
    dropFile.classList.add('loaded');
    setStatus(`Loaded: ${file.name}`, 'ok');
  } catch (e) {
    setStatus(`Error: ${e.message}`, 'err');
    console.error(e);
  }
}

async function onKey(file) {
  try {
    keyRGB = await loadImageFile(file);
    keyLbl.textContent = file.name;
    keyHint.textContent = '1024×1024';
    dropKey.classList.add('loaded');
    btnMul.disabled = false;
    setStatus(`Key: ${file.name}`, 'ok');
  } catch (e) {
    setStatus(`Error: ${e.message}`, 'err');
  }
}

// ─── Actions ────────────────────────────────────────────────────────────────

/** → Image: any input → 1024×1024 PNG. That's it. */
async function doToImage() {
  clearOutput();
  const txt = textInput.value.trim();
  if (txt && !fileRGB) {
    setStatus('Encoding text...', '');
    await tick();
    fileRGB = await textToRGB(txt);
    fileLbl.textContent = 'typed text';
    fileHint.textContent = 'text → 1024×1024';
    dropFile.classList.add('loaded');
  }
  if (!fileRGB) { setStatus('Drop a file or type text', 'err'); return; }

  await showRGB(fileRGB, 'encoded.png');
  setStatus('→ Image: 1024×1024', 'ok');
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

    // 3. Audio — blindly run iFFT, always works
    setStatus('Decoding audio...', '');
    await tick();
    const audio = rgbToAudio(fileRGB);
    showAudio(audio.samples, audio.sampleRate);

    setStatus('← Image: all outputs', 'ok');
  } catch (e) {
    setStatus(`Error: ${e.message}`, 'err');
    console.error(e);
  }
}

/** × Key: image × key → image. Pure multiply. */
async function doMultiply() {
  clearOutput();
  const txt = textInput.value.trim();
  if (txt && !fileRGB) {
    fileRGB = await textToRGB(txt);
  }
  if (!fileRGB) { setStatus('Drop a file', 'err'); return; }
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
textInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && e.ctrlKey) { e.preventDefault(); doToImage(); }
});
