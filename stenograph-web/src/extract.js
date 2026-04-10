/**
 * STENOGRAPH — Raw Image Extractor
 *
 * Takes a photo of a screen (or scan, or camera capture) containing
 * a 1024×1024 stenograph image and extracts it back to clean 1024×1024.
 *
 * Key insight: the encoded image is uniform random noise (~mean 128).
 * A dark background (screen bezel, desk, etc.) is much darker.
 * We blur heavily so noise averages out → bright uniform blob,
 * then threshold to find the rectangle.
 *
 * Sampling MUST be nearest-neighbor — bilinear/bicubic would blend
 * adjacent noise pixels and destroy the encoded data.
 */

const N = 1024;

/**
 * Extract a stenograph image from a raw photo.
 * @param {HTMLImageElement|HTMLCanvasElement} source
 * @returns {{ rgb: Uint8Array, preview: HTMLCanvasElement }}
 */
export function extractFromPhoto(source) {
  let sw, sh;
  if (source instanceof HTMLImageElement) {
    sw = source.naturalWidth;
    sh = source.naturalHeight;
  } else {
    sw = source.width;
    sh = source.height;
  }

  // Draw source to full-res canvas to get pixel data
  const fullCv = makeCanvas(sw, sh);
  fullCv.ctx.drawImage(source, 0, 0, sw, sh);
  const fullData = fullCv.ctx.getImageData(0, 0, sw, sh).data;

  // Work at reduced resolution for detection (speed)
  const maxWork = 512;
  const scale = Math.min(1, maxWork / Math.max(sw, sh));
  const ww = Math.round(sw * scale);
  const wh = Math.round(sh * scale);

  const workCv = makeCanvas(ww, wh);
  workCv.ctx.drawImage(source, 0, 0, ww, wh);
  const workData = workCv.ctx.getImageData(0, 0, ww, wh).data;

  // 1. Convert to grayscale
  const gray = new Float32Array(ww * wh);
  for (let i = 0; i < ww * wh; i++) {
    gray[i] = workData[i * 4] * 0.299 + workData[i * 4 + 1] * 0.587 + workData[i * 4 + 2] * 0.114;
  }

  // 2. Heavy box blur to average out noise (radius ~5% of image size)
  const blurR = Math.max(5, Math.round(Math.min(ww, wh) * 0.05));
  const blurred = boxBlur(gray, ww, wh, blurR);

  // 3. Otsu threshold on blurred image
  const blurU8 = new Uint8Array(ww * wh);
  for (let i = 0; i < blurred.length; i++) blurU8[i] = Math.round(Math.max(0, Math.min(255, blurred[i])));
  const thresh = otsu(blurU8);

  const binary = new Uint8Array(ww * wh);
  for (let i = 0; i < blurU8.length; i++) binary[i] = blurU8[i] > thresh ? 1 : 0;

  // 4. Morphological close to clean up edges
  const closed = morphClose(binary, ww, wh, 2);

  // 5. Largest connected component
  const { mask } = largestComponent(closed, ww, wh);

  // 6. Find tight bounding quad from mask edge scans + line fitting
  const quad = findQuadFromMask(mask, ww, wh);

  // 7. Order corners TL, TR, BR, BL
  const ordered = orderCorners(quad);

  // 8. Map back to full resolution
  const origCorners = ordered.map(p => ({ x: p.x / scale, y: p.y / scale }));

  // 9. Perspective-correct with NEAREST NEIGHBOR sampling
  const rgb = perspectiveExtractNN(fullData, sw, sh, origCorners, N);

  // 10. Debug preview
  const preview = makeCanvas(ww, wh);
  preview.ctx.drawImage(source, 0, 0, ww, wh);
  preview.ctx.strokeStyle = '#0f0';
  preview.ctx.lineWidth = 2;
  preview.ctx.beginPath();
  preview.ctx.moveTo(ordered[0].x, ordered[0].y);
  for (let i = 1; i <= 4; i++) preview.ctx.lineTo(ordered[i % 4].x, ordered[i % 4].y);
  preview.ctx.stroke();
  for (let i = 0; i < 4; i++) {
    preview.ctx.fillStyle = ['#f00', '#0f0', '#00f', '#ff0'][i];
    preview.ctx.beginPath();
    preview.ctx.arc(ordered[i].x, ordered[i].y, 4, 0, Math.PI * 2);
    preview.ctx.fill();
  }

  return { rgb, preview: preview.canvas };
}


// ─── Canvas helper ──────────────────────────────────────────────────────────

function makeCanvas(w, h) {
  const canvas = document.createElement('canvas');
  canvas.width = w; canvas.height = h;
  return { canvas, ctx: canvas.getContext('2d') };
}


// ─── Box blur (separable, 2-pass) ───────────────────────────────────────────

function boxBlur(src, w, h, r) {
  const tmp = new Float32Array(w * h);
  const dst = new Float32Array(w * h);
  const d = 2 * r + 1;

  // Horizontal pass
  for (let y = 0; y < h; y++) {
    let sum = 0;
    for (let x = -r; x <= r; x++) sum += src[y * w + Math.max(0, Math.min(w - 1, x))];
    for (let x = 0; x < w; x++) {
      tmp[y * w + x] = sum / d;
      const left = Math.max(0, x - r);
      const right = Math.min(w - 1, x + r + 1);
      sum += src[y * w + right] - src[y * w + left];
    }
  }

  // Vertical pass
  for (let x = 0; x < w; x++) {
    let sum = 0;
    for (let y = -r; y <= r; y++) sum += tmp[Math.max(0, Math.min(h - 1, y)) * w + x];
    for (let y = 0; y < h; y++) {
      dst[y * w + x] = sum / d;
      const top = Math.max(0, y - r);
      const bot = Math.min(h - 1, y + r + 1);
      sum += tmp[bot * w + x] - tmp[top * w + x];
    }
  }

  return dst;
}


// ─── Otsu's threshold ───────────────────────────────────────────────────────

function otsu(gray) {
  const hist = new Int32Array(256);
  for (let i = 0; i < gray.length; i++) hist[gray[i]]++;

  const total = gray.length;
  let sumAll = 0;
  for (let i = 0; i < 256; i++) sumAll += i * hist[i];

  let sumBg = 0, wBg = 0, best = 0, threshold = 0;
  for (let t = 0; t < 256; t++) {
    wBg += hist[t];
    if (wBg === 0) continue;
    const wFg = total - wBg;
    if (wFg === 0) break;
    sumBg += t * hist[t];
    const meanBg = sumBg / wBg;
    const meanFg = (sumAll - sumBg) / wFg;
    const between = wBg * wFg * (meanBg - meanFg) * (meanBg - meanFg);
    if (between > best) { best = between; threshold = t; }
  }
  return threshold;
}


// ─── Morphological close ────────────────────────────────────────────────────

function morphClose(bin, w, h, r) {
  let d = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let found = false;
      for (let dy = -r; dy <= r && !found; dy++) {
        for (let dx = -r; dx <= r && !found; dx++) {
          const nx = x + dx, ny = y + dy;
          if (nx >= 0 && nx < w && ny >= 0 && ny < h && bin[ny * w + nx]) found = true;
        }
      }
      d[y * w + x] = found ? 1 : 0;
    }
  }
  let e = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let all = true;
      for (let dy = -r; dy <= r && all; dy++) {
        for (let dx = -r; dx <= r && all; dx++) {
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || nx >= w || ny < 0 || ny >= h || !d[ny * w + nx]) all = false;
        }
      }
      e[y * w + x] = all ? 1 : 0;
    }
  }
  return e;
}


// ─── Largest connected component (BFS) ──────────────────────────────────────

function largestComponent(bin, w, h) {
  const labels = new Int32Array(w * h);
  labels.fill(-1);
  let bestLabel = -1, bestSize = 0;
  let label = 0;

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      if (!bin[idx] || labels[idx] >= 0) continue;

      const queue = [idx];
      labels[idx] = label;
      let size = 0, head = 0;

      while (head < queue.length) {
        const ci = queue[head++];
        size++;
        const cx = ci % w, cy = (ci / w) | 0;
        const neighbors = [
          cy > 0 ? ci - w : -1,
          cy < h - 1 ? ci + w : -1,
          cx > 0 ? ci - 1 : -1,
          cx < w - 1 ? ci + 1 : -1
        ];
        for (const ni of neighbors) {
          if (ni >= 0 && bin[ni] && labels[ni] < 0) {
            labels[ni] = label;
            queue.push(ni);
          }
        }
      }

      if (size > bestSize) { bestSize = size; bestLabel = label; }
      label++;
    }
  }

  const mask = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) {
    if (labels[i] === bestLabel) mask[i] = 1;
  }
  return { mask };
}


// ─── Find quad from binary mask ─────────────────────────────────────────────

/**
 * Find quadrilateral corners from binary mask.
 * Scan each row to find leftmost/rightmost active pixel,
 * scan each column for topmost/bottommost.
 * Fit lines to each edge with least squares (trimming outliers).
 * Intersect the 4 lines → 4 corners.
 */
function findQuadFromMask(mask, w, h) {
  const leftEdge = [];
  const rightEdge = [];
  const topEdge = [];
  const bottomEdge = [];

  for (let y = 0; y < h; y++) {
    let first = -1, last = -1;
    for (let x = 0; x < w; x++) {
      if (mask[y * w + x]) {
        if (first < 0) first = x;
        last = x;
      }
    }
    if (first >= 0) {
      leftEdge.push({ x: first, y });
      rightEdge.push({ x: last, y });
    }
  }

  for (let x = 0; x < w; x++) {
    let first = -1, last = -1;
    for (let y = 0; y < h; y++) {
      if (mask[y * w + x]) {
        if (first < 0) first = y;
        last = y;
      }
    }
    if (first >= 0) {
      topEdge.push({ x, y: first });
      bottomEdge.push({ x, y: last });
    }
  }

  const leftLine = fitLineXofY(leftEdge);
  const rightLine = fitLineXofY(rightEdge);
  const topLine = fitLineYofX(topEdge);
  const bottomLine = fitLineYofX(bottomEdge);

  const tl = intersectXY(leftLine, topLine);
  const tr = intersectXY(rightLine, topLine);
  const br = intersectXY(rightLine, bottomLine);
  const bl = intersectXY(leftLine, bottomLine);

  return [tl, tr, br, bl];
}

function fitLineXofY(points) {
  const sorted = points.slice().sort((a, b) => a.y - b.y);
  const trim = Math.round(sorted.length * 0.1);
  const pts = sorted.slice(trim, sorted.length - trim);
  if (pts.length < 2) return { a: 0, b: points[0]?.x ?? 0 };

  let sumY = 0, sumX = 0, sumYY = 0, sumYX = 0;
  for (const p of pts) { sumY += p.y; sumX += p.x; sumYY += p.y * p.y; sumYX += p.y * p.x; }
  const n = pts.length;
  const denom = n * sumYY - sumY * sumY;
  if (Math.abs(denom) < 1e-10) return { a: 0, b: sumX / n };
  const a = (n * sumYX - sumY * sumX) / denom;
  const b = (sumX - a * sumY) / n;
  return { a, b };
}

function fitLineYofX(points) {
  const sorted = points.slice().sort((a, b) => a.x - b.x);
  const trim = Math.round(sorted.length * 0.1);
  const pts = sorted.slice(trim, sorted.length - trim);
  if (pts.length < 2) return { a: 0, b: points[0]?.y ?? 0 };

  let sumX = 0, sumY = 0, sumXX = 0, sumXY = 0;
  for (const p of pts) { sumX += p.x; sumY += p.y; sumXX += p.x * p.x; sumXY += p.x * p.y; }
  const n = pts.length;
  const denom = n * sumXX - sumX * sumX;
  if (Math.abs(denom) < 1e-10) return { a: 0, b: sumY / n };
  const a = (n * sumXY - sumX * sumY) / denom;
  const b = (sumY - a * sumX) / n;
  return { a, b };
}

function intersectXY(vLine, hLine) {
  const denom = 1 - vLine.a * hLine.a;
  if (Math.abs(denom) < 1e-10) return { x: vLine.b, y: hLine.b };
  const x = (vLine.a * hLine.b + vLine.b) / denom;
  const y = hLine.a * x + hLine.b;
  return { x, y };
}


// ─── Order corners: TL, TR, BR, BL ─────────────────────────────────────────

function orderCorners(pts) {
  let cx = 0, cy = 0;
  for (const p of pts) { cx += p.x; cy += p.y; }
  cx /= pts.length; cy /= pts.length;

  const sorted = pts.slice().sort((a, b) => {
    const aa = Math.atan2(a.y - cy, a.x - cx);
    const ab = Math.atan2(b.y - cy, b.x - cx);
    return aa - ab;
  });

  let tlIdx = 0, minSum = Infinity;
  for (let i = 0; i < sorted.length; i++) {
    const s = sorted[i].x + sorted[i].y;
    if (s < minSum) { minSum = s; tlIdx = i; }
  }

  const result = [];
  for (let i = 0; i < sorted.length; i++) result.push(sorted[(tlIdx + i) % sorted.length]);
  return result;
}


// ─── Perspective transform (NEAREST NEIGHBOR) ──────────────────────────────

function perspectiveExtractNN(srcData, sw, sh, corners, size) {
  const [tl, tr, br, bl] = corners;
  const H = computePerspectiveMatrix(
    [{ x: 0, y: 0 }, { x: size - 1, y: 0 }, { x: size - 1, y: size - 1 }, { x: 0, y: size - 1 }],
    [tl, tr, br, bl]
  );

  const rgb = new Uint8Array(size * size * 3);
  for (let dy = 0; dy < size; dy++) {
    for (let dx = 0; dx < size; dx++) {
      const denom = H[6] * dx + H[7] * dy + H[8];
      const sx = Math.round((H[0] * dx + H[1] * dy + H[2]) / denom);
      const sy = Math.round((H[3] * dx + H[4] * dy + H[5]) / denom);

      const i = (dy * size + dx) * 3;
      const cx = Math.max(0, Math.min(sw - 1, sx));
      const cy = Math.max(0, Math.min(sh - 1, sy));
      const si = (cy * sw + cx) * 4;
      rgb[i]     = srcData[si];
      rgb[i + 1] = srcData[si + 1];
      rgb[i + 2] = srcData[si + 2];
    }
  }
  return rgb;
}


// ─── Perspective matrix ─────────────────────────────────────────────────────

function computePerspectiveMatrix(dst, src) {
  const A = [];
  const b = [];
  for (let i = 0; i < 4; i++) {
    const dx = dst[i].x, dy = dst[i].y;
    const sx = src[i].x, sy = src[i].y;
    A.push([dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy]);
    b.push(sx);
    A.push([0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy]);
    b.push(sy);
  }

  const n = 8;
  const M = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    let maxVal = Math.abs(M[col][col]), maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(M[row][col]) > maxVal) { maxVal = Math.abs(M[row][col]); maxRow = row; }
    }
    [M[col], M[maxRow]] = [M[maxRow], M[col]];
    for (let row = col + 1; row < n; row++) {
      const f = M[row][col] / M[col][col];
      for (let j = col; j <= n; j++) M[row][j] -= f * M[col][j];
    }
  }

  const h = new Array(n);
  for (let i = n - 1; i >= 0; i--) {
    h[i] = M[i][n];
    for (let j = i + 1; j < n; j++) h[i] -= M[i][j] * h[j];
    h[i] /= M[i][i];
  }
  return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0];
}
export function extractFromPhoto() { return null; }
