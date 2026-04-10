/**
 * STENOGRAPH — Raw Image Extractor
 *
 * Takes a photo of a screen (or scan, or camera capture) containing
 * a 1024×1024 stenograph image and extracts it back to clean 1024×1024.
 *
 * Algorithm:
 * 1. Convert to grayscale
 * 2. Adaptive threshold to find the bright noisy rectangle vs dark background
 * 3. Find the largest connected bright region (the stenograph square)
 * 4. Find convex hull → approximate to quadrilateral
 * 5. Perspective-correct the quad to 1024×1024
 *
 * Assumptions:
 * - The stenograph square occupies a large portion of the photo
 * - Background is darker than the encoded noise image
 * - Moderate rotation/perspective/distortion is OK
 */

const N = 1024;

/**
 * Extract a stenograph image from a raw photo.
 * @param {HTMLImageElement|HTMLCanvasElement} source — the raw photo
 * @returns {{ rgb: Uint8Array, preview: HTMLCanvasElement }} — 1024×1024 RGB + debug preview
 */
export function extractFromPhoto(source) {
  // 1. Draw source to working canvas at manageable resolution
  const maxWork = 1024;
  let sw, sh;
  if (source instanceof HTMLImageElement) {
    sw = source.naturalWidth;
    sh = source.naturalHeight;
  } else {
    sw = source.width;
    sh = source.height;
  }

  const scale = Math.min(1, maxWork / Math.max(sw, sh));
  const ww = Math.round(sw * scale);
  const wh = Math.round(sh * scale);

  const wc = document.createElement('canvas');
  wc.width = ww; wc.height = wh;
  const wctx = wc.getContext('2d');
  wctx.drawImage(source, 0, 0, ww, wh);
  const imgData = wctx.getImageData(0, 0, ww, wh);
  const pixels = imgData.data; // RGBA

  // 2. Convert to grayscale
  const gray = new Uint8Array(ww * wh);
  for (let i = 0; i < ww * wh; i++) {
    gray[i] = Math.round(
      pixels[i * 4] * 0.299 +
      pixels[i * 4 + 1] * 0.587 +
      pixels[i * 4 + 2] * 0.114
    );
  }

  // 3. Otsu threshold to separate bright region from dark background
  const threshold = otsu(gray);
  const binary = new Uint8Array(ww * wh);
  for (let i = 0; i < gray.length; i++) {
    binary[i] = gray[i] > threshold ? 1 : 0;
  }

  // 4. Morphological close (dilate then erode) to fill small holes
  const closed = morphClose(binary, ww, wh, 3);

  // 5. Find largest connected component
  const { mask, bbox } = largestComponent(closed, ww, wh);

  // 6. Find contour of the largest component
  const contour = traceContour(mask, ww, wh, bbox);

  // 7. Find convex hull
  const hull = convexHull(contour);

  // 8. Approximate hull to 4 corners (quadrilateral)
  const quad = hullToQuad(hull, ww, wh);

  // 9. Order corners: TL, TR, BR, BL
  const ordered = orderCorners(quad);

  // 10. Map corners back to original resolution
  const origCorners = ordered.map(p => ({
    x: p.x / scale,
    y: p.y / scale
  }));

  // 11. Perspective transform from source image to 1024×1024
  const result = perspectiveExtract(source, sw, sh, origCorners, N);

  // 12. Build debug preview
  const preview = document.createElement('canvas');
  preview.width = ww; preview.height = wh;
  const pctx = preview.getContext('2d');
  pctx.drawImage(source, 0, 0, ww, wh);
  pctx.strokeStyle = '#0f0';
  pctx.lineWidth = 2;
  pctx.beginPath();
  pctx.moveTo(ordered[0].x, ordered[0].y);
  for (let i = 1; i <= 4; i++) {
    pctx.lineTo(ordered[i % 4].x, ordered[i % 4].y);
  }
  pctx.stroke();
  // Draw corner dots
  for (let i = 0; i < 4; i++) {
    pctx.fillStyle = ['#f00', '#0f0', '#00f', '#ff0'][i];
    pctx.beginPath();
    pctx.arc(ordered[i].x, ordered[i].y, 4, 0, Math.PI * 2);
    pctx.fill();
  }

  return { rgb: result, preview };
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
  // Dilate
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
  // Erode
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

      // BFS
      const queue = [idx];
      labels[idx] = label;
      let size = 0;
      let head = 0;
      let minX = x, maxX = x, minY = y, maxY = y;

      while (head < queue.length) {
        const ci = queue[head++];
        size++;
        const cx = ci % w, cy = (ci / w) | 0;
        if (cx < minX) minX = cx;
        if (cx > maxX) maxX = cx;
        if (cy < minY) minY = cy;
        if (cy > maxY) maxY = cy;

        // 4-connected neighbors
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

      if (size > bestSize) {
        bestSize = size;
        bestLabel = label;
      }
      label++;
    }
  }

  // Build mask for best component
  const mask = new Uint8Array(w * h);
  let minX = w, maxX = 0, minY = h, maxY = 0;
  for (let i = 0; i < w * h; i++) {
    if (labels[i] === bestLabel) {
      mask[i] = 1;
      const x = i % w, y = (i / w) | 0;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }

  return { mask, bbox: { minX, maxX, minY, maxY } };
}


// ─── Contour tracing ────────────────────────────────────────────────────────

function traceContour(mask, w, h, bbox) {
  const points = [];
  const { minX, maxX, minY, maxY } = bbox;

  // Scan border pixels (pixels that are 1 with at least one 0-neighbor)
  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      if (!mask[y * w + x]) continue;
      // Check if border
      let border = false;
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) border = true;
      else {
        if (!mask[(y - 1) * w + x] || !mask[(y + 1) * w + x] ||
            !mask[y * w + x - 1] || !mask[y * w + x + 1]) border = true;
      }
      if (border) points.push({ x, y });
    }
  }

  return points;
}


// ─── Convex hull (Andrew's monotone chain) ──────────────────────────────────

function convexHull(points) {
  if (points.length < 3) return points.slice();
  const pts = points.slice().sort((a, b) => a.x - b.x || a.y - b.y);

  const cross = (O, A, B) =>
    (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);

  // Lower hull
  const lower = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0)
      lower.pop();
    lower.push(p);
  }
  // Upper hull
  const upper = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0)
      upper.pop();
    upper.push(p);
  }

  lower.pop();
  upper.pop();
  return lower.concat(upper);
}


// ─── Hull → Quadrilateral ───────────────────────────────────────────────────

/**
 * Reduce convex hull to 4-point quadrilateral.
 * Uses Ramer-Douglas-Peucker until we have exactly 4 points.
 */
function hullToQuad(hull, w, h) {
  if (hull.length <= 4) {
    while (hull.length < 4) hull.push(hull[hull.length - 1]);
    return hull;
  }

  // Binary search for epsilon that gives ~4 points
  let lo = 0, hi = Math.max(w, h) * 0.5;
  let best = hull;

  for (let iter = 0; iter < 50; iter++) {
    const mid = (lo + hi) / 2;
    const simplified = rdpClosed(hull, mid);
    if (simplified.length > 4) {
      lo = mid;
    } else if (simplified.length < 4) {
      hi = mid;
    } else {
      best = simplified;
      break;
    }
    if (simplified.length === 4) best = simplified;
    // Keep the closest to 4
    if (simplified.length >= 4 && (best.length > 4 || best.length < 4)) best = simplified;
  }

  // If we still don't have exactly 4, pick the 4 corners that maximize area
  if (best.length !== 4) {
    best = pick4Corners(hull);
  }

  return best;
}

/**
 * Ramer-Douglas-Peucker for closed polygon.
 */
function rdpClosed(points, epsilon) {
  if (points.length <= 4) return points.slice();

  // Find the two farthest points
  let maxDist = 0, idxA = 0, idxB = 0;
  for (let i = 0; i < points.length; i++) {
    for (let j = i + 1; j < points.length; j++) {
      const d = dist(points[i], points[j]);
      if (d > maxDist) { maxDist = d; idxA = i; idxB = j; }
    }
  }

  // Split into two chains and simplify each
  const chain1 = [], chain2 = [];
  for (let i = idxA; ; i = (i + 1) % points.length) {
    chain1.push(points[i]);
    if (i === idxB) break;
  }
  for (let i = idxB; ; i = (i + 1) % points.length) {
    chain2.push(points[i]);
    if (i === idxA) break;
  }

  const s1 = rdpLine(chain1, epsilon);
  const s2 = rdpLine(chain2, epsilon);

  // Merge, removing duplicates at junctions
  const result = s1.slice(0, -1).concat(s2.slice(0, -1));
  return result;
}

function rdpLine(points, epsilon) {
  if (points.length <= 2) return points.slice();

  let maxDist = 0, idx = 0;
  const a = points[0], b = points[points.length - 1];

  for (let i = 1; i < points.length - 1; i++) {
    const d = pointToLineDist(points[i], a, b);
    if (d > maxDist) { maxDist = d; idx = i; }
  }

  if (maxDist <= epsilon) return [a, b];

  const left = rdpLine(points.slice(0, idx + 1), epsilon);
  const right = rdpLine(points.slice(idx), epsilon);
  return left.slice(0, -1).concat(right);
}

function dist(a, b) {
  const dx = a.x - b.x, dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function pointToLineDist(p, a, b) {
  const dx = b.x - a.x, dy = b.y - a.y;
  const lenSq = dx * dx + dy * dy;
  if (lenSq === 0) return dist(p, a);
  const num = Math.abs(dy * p.x - dx * p.y + b.x * a.y - b.y * a.x);
  return num / Math.sqrt(lenSq);
}

/**
 * Pick 4 points from hull that maximize enclosed quadrilateral area.
 * Greedy: start with 4 extreme points.
 */
function pick4Corners(hull) {
  // Find extreme points
  let topIdx = 0, botIdx = 0, leftIdx = 0, rightIdx = 0;
  for (let i = 1; i < hull.length; i++) {
    if (hull[i].y < hull[topIdx].y) topIdx = i;
    if (hull[i].y > hull[botIdx].y) botIdx = i;
    if (hull[i].x < hull[leftIdx].x) leftIdx = i;
    if (hull[i].x > hull[rightIdx].x) rightIdx = i;
  }

  // Get unique indices
  const idxSet = new Set([topIdx, botIdx, leftIdx, rightIdx]);
  const indices = [...idxSet];

  // If we got 4 unique, return them
  if (indices.length === 4) {
    return indices.map(i => hull[i]);
  }

  // Fallback: pick 4 evenly spaced points
  const step = hull.length / 4;
  return [0, 1, 2, 3].map(i => hull[Math.round(i * step) % hull.length]);
}


// ─── Order corners: TL, TR, BR, BL ─────────────────────────────────────────

function orderCorners(pts) {
  // Centroid
  let cx = 0, cy = 0;
  for (const p of pts) { cx += p.x; cy += p.y; }
  cx /= pts.length; cy /= pts.length;

  // Sort by angle from centroid
  const sorted = pts.slice().sort((a, b) => {
    const aa = Math.atan2(a.y - cy, a.x - cx);
    const ab = Math.atan2(b.y - cy, b.x - cx);
    return aa - ab;
  });

  // Find top-left: smallest x+y sum
  let tlIdx = 0;
  let minSum = Infinity;
  for (let i = 0; i < sorted.length; i++) {
    const s = sorted[i].x + sorted[i].y;
    if (s < minSum) { minSum = s; tlIdx = i; }
  }

  // Rotate so TL is first
  const result = [];
  for (let i = 0; i < sorted.length; i++) {
    result.push(sorted[(tlIdx + i) % sorted.length]);
  }

  return result; // TL, TR, BR, BL (clockwise)
}


// ─── Perspective transform ──────────────────────────────────────────────────

/**
 * Extract perspective-corrected N×N image from source.
 * @param {HTMLImageElement|HTMLCanvasElement} source
 * @param {number} sw — source width
 * @param {number} sh — source height
 * @param {Array<{x,y}>} corners — [TL, TR, BR, BL] in source coords
 * @param {number} size — output square size (1024)
 * @returns {Uint8Array} — RGB flat array, size×size×3
 */
function perspectiveExtract(source, sw, sh, corners, size) {
  // Draw source at full resolution
  const sc = document.createElement('canvas');
  sc.width = sw; sc.height = sh;
  const sctx = sc.getContext('2d');
  sctx.drawImage(source, 0, 0, sw, sh);
  const srcData = sctx.getImageData(0, 0, sw, sh).data;

  // Compute the 3×3 perspective transform matrix
  // Maps destination (0..size-1, 0..size-1) → source corners
  const [tl, tr, br, bl] = corners;
  const H = computePerspectiveMatrix(
    // dst corners
    [{ x: 0, y: 0 }, { x: size - 1, y: 0 }, { x: size - 1, y: size - 1 }, { x: 0, y: size - 1 }],
    // src corners
    [tl, tr, br, bl]
  );

  // Sample source for each destination pixel
  const rgb = new Uint8Array(size * size * 3);
  for (let dy = 0; dy < size; dy++) {
    for (let dx = 0; dx < size; dx++) {
      // Apply perspective transform
      const denom = H[6] * dx + H[7] * dy + H[8];
      const sx = (H[0] * dx + H[1] * dy + H[2]) / denom;
      const sy = (H[3] * dx + H[4] * dy + H[5]) / denom;

      // Bilinear interpolation
      const x0 = Math.floor(sx), y0 = Math.floor(sy);
      const x1 = x0 + 1, y1 = y0 + 1;
      const fx = sx - x0, fy = sy - y0;

      const i = (dy * size + dx) * 3;

      if (x0 >= 0 && x1 < sw && y0 >= 0 && y1 < sh) {
        for (let c = 0; c < 3; c++) {
          const v00 = srcData[(y0 * sw + x0) * 4 + c];
          const v10 = srcData[(y0 * sw + x1) * 4 + c];
          const v01 = srcData[(y1 * sw + x0) * 4 + c];
          const v11 = srcData[(y1 * sw + x1) * 4 + c];
          const top = v00 + (v10 - v00) * fx;
          const bot = v01 + (v11 - v01) * fx;
          rgb[i + c] = Math.round(top + (bot - top) * fy);
        }
      } else {
        // Clamp to nearest edge
        const cx = Math.max(0, Math.min(sw - 1, Math.round(sx)));
        const cy = Math.max(0, Math.min(sh - 1, Math.round(sy)));
        const si = (cy * sw + cx) * 4;
        rgb[i] = srcData[si];
        rgb[i + 1] = srcData[si + 1];
        rgb[i + 2] = srcData[si + 2];
      }
    }
  }

  return rgb;
}

/**
 * Compute 3×3 perspective transform matrix mapping dst→src.
 * dst and src are arrays of 4 points: [TL, TR, BR, BL].
 *
 * Solves: for each pair (dx,dy) → (sx,sy):
 *   sx = (h0*dx + h1*dy + h2) / (h6*dx + h7*dy + 1)
 *   sy = (h3*dx + h4*dy + h5) / (h6*dx + h7*dy + 1)
 *
 * This gives 8 equations for 8 unknowns (h0..h7), h8=1.
 */
function computePerspectiveMatrix(dst, src) {
  // Build 8×8 system: A * h = b
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

  // Gaussian elimination with partial pivoting
  const n = 8;
  const M = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    // Find pivot
    let maxVal = Math.abs(M[col][col]), maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(M[row][col]) > maxVal) {
        maxVal = Math.abs(M[row][col]);
        maxRow = row;
      }
    }
    [M[col], M[maxRow]] = [M[maxRow], M[col]];

    // Eliminate below
    for (let row = col + 1; row < n; row++) {
      const f = M[row][col] / M[col][col];
      for (let j = col; j <= n; j++) {
        M[row][j] -= f * M[col][j];
      }
    }
  }

  // Back-substitution
  const h = new Array(n);
  for (let i = n - 1; i >= 0; i--) {
    h[i] = M[i][n];
    for (let j = i + 1; j < n; j++) {
      h[i] -= M[i][j] * h[j];
    }
    h[i] /= M[i][i];
  }

  return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0];
}
