import { padTo8 } from "./utils.js";

const OUTPUT_QUALITY = 96;

function mirrorPad(pixels, srcW, srcH, dstW, dstH) {
  const out = new Uint8ClampedArray(dstW * dstH * 4);
  for (let y = 0; y < dstH; y++) {
    for (let x = 0; x < dstW; x++) {
      const sx = x < srcW ? x : srcW - (x - srcW) - 1;
      const sy = y < srcH ? y : srcH - (y - srcH) - 1;
      const si = (sy * srcW + sx) * 4;
      const di = (y * dstW + x) * 4;
      out[di] = pixels[si];
      out[di + 1] = pixels[si + 1];
      out[di + 2] = pixels[si + 2];
      out[di + 3] = pixels[si + 3];
    }
  }
  return out;
}

function tileImageData(imageData, sx, sy, sw, sh) {
  const data = new Uint8ClampedArray(sw * sh * 4);
  const src = imageData.data;
  const rowStride = imageData.width * 4;
  for (let y = 0; y < sh; y++) {
    const srcStart = (sy + y) * rowStride + sx * 4;
    data.set(src.subarray(srcStart, srcStart + sw * 4), y * sw * 4);
  }
  return data;
}

/**
 * Encode an array of leaves (variable size 8/16/32, plus truncated edge
 * leaves) as JPEG, batched by (size, quality). Returns an OffscreenCanvas
 * with the final composited image. Each leaf has { x, y, w, h, quality }.
 */
export async function processTiles(imageData, leaves, onProgress) {
  const t0 = performance.now();
  const { width, height } = imageData;

  const srcBitmap = await createImageBitmap(imageData);
  const t1 = performance.now();

  const outCanvas = new OffscreenCanvas(width, height);
  const outCtx = outCanvas.getContext("2d");

  const groups = new Map();
  const edgeTiles = [];

  for (const leaf of leaves) {
    const { x, y, w, h, quality } = leaf;

    if (quality >= OUTPUT_QUALITY) {
      outCtx.drawImage(srcBitmap, x, y, w, h, x, y, w, h);
      continue;
    }

    if (w < 8 || h < 8) {
      edgeTiles.push(leaf);
    } else {
      const size = Math.max(w, h);
      const key = `${size}_${quality}`;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(leaf);
    }
  }

  let encodesDone = 0;
  const totalEncodes = groups.size + edgeTiles.length;
  const groupEntries = Array.from(groups);

  for (let gi = 0; gi < groupEntries.length; gi++) {
    const [key, tiles] = groupEntries[gi];
    const size = parseInt(key.slice(0, key.indexOf("_")), 10);
    const quality = parseInt(key.slice(key.indexOf("_") + 1), 10);
    const n = tiles.length;
    const cols = Math.ceil(Math.sqrt(n));
    const rows = Math.ceil(n / cols);
    const batchW = cols * size;
    const batchH = rows * size;

    const batchCanvas = new OffscreenCanvas(batchW, batchH);
    const batchCtx = batchCanvas.getContext("2d");

    for (let i = 0; i < n; i++) {
      const t = tiles[i];
      const col = i % cols;
      const row = Math.floor(i / cols);
      batchCtx.drawImage(srcBitmap, t.x, t.y, t.w, t.h, col * size, row * size, t.w, t.h);
    }

    if (gi === groupEntries.length - 1) srcBitmap.close();

    const blob = await batchCanvas.convertToBlob({ type: "image/jpeg", quality: quality / 100 });
    const bitmap = await createImageBitmap(blob);

    for (let i = 0; i < n; i++) {
      const t = tiles[i];
      const col = i % cols;
      const row = Math.floor(i / cols);
      outCtx.drawImage(bitmap, col * size, row * size, t.w, t.h, t.x, t.y, t.w, t.h);
    }
    bitmap.close();

    encodesDone++;
    if (onProgress) onProgress(`encoding ${size}px Q${quality} (${encodesDone}/${totalEncodes})`);
  }

  for (const t of edgeTiles) {
    const padded = padTo8(t.w, t.h);
    const srcPixels = tileImageData(imageData, t.x, t.y, t.w, t.h);
    const paddedPixels = mirrorPad(srcPixels, t.w, t.h, padded.w, padded.h);
    const tileCanvas = new OffscreenCanvas(padded.w, padded.h);
    const tileCtx = tileCanvas.getContext("2d");
    const id = tileCtx.createImageData(padded.w, padded.h);
    id.data.set(paddedPixels);
    tileCtx.putImageData(id, 0, 0);
    const blob = await tileCanvas.convertToBlob({ type: "image/jpeg", quality: t.quality / 100 });
    const bitmap = await createImageBitmap(blob);
    outCtx.drawImage(bitmap, 0, 0, t.w, t.h, t.x, t.y, t.w, t.h);
    bitmap.close();

    encodesDone++;
    if (onProgress) onProgress(`encoding edge tile (${encodesDone}/${totalEncodes})`);
  }

  if (groupEntries.length === 0) srcBitmap.close();
  const tEnd = performance.now();
  console.log(
    `[processTiles] bitmap=${(t1-t0).toFixed(0)} ` +
    `groups=${groups.size} ` +
    `edges=${edgeTiles.length} ` +
    `batchEncode=${(tEnd-t1).toFixed(0)} ` +
    `total=${(tEnd-t0).toFixed(0)} ms`
  );
  return outCanvas;
}
