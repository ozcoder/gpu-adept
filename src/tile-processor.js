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
 * Group tiles by quality level, batch-encode each group, composite to output.
 * Returns an OffscreenCanvas with the final composited image.
 */
export async function processTiles(imageData, tileQualities, tileSize, onProgress) {
  const { width, height } = imageData;
  const tilesX = Math.ceil(width / tileSize);
  const tilesY = Math.ceil(height / tileSize);

  const srcBitmap = await createImageBitmap(imageData);

  const outCanvas = new OffscreenCanvas(width, height);
  const outCtx = outCanvas.getContext("2d");

  const groups = new Map();
  const edgeTiles = [];

  for (let ty = 0; ty < tilesY; ty++) {
    for (let tx = 0; tx < tilesX; tx++) {
      const tileIdx = ty * tilesX + tx;
      const q = tileQualities[tileIdx];
      const tileX = tx * tileSize;
      const tileY = ty * tileSize;
      const tileW = Math.min(tileSize, width - tileX);
      const tileH = Math.min(tileSize, height - tileY);

      if (q >= OUTPUT_QUALITY) {
        outCtx.drawImage(srcBitmap, tileX, tileY, tileW, tileH, tileX, tileY, tileW, tileH);
        continue;
      }

      const tile = { tx, ty, tileX, tileY, tileW, tileH };
      if (tileW < 8 || tileH < 8) {
        edgeTiles.push({ ...tile, quality: q });
      } else {
        if (!groups.has(q)) groups.set(q, []);
        groups.get(q).push(tile);
      }
    }
  }

  let encodesDone = 0;
  const totalEncodes = groups.size + edgeTiles.length;
  const groupEntries = Array.from(groups);

  for (let gi = 0; gi < groupEntries.length; gi++) {
    const [quality, tiles] = groupEntries[gi];
    const n = tiles.length;
    const cols = Math.ceil(Math.sqrt(n));
    const rows = Math.ceil(n / cols);
    const batchW = cols * tileSize;
    const batchH = rows * tileSize;

    const batchCanvas = new OffscreenCanvas(batchW, batchH);
    const batchCtx = batchCanvas.getContext("2d");

    for (let i = 0; i < n; i++) {
      const t = tiles[i];
      const col = i % cols;
      const row = Math.floor(i / cols);
      batchCtx.drawImage(srcBitmap, t.tileX, t.tileY, t.tileW, t.tileH, col * tileSize, row * tileSize, t.tileW, t.tileH);
    }

    // srcBitmap not needed after the last batch's draw-to-batch-canvas loop
    if (gi === groupEntries.length - 1) srcBitmap.close();

    const blob = await batchCanvas.convertToBlob({ type: "image/jpeg", quality: quality / 100 });
    const bitmap = await createImageBitmap(blob);

    for (let i = 0; i < n; i++) {
      const t = tiles[i];
      const col = i % cols;
      const row = Math.floor(i / cols);
      outCtx.drawImage(bitmap, col * tileSize, row * tileSize, t.tileW, t.tileH, t.tileX, t.tileY, t.tileW, t.tileH);
    }
    bitmap.close();

    encodesDone++;
    if (onProgress) onProgress(`encoding Q${quality} (${encodesDone}/${totalEncodes})`);
  }

  for (const t of edgeTiles) {
    const padded = padTo8(t.tileW, t.tileH);
    const srcPixels = tileImageData(imageData, t.tileX, t.tileY, t.tileW, t.tileH);
    const paddedPixels = mirrorPad(srcPixels, t.tileW, t.tileH, padded.w, padded.h);
    const tileCanvas = new OffscreenCanvas(padded.w, padded.h);
    const tileCtx = tileCanvas.getContext("2d");
    const id = tileCtx.createImageData(padded.w, padded.h);
    id.data.set(paddedPixels);
    tileCtx.putImageData(id, 0, 0);
    const blob = await tileCanvas.convertToBlob({ type: "image/jpeg", quality: t.quality / 100 });
    const bitmap = await createImageBitmap(blob);
    outCtx.drawImage(bitmap, 0, 0, t.tileW, t.tileH, t.tileX, t.tileY, t.tileW, t.tileH);
    bitmap.close();

    encodesDone++;
    if (onProgress) onProgress(`encoding edge tile (${encodesDone}/${totalEncodes})`);
  }

  if (groupEntries.length === 0) srcBitmap.close();
  return outCanvas;
}
