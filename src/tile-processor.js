import { padTo8 } from "./utils.js";

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
  for (let y = 0; y < sh; y++) {
    for (let x = 0; x < sw; x++) {
      const si = ((sy + y) * imageData.width + (sx + x)) * 4;
      const di = (y * sw + x) * 4;
      data[di] = imageData.data[si];
      data[di + 1] = imageData.data[si + 1];
      data[di + 2] = imageData.data[si + 2];
      data[di + 3] = imageData.data[si + 3];
    }
  }
  return data;
}

export async function processTiles(
  imageData,
  tileQualities,
  tileSize,
  outputQuality,
  onProgress
) {
  const { width, height } = imageData;
  const tilesX = Math.ceil(width / tileSize);
  const tilesY = Math.ceil(height / tileSize);

  const srcCanvas = new OffscreenCanvas(width, height);
  const srcCtx = srcCanvas.getContext("2d");
  const srcImageData = srcCtx.createImageData(width, height);
  srcImageData.data.set(imageData.data);
  srcCtx.putImageData(srcImageData, 0, 0);

  const outCanvas = new OffscreenCanvas(width, height);
  const outCtx = outCanvas.getContext("2d");

  let progress = 0;
  const totalTiles = tilesX * tilesY;

  const tasks = [];

  for (let ty = 0; ty < tilesY; ty++) {
    for (let tx = 0; tx < tilesX; tx++) {
      const tileX = tx * tileSize;
      const tileY = ty * tileSize;
      const tileW = Math.min(tileSize, width - tileX);
      const tileH = Math.min(tileSize, height - tileY);

      const tileIdx = ty * tilesX + tx;
      const tileQuality = tileQualities[tileIdx];
      const needsEncode = tileQuality < outputQuality;

      tasks.push(async () => {
        if (needsEncode) {
          const needPad = tileW < 8 || tileH < 8;
          if (needPad) {
            const padded = padTo8(tileW, tileH);
            const srcPixels = tileImageData(imageData, tileX, tileY, tileW, tileH);
            const paddedPixels = mirrorPad(srcPixels, tileW, tileH, padded.w, padded.h);
            const tileCanvas = new OffscreenCanvas(padded.w, padded.h);
            const tileCtx = tileCanvas.getContext("2d");
            const id = tileCtx.createImageData(padded.w, padded.h);
            id.data.set(paddedPixels);
            tileCtx.putImageData(id, 0, 0);
            const blob = await tileCanvas.convertToBlob({ type: "image/jpeg", quality: tileQuality / 100 });
            const bitmap = await createImageBitmap(blob);
            outCtx.drawImage(bitmap, 0, 0, tileW, tileH, tileX, tileY, tileW, tileH);
            bitmap.close();
          } else {
            const tileCanvas = new OffscreenCanvas(tileW, tileH);
            const tileCtx = tileCanvas.getContext("2d");
            tileCtx.drawImage(srcCanvas, tileX, tileY, tileW, tileH, 0, 0, tileW, tileH);
            const blob = await tileCanvas.convertToBlob({ type: "image/jpeg", quality: tileQuality / 100 });
            const bitmap = await createImageBitmap(blob);
            outCtx.drawImage(bitmap, tileX, tileY);
            bitmap.close();
          }
        } else {
          outCtx.drawImage(srcCanvas, tileX, tileY, tileW, tileH, tileX, tileY, tileW, tileH);
        }

        progress++;
        if (onProgress) onProgress(progress, totalTiles);
      });
    }
  }

  const BATCH = 32;
  for (let i = 0; i < tasks.length; i += BATCH) {
    const batch = tasks.slice(i, i + BATCH);
    await Promise.all(batch.map((fn) => fn()));
  }

  return outCtx.getImageData(0, 0, width, height);
}
