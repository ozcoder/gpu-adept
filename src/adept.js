import { MSSS } from "gpu-msss";
import { TileAnalysisPipeline } from "./tile-analysis.js";
import { processTiles } from "./tile-processor.js";
import { autoTileSize } from "./utils.js";

function binarizeMean(pixels, threshold) {
  const tVal = threshold * 2.55;
  let sum = 0;
  const total = pixels.length / 4;
  for (let i = 0; i < pixels.length; i += 4) {
    sum += pixels[i] > tVal ? 255 : 0;
  }
  return sum / total;
}

function findOptimalThreshold(rawSaliency, width, height) {
  let lower = 0;
  let upper = 100;
  let threshold = 50;

  let mean = binarizeMean(rawSaliency, threshold);
  let iterations = 0;

  while ((mean > 40 || mean < 20) && lower < upper - 1 && iterations < 20) {
    if (mean < 20) {
      upper = threshold;
    } else {
      lower = threshold;
    }
    threshold = Math.floor((upper - lower) / 2 + lower);
    mean = binarizeMean(rawSaliency, threshold);
    iterations++;
  }

  return threshold;
}

export class AdeptJPEG {
  constructor(device, msss) {
    this._device = device;
    this._msss = msss;
    this._tileAnalysis = null;
  }

  static async create() {
    if (!navigator.gpu) {
      throw new Error("WebGPU not available");
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) {
      throw new Error("No WebGPU adapter found");
    }
    const device = await adapter.requestDevice({ requiredLimits: adapter.limits });
    const msss = new MSSS(device);
    return new AdeptJPEG(device, msss);
  }

  async compress(imageData, opts = {}) {
    const { width, height } = imageData;
    const tileSize = opts.tileSize || autoTileSize(width, height);
    const inputQuality = opts.inputQuality || 100;
    const highQuality = opts.highQuality ?? Math.min(inputQuality, 92);
    const lowQuality = opts.lowQuality ?? 69;
    const onProgress = opts.onProgress || null;

    if (onProgress) onProgress("computing saliency");

    const rawSaliency = await this._msss.compute(imageData);

    if (onProgress) onProgress("finding optimal threshold");

    const threshold = findOptimalThreshold(rawSaliency, width, height);

    if (onProgress) onProgress("analyzing tiles");

    this._tileAnalysis = new TileAnalysisPipeline(this._device, width, height, tileSize);
    const tileMeans = await this._tileAnalysis.analyze(rawSaliency, threshold * 2.55);

    if (onProgress) onProgress("encoding tiles");

    const tileQualities = new Uint8Array(tileMeans.length);
    let lowComplexityCount = 0;
    for (let i = 0; i < tileMeans.length; i++) {
      const frac = Math.min(1, tileMeans[i] / 255);
      const q = Math.round(lowQuality + (highQuality - lowQuality) * frac);
      tileQualities[i] = Math.min(q, highQuality);
      if (q < highQuality) lowComplexityCount++;
    }

    const resultData = await processTiles(
      imageData, tileQualities, tileSize,
      highQuality,
      (done, total) => {
        if (onProgress) onProgress(`encoding tiles (${done}/${total})`);
      }
    );

    const outCanvas = new OffscreenCanvas(width, height);
    const outCtx = outCanvas.getContext("2d");
    outCtx.putImageData(resultData, 0, 0);

    if (onProgress) onProgress("generating output");

    const blob = await outCanvas.convertToBlob({
      type: "image/jpeg",
      quality: highQuality / 100,
    });

    return {
      blob,
      tileSize,
      tilesX: Math.ceil(width / tileSize),
      tilesY: Math.ceil(height / tileSize),
      tileQualities,
      lowComplexityCount,
      totalTiles: tileMeans.length,
      highQuality,
      lowQuality,
      threshold,
    };
  }

  destroy() {
    if (this._tileAnalysis) {
      this._tileAnalysis.destroy();
      this._tileAnalysis = null;
    }
    this._msss?.destroy();
  }
}
