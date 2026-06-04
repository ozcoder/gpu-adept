import { SaliencyPipeline } from "./pipeline.js";
import { processTiles } from "./tile-processor.js";
import { autoTileSize } from "./utils.js";

const OUTPUT_QUALITY = 96;
const QUALITY_MIN = 64;
const QUALITY_STEP = 2;

function findOptimalThreshold(hist, totalPixels) {
  let lower = 0;
  let upper = 100;
  let threshold = 50;

  for (let iter = 0; iter < 20 && lower < upper - 1; iter++) {
    const tVal = threshold * 2.55;
    let countAbove = 0;
    for (let g = Math.ceil(tVal); g < 256; g++) {
      countAbove += hist[g];
    }
    const mean = (255 * countAbove) / totalPixels;

    if (mean < 20) {
      upper = threshold;
    } else if (mean > 40) {
      lower = threshold;
    } else {
      break;
    }
    threshold = Math.floor((upper - lower) / 2 + lower);
  }

  return threshold;
}

function snapQuality(raw) {
  if (raw >= OUTPUT_QUALITY) return OUTPUT_QUALITY;
  const clamped = Math.max(QUALITY_MIN, Math.min(OUTPUT_QUALITY - QUALITY_STEP, raw));
  return Math.round((clamped - QUALITY_MIN) / QUALITY_STEP) * QUALITY_STEP + QUALITY_MIN;
}

export class AdeptJPEG {
  constructor(device) {
    this._device = device;
    this._pipeline = new SaliencyPipeline(device);
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
    return new AdeptJPEG(device);
  }

  async compress(imageData, opts = {}) {
    const { width, height, data } = imageData;
    const tileSize = opts.tileSize || autoTileSize(width, height);
    const onProgress = opts.onProgress || null;

    if (onProgress) onProgress("computing saliency");
    this._pipeline.setup(width, height, tileSize, data);

    const hist = await this._pipeline.computeSaliency();

    if (onProgress) onProgress("finding optimal threshold");
    const totalPixels = width * height;
    const threshold = findOptimalThreshold(hist, totalPixels);

    if (onProgress) onProgress("analyzing tiles");
    const tileMeans = await this._pipeline.analyzeTiles(threshold * 2.55);

    if (onProgress) onProgress("encoding tiles");
    const tileQualities = new Uint8Array(tileMeans.length);
    let lowComplexityCount = 0;
    for (let i = 0; i < tileMeans.length; i++) {
      const frac = Math.min(1, tileMeans[i] / 255);
      const raw = QUALITY_MIN + (OUTPUT_QUALITY - QUALITY_MIN) * frac;
      tileQualities[i] = snapQuality(raw);
      if (tileQualities[i] < OUTPUT_QUALITY) lowComplexityCount++;
    }

    const outCanvas = await processTiles(
      imageData, tileQualities, tileSize,
      (msg) => { if (onProgress) onProgress(msg); }
    );

    if (onProgress) onProgress("generating output");
    const blob = await outCanvas.convertToBlob({
      type: "image/jpeg",
      quality: OUTPUT_QUALITY / 100,
    });

    return {
      blob,
      tileSize,
      tilesX: Math.ceil(width / tileSize),
      tilesY: Math.ceil(height / tileSize),
      tileQualities,
      lowComplexityCount,
      totalTiles: tileMeans.length,
      highQuality: OUTPUT_QUALITY,
      lowQuality: QUALITY_MIN,
      threshold,
    };
  }

  destroy() {
    this._pipeline?.destroy();
  }
}
