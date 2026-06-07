import { SaliencyPipeline } from "./pipeline.js";
import { processTiles } from "./tile-processor.js";
import { buildAdaptiveTiles, buildUniformTiles } from "./quadtree.js";

const OUTPUT_QUALITY = 96;
const QUALITY_MIN = 64;
const QUALITY_STEP = 2;

export function findOptimalThreshold(hist, totalPixels) {
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

export function snapQuality(raw) {
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
    device.lost?.then((info) => {
      console.warn(`[AdeptJPEG] WebGPU device lost: ${info.reason} "${info.message}"`);
    });
    return new AdeptJPEG(device);
  }

  async compress(imageData, opts = {}) {
    const { width, height, data } = imageData;
    const onProgress = opts.onProgress || null;
    const tileSize = opts.tileSize ?? "auto";
    const _t0 = performance.now();

    if (onProgress) onProgress("computing saliency");
    this._pipeline.setup(width, height, 8, data);
    const _t1 = performance.now();

    const hist = await this._pipeline.computeSaliency();
    const _t2 = performance.now();

    if (onProgress) onProgress("finding optimal threshold");
    const totalPixels = width * height;
    const threshold = findOptimalThreshold(hist, totalPixels);
    const _t3 = performance.now();

    if (onProgress) onProgress("analyzing tiles");
    const tileMeans = await this._pipeline.analyzeTiles(threshold * 2.55);
    const _t4 = performance.now();

    const tilesX8 = Math.ceil(width / 8);
    const tilesY8 = Math.ceil(height / 8);
    let leaves;
    if (tileSize === "auto") {
      if (onProgress) onProgress("building adaptive tiles");
      leaves = buildAdaptiveTiles(tileMeans, tilesX8, tilesY8, width, height);
    } else {
      if (onProgress) onProgress(`building uniform ${tileSize}×${tileSize} tiles`);
      leaves = buildUniformTiles(tileMeans, tilesX8, tilesY8, width, height, tileSize);
    }

    let lowComplexityCount = 0;
    const sizeCounts = {};
    for (const leaf of leaves) {
      const frac = Math.min(1, leaf.salience / 255);
      const raw = QUALITY_MIN + (OUTPUT_QUALITY - QUALITY_MIN) * frac;
      leaf.quality = snapQuality(raw);
      if (leaf.quality < OUTPUT_QUALITY) lowComplexityCount++;
      const k = Math.max(leaf.w, leaf.h);
      sizeCounts[k] = (sizeCounts[k] || 0) + 1;
    }

    if (onProgress) onProgress("encoding tiles");
    const outCanvas = await processTiles(
      imageData, leaves,
      (msg) => { if (onProgress) onProgress(msg); }
    );
    const _t5 = performance.now();

    if (onProgress) onProgress("generating output");
    const blob = await outCanvas.convertToBlob({
      type: "image/jpeg",
      quality: OUTPUT_QUALITY / 100,
    });
    const _t6 = performance.now();

    console.log(
      `[timing] setupGPU=${(_t1-_t0).toFixed(0)} ` +
      `saliency=${(_t2-_t1).toFixed(0)} ` +
      `threshold=${(_t3-_t2).toFixed(0)} ` +
      `tiles=${(_t4-_t3).toFixed(0)} ` +
      `cpuTiling=${(_t5-_t4).toFixed(0)} ` +
      `finalBlob=${(_t6-_t5).toFixed(0)} ` +
      `total=${(_t6-_t0).toFixed(0)} ms`
    );

    return {
      blob,
      leaves,
      lowComplexityCount,
      totalLeaves: leaves.length,
      sizeCounts,
      highQuality: OUTPUT_QUALITY,
      lowQuality: QUALITY_MIN,
      threshold,
    };
  }

  destroy() {
    this._pipeline?.destroy();
  }
}
