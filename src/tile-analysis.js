import { TILE_ANALYSIS } from "./shaders.js";
import { tileCount } from "./utils.js";

function ceilDiv(a, b) {
  return Math.ceil(a / b);
}

export class TileAnalysisPipeline {
  constructor(device, width, height, tileSize) {
    this.device = device;
    this.w = width;
    this.h = height;
    this.tileSize = tileSize;
    this.tilesX = tileCount(width, tileSize);
    this.tilesY = tileCount(height, tileSize);
    this.numTiles = this.tilesX * this.tilesY;

    this.buf = this._createBuffers();
    this.pipeline = this._createPipeline();
    this.bindGroup = this._createBindGroup();
    this._writeParams(0);
  }

  _createBuffers() {
    const d = this.device;
    const S = GPUBufferUsage;
    const n = this.w * this.h;
    const numTiles = this.numTiles;

    return {
      saliency: d.createBuffer({ size: n * 4, usage: S.STORAGE | S.COPY_DST }),
      tileMeans: d.createBuffer({ size: numTiles * 4, usage: S.STORAGE | S.COPY_SRC }),
      params: d.createBuffer({ size: 32, usage: S.UNIFORM | S.COPY_DST }),
      staging: d.createBuffer({ size: numTiles * 4, usage: S.MAP_READ | S.COPY_DST }),
    };
  }

  _createPipeline() {
    const mod = this.device.createShaderModule({ code: TILE_ANALYSIS });
    return this.device.createComputePipeline({
      layout: "auto",
      compute: { module: mod, entryPoint: "tileAnalysis" },
    });
  }

  _createBindGroup() {
    const layout = this.pipeline.getBindGroupLayout(0);
    return this.device.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: { buffer: this.buf.params } },
        { binding: 1, resource: { buffer: this.buf.saliency } },
        { binding: 2, resource: { buffer: this.buf.tileMeans } },
      ],
    });
  }

  _writeParams(threshold) {
    const buf = new ArrayBuffer(32);
    const dv = new DataView(buf);
    dv.setUint32(0, this.w, true);
    dv.setUint32(4, this.h, true);
    dv.setUint32(8, this.tileSize, true);
    dv.setUint32(12, this.tilesX, true);
    dv.setUint32(16, this.tilesY, true);
    dv.setFloat32(20, threshold, true);
    this.device.queue.writeBuffer(this.buf.params, 0, buf);
  }

  async analyze(saliencyPixels, threshold) {
    const d = this.device;
    const n = this.w * this.h;
    const numTiles = this.numTiles;

    this._writeParams(threshold);

    d.queue.writeBuffer(this.buf.saliency, 0, new Uint32Array(saliencyPixels.buffer));

    const encoder = d.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(this.tilesX, this.tilesY);
    pass.end();

    encoder.copyBufferToBuffer(this.buf.tileMeans, 0, this.buf.staging, 0, numTiles * 4);
    d.queue.submit([encoder.finish()]);

    await this.buf.staging.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(this.buf.staging.getMappedRange());
    const means = new Float32Array(result.length);
    means.set(result);
    this.buf.staging.unmap();

    return means;
  }

  destroy() {
    for (const key of Object.keys(this.buf)) {
      this.buf[key]?.destroy();
    }
  }
}
