import {
  RGB2LAB, GAUSSIAN_X, GAUSSIAN_Y,
  INTEGRAL_H, INTEGRAL_H_FIXUP, SEG_SCAN_H,
  INTEGRAL_V, INTEGRAL_V_FIXUP, SEG_SCAN_V,
  SALIENCY, NORMALIZE_HIST,
  TILE_ANALYSIS,
} from './shaders.js';

const makePipeline = (device, code, entry) => {
  const module = device.createShaderModule({ code });
  return device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: entry },
  });
};

/* Write a scalar to an ArrayBuffer as u32 (little-endian). */
function wu32(buf, offset, val) {
  new DataView(buf).setUint32(offset, val, true);
}

/* Write a scalar to an ArrayBuffer as f32 (little-endian). */
function wf32(buf, offset, val) {
  new DataView(buf).setFloat32(offset, val, true);
}

export class SaliencyPipeline {
  constructor(device) {
    this.device = device;
    this.pipelines = {
      rgb2lab:       makePipeline(device, RGB2LAB,       'rgb2lab'),
      gaussianX:     makePipeline(device, GAUSSIAN_X,    'gaussianX'),
      gaussianY:     makePipeline(device, GAUSSIAN_Y,    'gaussianY'),
      integralH:     makePipeline(device, INTEGRAL_H,    'integralH'),
      integralHFixup:makePipeline(device, INTEGRAL_H_FIXUP, 'integralHFixup'),
      segScanH:      makePipeline(device, SEG_SCAN_H,    'segScanH'),
      integralV:     makePipeline(device, INTEGRAL_V,    'integralV'),
      integralVFixup:makePipeline(device, INTEGRAL_V_FIXUP, 'integralVFixup'),
      segScanV:      makePipeline(device, SEG_SCAN_V,    'segScanV'),
      saliency:      makePipeline(device, SALIENCY,      'computeSaliency'),
      normalizeHist: makePipeline(device, NORMALIZE_HIST,'normalizeHist'),
      tileAnalysis:  makePipeline(device, TILE_ANALYSIS, 'tileAnalysis'),
    };
    this._bgCache = {};
  }

  setup(width, height, tileSize, initialData) {
    this.destroy();
    if (!(width > 0 && height > 0)) {
      throw new Error(`Invalid image dimensions: ${width}×${height}`);
    }
    const device = this.device;
    const pixelCount = width * height;
    const numSegX = Math.max(1, Math.ceil(width / 256));
    const numSegY = Math.max(1, Math.ceil(height / 256));
    const analysisTileSize = 8;
    const tilesX = Math.ceil(width / analysisTileSize);
    const tilesY = Math.ceil(height / analysisTileSize);

    this._width = width;
    this._height = height;
    this._pixelCount = pixelCount;
    this._tileSize = analysisTileSize;
    this._tilesX = tilesX;
    this._tilesY = tilesY;
    this._numSegX = numSegX;
    this._numSegY = numSegY;
    const maxBuf = device.limits.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
    const maxAlign = Math.max(256, device.limits.maxBufferSize ?? maxBuf);
    const floatSz = pixelCount * 16;
    if (floatSz > maxBuf) {
      throw new Error(
        `Image too large for this device: vec4<f32> buffer needs ` +
        `${(floatSz / 1024 / 1024).toFixed(1)} MB (${width}×${height}), ` +
        `maxStorageBufferBindingSize is ${(maxBuf / 1024 / 1024).toFixed(1)} MB`
      );
    }
    if (floatSz > maxAlign) {
      throw new Error(
        `Image too large for this device: ${(floatSz / 1024 / 1024).toFixed(1)} MB exceeds maxBufferSize`
      );
    }

    const S = GPUBufferUsage;

    /* ─── Saliency params storage buffer (16 bytes, matches SAL_STRUCT) ─── */
    this._salRaw = new ArrayBuffer(16);
    wu32(this._salRaw, 0, width);
    wu32(this._salRaw, 4, height);
    wu32(this._salRaw, 8, numSegX);
    wu32(this._salRaw, 12, numSegY);
    this.salParamsBuf = device.createBuffer({
      size: 16, usage: S.STORAGE | S.COPY_DST,
    });

    /* ─── Input RGBA (u32) ─── */
    this.inputBuf = device.createBuffer({
      size: pixelCount * 4, usage: S.STORAGE | S.COPY_DST,
    });
    device.queue.writeBuffer(this.inputBuf, 0, initialData);

    /* ─── Working float buffers (vec4<f32> per pixel) ─── */
    this.labBuf  = device.createBuffer({ size: floatSz, usage: S.STORAGE });
    this.tempBuf = device.createBuffer({ size: floatSz, usage: S.STORAGE });
    this.blurBuf = device.createBuffer({ size: floatSz, usage: S.STORAGE });
    this.intBuf  = device.createBuffer({ size: floatSz, usage: S.STORAGE });

    /* ─── Saliency map (f32) ─── */
    this.saliencyBuf = device.createBuffer({
      size: pixelCount * 4, usage: S.STORAGE,
    });

    /* ─── Global min/max (atomic<u32>[2]) ─── */
    this.globalMinMaxBuf = device.createBuffer({
      size: 8, usage: S.STORAGE | S.COPY_DST,
    });
    const initMM = new Uint32Array([0x7F7FFFFF, 0]);
    device.queue.writeBuffer(this.globalMinMaxBuf, 0, initMM);

    /* ─── Segment sum buffers (vec4<f32> per segment) ─── */
    this.segSumsHBuf = device.createBuffer({
      size: width * numSegX * 16, usage: S.STORAGE,
    });
    this.segSumsVBuf = device.createBuffer({
      size: height * numSegY * 16, usage: S.STORAGE,
    });

    /* ─── Normalized output (u32 RGBA) ─── */
    this.outputBuf = device.createBuffer({
      size: pixelCount * 4, usage: S.STORAGE | S.COPY_SRC,
    });

    /* ─── Histogram (256 × atomic<u32>) ─── */
    this.histBuf = device.createBuffer({
      size: 1024, usage: S.STORAGE | S.COPY_SRC,
    });

    /* ─── Tile means (f32 per tile) ─── */
    this.tileMeansBuf = device.createBuffer({
      size: tilesX * tilesY * 4, usage: S.STORAGE | S.COPY_SRC,
    });

    /* ─── Tile params uniform buffer (32 bytes, matches TILE_STRUCT) ─── */
    this._tileRaw = new ArrayBuffer(32);
    wu32(this._tileRaw, 0, width);
    wu32(this._tileRaw, 4, height);
    wu32(this._tileRaw, 8, analysisTileSize);
    wu32(this._tileRaw, 12, tilesX);
    wu32(this._tileRaw, 16, tilesY);
    wf32(this._tileRaw, 20, 0);  // threshold (set later)
    this.tileParamsBuf = device.createBuffer({
      size: 32, usage: S.UNIFORM | S.COPY_DST,
    });

    /* ─── Staging buffers ─── */
    this.histStaging = device.createBuffer({
      size: 1024, usage: S.MAP_READ | S.COPY_DST,
    });
    this.tileMeansStaging = device.createBuffer({
      size: tilesX * tilesY * 4, usage: S.MAP_READ | S.COPY_DST,
    });
    this.outputStaging = device.createBuffer({
      size: pixelCount * 4, usage: S.MAP_READ | S.COPY_DST,
    });
  }

  /* ─────────── Host buffer writes ─────────── */

  _writeSalParams() {
    this.device.queue.writeBuffer(this.salParamsBuf, 0, this._salRaw);
  }

  _writeTileParams(threshold) {
    wf32(this._tileRaw, 20, threshold);
    this.device.queue.writeBuffer(this.tileParamsBuf, 0, this._tileRaw);
  }

  /* ─────────── Bind group helpers ─────────── */

  _bg(name, entries) {
    if (this._bgCache[name]) return this._bgCache[name];
    const layout = this.pipelines[name].getBindGroupLayout(0);
    this._bgCache[name] = this.device.createBindGroup({ layout, entries });
    return this._bgCache[name];
  }

  /* ─────────── Phase 1: saliency + histogram ─── */

  async computeSaliency() {
    const device = this.device;
    const W = this._width, H = this._height;
    const nX = this._numSegX;
    const nY = this._numSegY;
    const gX = Math.ceil(W / 16);
    const gY = Math.ceil(H / 16);

    // Reset global min/max atomics
    device.queue.writeBuffer(this.globalMinMaxBuf, 0, new Uint32Array([0x7F7FFFFF, 0]));

    this._writeSalParams();

    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    const P = this.pipelines;
    const bg = (n, e) => this._bg(n, e);

    // RGB → Lab
    pass.setPipeline(P.rgb2lab);
    pass.setBindGroup(0, bg('rgb2lab', [
      { binding: 0, resource: { buffer: this.salParamsBuf } },
      { binding: 1, resource: { buffer: this.inputBuf } },
      { binding: 2, resource: { buffer: this.labBuf } },
    ]));
    pass.dispatchWorkgroups(gX, gY);

    // Gaussian blur X → tempBuf
    pass.setPipeline(P.gaussianX);
    pass.setBindGroup(0, bg('gaussianX', [
      { binding: 0, resource: { buffer: this.salParamsBuf } },
      { binding: 1, resource: { buffer: this.labBuf } },
      { binding: 2, resource: { buffer: this.tempBuf } },
    ]));
    pass.dispatchWorkgroups(gX, gY);

    // Gaussian blur Y → blurBuf
    pass.setPipeline(P.gaussianY);
    pass.setBindGroup(0, bg('gaussianY', [
      { binding: 0, resource: { buffer: this.salParamsBuf } },
      { binding: 1, resource: { buffer: this.tempBuf } },
      { binding: 2, resource: { buffer: this.blurBuf } },
    ]));
    pass.dispatchWorkgroups(gX, gY);

    // Integral H
    pass.setPipeline(P.integralH);
    pass.setBindGroup(0, bg('integralH', [
      { binding: 0, resource: { buffer: this.salParamsBuf } },
      { binding: 1, resource: { buffer: this.blurBuf } },
      { binding: 2, resource: { buffer: this.intBuf } },
      { binding: 3, resource: { buffer: this.segSumsHBuf } },
    ]));
    pass.dispatchWorkgroups(nX, H);

    // SegScan H
    pass.setPipeline(P.segScanH);
    pass.setBindGroup(0, bg('segScanH', [
      { binding: 0, resource: { buffer: this.salParamsBuf } },
      { binding: 1, resource: { buffer: this.segSumsHBuf } },
    ]));
    pass.dispatchWorkgroups(1, H);

    // Integral H fixup
    pass.setPipeline(P.integralHFixup);
    pass.setBindGroup(0, bg('integralHFixup', [
      { binding: 0, resource: { buffer: this.salParamsBuf } },
      { binding: 1, resource: { buffer: this.intBuf } },
      { binding: 2, resource: { buffer: this.segSumsHBuf } },
    ]));
    pass.dispatchWorkgroups(nX, H);

    // Integral V
    pass.setPipeline(P.integralV);
    pass.setBindGroup(0, bg('integralV', [
      { binding: 0, resource: { buffer: this.salParamsBuf } },
      { binding: 1, resource: { buffer: this.intBuf } },
      { binding: 2, resource: { buffer: this.segSumsVBuf } },
    ]));
    pass.dispatchWorkgroups(W, nY);

    // SegScan V
    pass.setPipeline(P.segScanV);
    pass.setBindGroup(0, bg('segScanV', [
      { binding: 0, resource: { buffer: this.salParamsBuf } },
      { binding: 1, resource: { buffer: this.segSumsVBuf } },
    ]));
    pass.dispatchWorkgroups(W, 1);

    // Integral V fixup
    pass.setPipeline(P.integralVFixup);
    pass.setBindGroup(0, bg('integralVFixup', [
      { binding: 0, resource: { buffer: this.salParamsBuf } },
      { binding: 1, resource: { buffer: this.intBuf } },
      { binding: 2, resource: { buffer: this.segSumsVBuf } },
    ]));
    pass.dispatchWorkgroups(W, nY);

    // Saliency (writes pixel values + atomic min/max)
    pass.setPipeline(P.saliency);
    pass.setBindGroup(0, bg('saliency', [
      { binding: 0, resource: { buffer: this.salParamsBuf } },
      { binding: 1, resource: { buffer: this.blurBuf } },
      { binding: 2, resource: { buffer: this.intBuf } },
      { binding: 3, resource: { buffer: this.saliencyBuf } },
      { binding: 4, resource: { buffer: this.globalMinMaxBuf } },
    ]));
    pass.dispatchWorkgroups(gX, gY);

    // Normalize + histogram (reads atomic min/max)
    pass.setPipeline(P.normalizeHist);
    pass.setBindGroup(0, bg('normalizeHist', [
      { binding: 0, resource: { buffer: this.salParamsBuf } },
      { binding: 1, resource: { buffer: this.saliencyBuf } },
      { binding: 2, resource: { buffer: this.outputBuf } },
      { binding: 3, resource: { buffer: this.histBuf } },
      { binding: 4, resource: { buffer: this.globalMinMaxBuf } },
    ]));
    pass.dispatchWorkgroups(gX, gY);

    pass.end();
    enc.copyBufferToBuffer(this.histBuf, 0, this.histStaging, 0, 1024);
    device.queue.submit([enc.finish()]);

    await this.histStaging.mapAsync(GPUMapMode.READ);
    const hist = new Uint32Array(this.histStaging.getMappedRange()).slice();
    this.histStaging.unmap();

    return hist;
  }

  /* ─────────── Phase 2: tile analysis ─────── */

  async analyzeTiles(threshold) {
    this._writeTileParams(threshold);
    const device = this.device;
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();

    pass.setPipeline(this.pipelines.tileAnalysis);
    pass.setBindGroup(0, this._bg('tileAnalysis', [
      { binding: 0, resource: { buffer: this.tileParamsBuf } },
      { binding: 1, resource: { buffer: this.outputBuf } },
      { binding: 2, resource: { buffer: this.tileMeansBuf } },
    ]));
    pass.dispatchWorkgroups(this._tilesX, this._tilesY);
    pass.end();

    enc.copyBufferToBuffer(this.tileMeansBuf, 0, this.tileMeansStaging, 0,
      this._tilesX * this._tilesY * 4);
    device.queue.submit([enc.finish()]);

    await this.tileMeansStaging.mapAsync(GPUMapMode.READ);
    const means = new Float32Array(this.tileMeansStaging.getMappedRange()).slice();
    this.tileMeansStaging.unmap();
    return means;
  }

  /* ─────────── Debug readback ─────────────── */

  async getNormalizedOutput() {
    const device = this.device;
    const sz = this._pixelCount * 4;
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(this.outputBuf, 0, this.outputStaging, 0, sz);
    device.queue.submit([enc.finish()]);
    await this.outputStaging.mapAsync(GPUMapMode.READ);
    const data = new Uint8Array(this.outputStaging.getMappedRange()).slice();
    this.outputStaging.unmap();
    return data;
  }

  /* ─────────── Cleanup ────────────────────── */

  destroy() {
    const bufs = [
      this.salParamsBuf, this.inputBuf,
      this.labBuf, this.tempBuf, this.blurBuf, this.intBuf,
      this.saliencyBuf, this.globalMinMaxBuf,
      this.segSumsHBuf, this.segSumsVBuf,
      this.outputBuf, this.histBuf, this.tileMeansBuf,
      this.tileParamsBuf,
      this.histStaging, this.tileMeansStaging,
      this.outputStaging,
    ];
    for (const b of bufs) {
      if (b) b.destroy();
    }
    this._bgCache = {};
  }
}
