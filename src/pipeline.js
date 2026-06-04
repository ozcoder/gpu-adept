import {
  RGB2LAB, GAUSSIAN_X, GAUSSIAN_Y,
  INTEGRAL_H, INTEGRAL_H_FIXUP, SEG_SCAN_H,
  INTEGRAL_V, INTEGRAL_V_FIXUP, SEG_SCAN_V,
  SALIENCY, REDUCE_MAX, NORMALIZE_HIST,
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
      reduceMax:     makePipeline(device, REDUCE_MAX,    'reduceMax'),
      normalizeHist: makePipeline(device, NORMALIZE_HIST,'normalizeHist'),
      tileAnalysis:  makePipeline(device, TILE_ANALYSIS, 'tileAnalysis'),
    };
    this._bgCache = {};
  }

  setup(width, height, tileSize, initialData) {
    this.destroy();
    const device = this.device;
    const pixelCount = width * height;
    const numSegX = Math.max(1, Math.ceil(width / 256));
    const numSegY = Math.max(1, Math.ceil(height / 256));
    const tilesX = Math.ceil(width / tileSize);
    const tilesY = Math.ceil(height / tileSize);

    this._width = width;
    this._height = height;
    this._pixelCount = pixelCount;
    this._tileSize = tileSize;
    this._tilesX = tilesX;
    this._tilesY = tilesY;
    this._numSegX = numSegX;
    this._numSegY = numSegY;
    this._maxWgCount = Math.ceil(pixelCount / 256);

    const S = GPUBufferUsage;

    /* ─── Saliency params storage buffer (24 bytes, matches SAL_STRUCT) ─── */
    this._salRaw = new ArrayBuffer(24);
    wu32(this._salRaw, 0, width);
    wu32(this._salRaw, 4, height);
    wf32(this._salRaw, 8, 0);   // maxVal (filled later)
    wf32(this._salRaw, 12, 0);  // minVal (filled later)
    wu32(this._salRaw, 16, numSegX);
    wu32(this._salRaw, 20, numSegY);
    this.salParamsBuf = device.createBuffer({
      size: 24, usage: S.STORAGE | S.COPY_DST,
    });

    /* ─── Input RGBA (u32) ─── */
    this.inputBuf = device.createBuffer({
      size: pixelCount * 4, usage: S.STORAGE | S.COPY_DST,
    });
    device.queue.writeBuffer(this.inputBuf, 0, initialData);

    /* ─── Working float buffers (vec4<f32> per pixel) ─── */
    const floatSz = pixelCount * 16;
    this.labBuf  = device.createBuffer({ size: floatSz, usage: S.STORAGE });
    this.tempBuf = device.createBuffer({ size: floatSz, usage: S.STORAGE });
    this.blurBuf = device.createBuffer({ size: floatSz, usage: S.STORAGE });
    this.intBuf  = device.createBuffer({ size: floatSz, usage: S.STORAGE });

    /* ─── Saliency map (f32) ─── */
    this.saliencyBuf = device.createBuffer({
      size: pixelCount * 4, usage: S.STORAGE,
    });

    /* ─── Max reduction output (vec2<f32> per workgroup) ─── */
    this.maxValsBuf = device.createBuffer({
      size: this._maxWgCount * 8, usage: S.STORAGE | S.COPY_SRC,
    });

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
    wu32(this._tileRaw, 8, tileSize);
    wu32(this._tileRaw, 12, tilesX);
    wu32(this._tileRaw, 16, tilesY);
    wf32(this._tileRaw, 20, 0);  // threshold (set later)
    this.tileParamsBuf = device.createBuffer({
      size: 32, usage: S.UNIFORM | S.COPY_DST,
    });

    /* ─── Staging buffers ─── */
    this.maxValsStaging = device.createBuffer({
      size: this._maxWgCount * 8, usage: S.MAP_READ | S.COPY_DST,
    });
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
    const mW = this._maxWgCount;
    const gX = Math.ceil(W / 16);
    const gY = Math.ceil(H / 16);

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

    // Gaussian X
    pass.setPipeline(P.gaussianX);
    pass.setBindGroup(0, bg('gaussianX', [
      { binding: 0, resource: { buffer: this.salParamsBuf } },
      { binding: 1, resource: { buffer: this.labBuf } },
      { binding: 2, resource: { buffer: this.tempBuf } },
    ]));
    pass.dispatchWorkgroups(gX, gY);

    // Gaussian Y
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

    // Saliency
    pass.setPipeline(P.saliency);
    pass.setBindGroup(0, bg('saliency', [
      { binding: 0, resource: { buffer: this.salParamsBuf } },
      { binding: 1, resource: { buffer: this.blurBuf } },
      { binding: 2, resource: { buffer: this.intBuf } },
      { binding: 3, resource: { buffer: this.saliencyBuf } },
    ]));
    pass.dispatchWorkgroups(gX, gY);

    // Reduce max/min
    pass.setPipeline(P.reduceMax);
    pass.setBindGroup(0, bg('reduceMax', [
      { binding: 0, resource: { buffer: this.salParamsBuf } },
      { binding: 1, resource: { buffer: this.saliencyBuf } },
      { binding: 2, resource: { buffer: this.maxValsBuf } },
    ]));
    pass.dispatchWorkgroups(mW);

    pass.end();
    enc.copyBufferToBuffer(this.maxValsBuf, 0, this.maxValsStaging, 0, mW * 8);
    device.queue.submit([enc.finish()]);

    // Read back min/max
    await this.maxValsStaging.mapAsync(GPUMapMode.READ);
    const maxRaw = new Float32Array(this.maxValsStaging.getMappedRange());
    let gMin = Infinity, gMax = -Infinity;
    for (let i = 0; i < mW; i++) {
      const mn = maxRaw[i * 2], mx = maxRaw[i * 2 + 1];
      if (mn < gMin) gMin = mn;
      if (mx > gMax) gMax = mx;
    }
    this.maxValsStaging.unmap();

    // Update saliency params with computed min/max
    wf32(this._salRaw, 8, gMax);
    wf32(this._salRaw, 12, gMin);
    this._writeSalParams();

    // Second pass: normalize + histogram (single dispatch)
    const enc2 = device.createCommandEncoder();
    const pass2 = enc2.beginComputePass();

    pass2.setPipeline(P.normalizeHist);
    pass2.setBindGroup(0, bg('normalizeHist', [
      { binding: 0, resource: { buffer: this.salParamsBuf } },
      { binding: 1, resource: { buffer: this.saliencyBuf } },
      { binding: 2, resource: { buffer: this.outputBuf } },
      { binding: 3, resource: { buffer: this.histBuf } },
    ]));
    pass2.dispatchWorkgroups(gX, gY);

    pass2.end();
    enc2.copyBufferToBuffer(this.histBuf, 0, this.histStaging, 0, 1024);
    device.queue.submit([enc2.finish()]);

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
      this.saliencyBuf, this.maxValsBuf,
      this.segSumsHBuf, this.segSumsVBuf,
      this.outputBuf, this.histBuf, this.tileMeansBuf,
      this.tileParamsBuf,
      this.maxValsStaging, this.histStaging, this.tileMeansStaging,
      this.outputStaging,
    ];
    for (const b of bufs) {
      if (b) b.destroy();
    }
    this._bgCache = {};
  }
}
