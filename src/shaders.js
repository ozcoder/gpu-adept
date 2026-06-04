// ── Saliency pipeline params ─────────────────────────────────────────

const SAL_STRUCT = `
struct SalParams {
  width: u32,
  height: u32,
  maxVal: f32,
  minVal: f32,
  numSegX: u32,
  numSegY: u32,
};
`;

const SAL_BIND = {
  p:    `@group(0) @binding(0) var<storage, read> p: SalParams;`,
  in:   `@group(0) @binding(1) var<storage, read> input: array<u32>;`,
  lab:  `@group(0) @binding(1) var<storage, read> lab: array<vec4<f32>>;`,
  labW: `@group(0) @binding(1) var<storage, read_write> lab: array<vec4<f32>>;`,
  tmp:  `@group(0) @binding(1) var<storage, read> temp: array<vec4<f32>>;`,
  tmpW: `@group(0) @binding(1) var<storage, read_write> temp: array<vec4<f32>>;`,
  blr:  `@group(0) @binding(1) var<storage, read> blurred: array<vec4<f32>>;`,
  blrW: `@group(0) @binding(1) var<storage, read_write> blurred: array<vec4<f32>>;`,
  intW: `@group(0) @binding(1) var<storage, read_write> integral: array<vec4<f32>>;`,
  int:  `@group(0) @binding(1) var<storage, read> integral: array<vec4<f32>>;`,
  sHr:  `@group(0) @binding(2) var<storage, read> segSumsH: array<vec4<f32>>;`,
  sHw:  `@group(0) @binding(2) var<storage, read_write> segSumsH: array<vec4<f32>>;`,
  sVr:  `@group(0) @binding(2) var<storage, read> segSumsV: array<vec4<f32>>;`,
  sVw:  `@group(0) @binding(2) var<storage, read_write> segSumsV: array<vec4<f32>>;`,
  sal:  `@group(0) @binding(1) var<storage, read_write> saliency: array<f32>;`,
  mx:   `@group(0) @binding(1) var<storage, read_write> maxVals: array<f32>;`,
  out:  `@group(0) @binding(1) var<storage, read_write> output: array<u32>;`,

  // binding 2
  labW2: `@group(0) @binding(2) var<storage, read_write> lab: array<vec4<f32>>;`,
  tmpW2: `@group(0) @binding(2) var<storage, read_write> temp: array<vec4<f32>>;`,
  blrW2: `@group(0) @binding(2) var<storage, read_write> blurred: array<vec4<f32>>;`,
  intW2: `@group(0) @binding(2) var<storage, read_write> integral: array<vec4<f32>>;`,
  int2:  `@group(0) @binding(2) var<storage, read> integral: array<vec4<f32>>;`,
  sal2:  `@group(0) @binding(2) var<storage, read_write> saliency: array<f32>;`,
  mx2:   `@group(0) @binding(2) var<storage, read_write> maxVals: array<vec2<f32>>;`,
  out2:  `@group(0) @binding(2) var<storage, read_write> output: array<u32>;`,

  // binding 3
  intW3: `@group(0) @binding(3) var<storage, read_write> integral: array<vec4<f32>>;`,
  sHw3:  `@group(0) @binding(3) var<storage, read_write> segSumsH: array<vec4<f32>>;`,
  sal3:  `@group(0) @binding(3) var<storage, read_write> saliency: array<f32>;`,
  hist3: `@group(0) @binding(3) var<storage, read_write> hist: array<atomic<u32>>;`,

  // segment scan
  sHsc: `@group(0) @binding(1) var<storage, read_write> segSumsH: array<vec4<f32>>;`,
  sVsc: `@group(0) @binding(1) var<storage, read_write> segSumsV: array<vec4<f32>>;`,

  // binarize
  outR:  `@group(0) @binding(1) var<storage, read> output: array<u32>;`,
  bwW:   `@group(0) @binding(2) var<storage, read_write> bwOut: array<u32>;`,
};

export const RGB2LAB = `
${SAL_STRUCT}
${SAL_BIND.p}
${SAL_BIND.in}
${SAL_BIND.labW2}

@compute @workgroup_size(16, 16)
fn rgb2lab(@builtin(global_invocation_id) id: vec3<u32>) {
  let x = id.x;
  let y = id.y;
  if (x >= p.width || y >= p.height) { return; }
  let idx = y * p.width + x;
  let rgba = input[idx];
  let r8 =  rgba        & 0xFFu;
  let g8 = (rgba >> 8u)  & 0xFFu;
  let b8 = (rgba >> 16u) & 0xFFu;
  let R = f32(r8) / 255.0;
  let G = f32(g8) / 255.0;
  let B = f32(b8) / 255.0;
  let r = select(pow((R + 0.055) / 1.055, 2.4), R / 12.92, R <= 0.04045);
  let g = select(pow((G + 0.055) / 1.055, 2.4), G / 12.92, G <= 0.04045);
  let b = select(pow((B + 0.055) / 1.055, 2.4), B / 12.92, B <= 0.04045);
  let X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
  let Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
  let Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;
  let epsilon = 0.008856;
  let kappa   = 903.3;
  let Xr = 0.950456;
  let Yr = 1.0;
  let Zr = 1.088754;
  let xr = X / Xr;
  let yr = Y / Yr;
  let zr = Z / Zr;
  let fx = select(pow(xr, 1.0 / 3.0), (kappa * xr + 16.0) / 116.0, xr <= epsilon);
  let fy = select(pow(yr, 1.0 / 3.0), (kappa * yr + 16.0) / 116.0, yr <= epsilon);
  let fz = select(pow(zr, 1.0 / 3.0), (kappa * zr + 16.0) / 116.0, zr <= epsilon);
  lab[idx] = vec4(116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz), 0.0);
}
`;

export const GAUSSIAN_X = `
${SAL_STRUCT}
${SAL_BIND.p}
${SAL_BIND.lab}
${SAL_BIND.tmpW2}

@compute @workgroup_size(16, 16)
fn gaussianX(@builtin(global_invocation_id) id: vec3<u32>) {
  let x = id.x;
  let y = id.y;
  if (x >= p.width || y >= p.height) { return; }
  let idx = y * p.width + x;
  var sum = lab[idx] * 2.0;
  var ksum = 2.0;
  if (x > 0u) { sum += lab[idx - 1u]; ksum += 1.0; }
  if (x + 1u < p.width) { sum += lab[idx + 1u]; ksum += 1.0; }
  temp[idx] = sum / ksum;
}
`;

export const GAUSSIAN_Y = `
${SAL_STRUCT}
${SAL_BIND.p}
${SAL_BIND.tmp}
${SAL_BIND.blrW2}

@compute @workgroup_size(16, 16)
fn gaussianY(@builtin(global_invocation_id) id: vec3<u32>) {
  let x = id.x;
  let y = id.y;
  if (x >= p.width || y >= p.height) { return; }
  let idx = y * p.width + x;
  var sum = temp[idx] * 2.0;
  var ksum = 2.0;
  if (y > 0u) { sum += temp[(y - 1u) * p.width + x]; ksum += 1.0; }
  if (y + 1u < p.height) { sum += temp[(y + 1u) * p.width + x]; ksum += 1.0; }
  blurred[idx] = sum / ksum;
}
`;

export const INTEGRAL_H = `
${SAL_STRUCT}
${SAL_BIND.p}
${SAL_BIND.lab}
${SAL_BIND.intW2}
${SAL_BIND.sHw3}

@compute @workgroup_size(256, 1)
fn integralH(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wg: vec3<u32>,
) {
  let row = wg.y;
  let seg = wg.x;
  let segStart = seg * 256u;
  let segEnd   = min(segStart + 256u, p.width);
  let segLen   = segEnd - segStart;
  let idx = lid.x;

  var shOrig: array<vec4<f32>, 256>;
  var shScan: array<vec4<f32>, 256>;

  shOrig[idx] = select(vec4(0.0), lab[row * p.width + segStart + idx], idx < segLen);
  shScan[idx] = shOrig[idx];
  workgroupBarrier();

  // Up-sweep
  for (var d = 0u; d < 8u; d = d + 1u) {
    let stride = 1u << (d + 1u);
    let half = stride >> 1u;
    if ((idx + 1u) % stride == 0u) {
      shScan[idx] = shScan[idx - half] + shScan[idx];
    }
    workgroupBarrier();
  }

  if (idx == 255u) {
    shScan[idx] = vec4(0.0);
  }
  workgroupBarrier();

  // Down-sweep
  for (var d = 8u; d > 0u; d = d - 1u) {
    let d2 = d - 1u;
    let stride = 1u << (d2 + 1u);
    let half = stride >> 1u;
    if ((idx + 1u) % stride == 0u) {
      let tmp = shScan[idx - half];
      shScan[idx - half] = shScan[idx];
      shScan[idx] = tmp + shScan[idx];
    }
    workgroupBarrier();
  }

  // Exclusive → inclusive by adding original
  if (idx < segLen) {
    integral[row * p.width + segStart + idx] = shScan[idx] + shOrig[idx];
  }
  if (idx == 0u) {
    let lastIdx = select(0u, segLen - 1u, segLen > 0u);
    segSumsH[row * p.numSegX + seg] = shOrig[lastIdx] + shScan[lastIdx];
  }
}
`;

export const INTEGRAL_H_FIXUP = `
${SAL_STRUCT}
${SAL_BIND.p}
${SAL_BIND.intW}
${SAL_BIND.sHr}

@compute @workgroup_size(256, 1)
fn integralHFixup(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
  let row = wg.y;
  let seg = wg.x;
  let segStart = seg * 256u;
  let segEnd   = min(segStart + 256u, p.width);
  let segLen   = segEnd - segStart;
  let offset = segSumsH[row * p.numSegX + seg];
  for (var i = lid.x; i < segLen; i += 256u) {
    integral[row * p.width + segStart + i] += offset;
  }
}
`;

export const SEG_SCAN_H = `
${SAL_STRUCT}
${SAL_BIND.p}
${SAL_BIND.sHsc}

@compute @workgroup_size(1, 1)
fn segScanH(@builtin(workgroup_id) wg: vec3<u32>) {
  let row = wg.y;
  let segCount = p.numSegX;
  let base = row * segCount;
  var running = vec4(0.0);
  for (var i = 0u; i < segCount; i++) {
    let tmp = segSumsH[base + i];
    segSumsH[base + i] = running;
    running += tmp;
  }
}
`;

export const INTEGRAL_V = `
${SAL_STRUCT}
${SAL_BIND.p}
${SAL_BIND.intW}
${SAL_BIND.sVw}

@compute @workgroup_size(1, 256)
fn integralV(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wg: vec3<u32>,
) {
  let col = wg.x;
  let seg = wg.y;
  let segStart = seg * 256u;
  let segEnd   = min(segStart + 256u, p.height);
  let segLen   = segEnd - segStart;
  let idx = lid.y;

  var shOrig: array<vec4<f32>, 256>;
  var shScan: array<vec4<f32>, 256>;

  shOrig[idx] = select(vec4(0.0), integral[(segStart + idx) * p.width + col], idx < segLen);
  shScan[idx] = shOrig[idx];
  workgroupBarrier();

  for (var d = 0u; d < 8u; d = d + 1u) {
    let stride = 1u << (d + 1u);
    let half = stride >> 1u;
    if ((idx + 1u) % stride == 0u) {
      shScan[idx] = shScan[idx - half] + shScan[idx];
    }
    workgroupBarrier();
  }

  if (idx == 255u) {
    shScan[idx] = vec4(0.0);
  }
  workgroupBarrier();

  for (var d = 8u; d > 0u; d = d - 1u) {
    let d2 = d - 1u;
    let stride = 1u << (d2 + 1u);
    let half = stride >> 1u;
    if ((idx + 1u) % stride == 0u) {
      let tmp = shScan[idx - half];
      shScan[idx - half] = shScan[idx];
      shScan[idx] = tmp + shScan[idx];
    }
    workgroupBarrier();
  }

  if (idx < segLen) {
    integral[(segStart + idx) * p.width + col] = shScan[idx] + shOrig[idx];
  }
  if (idx == 0u) {
    let lastIdx = select(0u, segLen - 1u, segLen > 0u);
    segSumsV[col * p.numSegY + seg] = shOrig[lastIdx] + shScan[lastIdx];
  }
}
`;

export const INTEGRAL_V_FIXUP = `
${SAL_STRUCT}
${SAL_BIND.p}
${SAL_BIND.intW}
${SAL_BIND.sVr}

@compute @workgroup_size(1, 256)
fn integralVFixup(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
  let col = wg.x;
  let seg = wg.y;
  let segStart = seg * 256u;
  let segEnd   = min(segStart + 256u, p.height);
  let segLen   = segEnd - segStart;
  let offset = segSumsV[col * p.numSegY + seg];
  for (var i = lid.y; i < segLen; i += 256u) {
    integral[(segStart + i) * p.width + col] += offset;
  }
}
`;

export const SEG_SCAN_V = `
${SAL_STRUCT}
${SAL_BIND.p}
${SAL_BIND.sVsc}

@compute @workgroup_size(1, 1)
fn segScanV(@builtin(workgroup_id) wg: vec3<u32>) {
  let col = wg.x;
  let segCount = p.numSegY;
  let base = col * segCount;
  var running = vec4(0.0);
  for (var i = 0u; i < segCount; i++) {
    let tmp = segSumsV[base + i];
    segSumsV[base + i] = running;
    running += tmp;
  }
}
`;

export const SALIENCY = `
${SAL_STRUCT}
${SAL_BIND.p}
${SAL_BIND.blr}
${SAL_BIND.int2}
${SAL_BIND.sal3}

fn getIntegralSum(x1: u32, y1: u32, x2: u32, y2: u32) -> vec4<f32> {
  let a = integral[y2 * p.width + x2];
  let b = select(vec4(0.0), integral[y2 * p.width + x1 - 1u], x1 > 0u);
  let c = select(vec4(0.0), integral[(y1 - 1u) * p.width + x2], y1 > 0u);
  let d = select(vec4(0.0), integral[(y1 - 1u) * p.width + x1 - 1u], x1 > 0u && y1 > 0u);
  return a - b - c + d;
}

@compute @workgroup_size(16, 16)
fn computeSaliency(@builtin(global_invocation_id) id: vec3<u32>) {
  let x = id.x;
  let y = id.y;
  if (x >= p.width || y >= p.height) { return; }
  let idx = y * p.width + x;
  let xoff = min(x, p.width  - x);
  let yoff = min(y, p.height - y);
  let x1 = x - xoff;
  let x2 = min(x + xoff, p.width  - 1u);
  let y1 = y - yoff;
  let y2 = min(y + yoff, p.height - 1u);
  let area = f32((x2 - x1 + 1u) * (y2 - y1 + 1u));
  let meanLab = getIntegralSum(x1, y1, x2, y2) / area;
  let pixLab  = blurred[idx];
  let diff    = meanLab - pixLab;
  saliency[idx] = dot(diff, diff);
}
`;

export const REDUCE_MAX = `
${SAL_STRUCT}
${SAL_BIND.p}
${SAL_BIND.sal}
${SAL_BIND.mx2}

var<workgroup> shMin: array<f32, 256>;
var<workgroup> shMax: array<f32, 256>;

@compute @workgroup_size(256)
fn reduceMax(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
  let total = p.width * p.height;
  let idx = id.x;
  if (idx < total) {
    let v = saliency[idx];
    shMin[lid.x] = v;
    shMax[lid.x] = v;
  } else {
    shMin[lid.x] = 1.0e30;
    shMax[lid.x] = -1.0;
  }
  workgroupBarrier();
  for (var d = 128u; d > 0u; d /= 2u) {
    if (lid.x < d) {
      shMin[lid.x] = min(shMin[lid.x], shMin[lid.x + d]);
      shMax[lid.x] = max(shMax[lid.x], shMax[lid.x + d]);
    }
    workgroupBarrier();
  }
  if (lid.x == 0u) {
    maxVals[wg.x] = vec2(shMin[0], shMax[0]);
  }
}
`;

export const NORMALIZE_HIST = `
${SAL_STRUCT}
${SAL_BIND.p}
${SAL_BIND.sal}
${SAL_BIND.out2}
${SAL_BIND.hist3}

@compute @workgroup_size(16, 16)
fn normalizeHist(@builtin(global_invocation_id) id: vec3<u32>) {
  let x = id.x;
  let y = id.y;
  if (x >= p.width || y >= p.height) { return; }
  let idx = y * p.width + x;
  let range = p.maxVal - p.minVal;
  let s = select(0.0, (saliency[idx] - p.minVal) / range, range > 0.0);
  let val = u32(clamp(s * 255.0, 0.0, 255.0));
  output[idx] = val | (val << 8u) | (val << 16u) | (255u << 24u);
  atomicAdd(&hist[val], 1u);
}
`;

// ── Tile analysis ────────────────────────────────────────────────────

const TILE_STRUCT = `
struct TileParams {
  imgW: u32,
  imgH: u32,
  tileSize: u32,
  tilesX: u32,
  tilesY: u32,
  threshold: f32,
};
`;

export const TILE_ANALYSIS = `
${TILE_STRUCT}

@group(0) @binding(0) var<uniform> tp: TileParams;
@group(0) @binding(1) var<storage, read> saliency: array<u32>;
@group(0) @binding(2) var<storage, read_write> tileMeans: array<f32>;

var<workgroup> wgSum: array<f32, 64>;
var<workgroup> wgCnt: array<u32, 64>;

@compute @workgroup_size(8, 8)
fn tileAnalysis(
  @builtin(workgroup_id) wgId: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  if (wgId.x >= tp.tilesX || wgId.y >= tp.tilesY) { return; }

  let blockW = (tp.tileSize + 7u) / 8u;
  let blockH = (tp.tileSize + 7u) / 8u;
  let baseX = wgId.x * tp.tileSize;
  let baseY = wgId.y * tp.tileSize;

  var sum = 0.0;
  var cnt = 0u;

  for (var dy = 0u; dy < blockH; dy++) {
    let py = baseY + lid.y * blockH + dy;
    if (py >= tp.imgH) { continue; }
    for (var gx = 0u; gx < blockW; gx += 4u) {
      let px0 = baseX + lid.x * blockW + gx;
      if (gx + 4u <= blockW && px0 + 4u <= tp.imgW) {
        // Vec4 load: 4 consecutive owned pixels, all in-bounds
        let v0 = saliency[py * tp.imgW + px0];
        let v1 = saliency[py * tp.imgW + px0 + 1u];
        let v2 = saliency[py * tp.imgW + px0 + 2u];
        let v3 = saliency[py * tp.imgW + px0 + 3u];
        let r = vec4<f32>(f32(v0 & 0xFFu), f32(v1 & 0xFFu), f32(v2 & 0xFFu), f32(v3 & 0xFFu));
        let bw = select(vec4<f32>(0.0), vec4<f32>(255.0), r > vec4<f32>(tp.threshold));
        sum += bw.x + bw.y + bw.z + bw.w;
        cnt += 4u;
      } else {
        // Per-pixel fallback for partial group or image edge
        for (var i = 0u; i < 4u; i++) {
          let px = px0 + i;
          if (px < tp.imgW && gx + i < blockW) {
            let gray = f32(saliency[py * tp.imgW + px] & 0xFFu);
            let bw = select(0.0, 255.0, gray > tp.threshold);
            sum += bw;
            cnt++;
          }
        }
      }
    }
  }

  let tid = lid.y * 8u + lid.x;
  wgSum[tid] = sum;
  wgCnt[tid] = cnt;
  workgroupBarrier();

  var s = 32u;
  while (s > 0u) {
    if (tid < s) {
      wgSum[tid] = wgSum[tid] + wgSum[tid + s];
      wgCnt[tid] = wgCnt[tid] + wgCnt[tid + s];
    }
    workgroupBarrier();
    s = s >> 1u;
  }

  if (tid == 0u) {
    tileMeans[wgId.y * tp.tilesX + wgId.x] = wgSum[0] / f32(wgCnt[0]);
  }
}
`;
