const STRUCT = `
struct Params {
  imgW: u32,
  imgH: u32,
  tileSize: u32,
  tilesX: u32,
  tilesY: u32,
  threshold: f32,
};
`;

const BIND = {
  params: `@group(0) @binding(0) var<uniform> params: Params;`,
  input:  `@group(0) @binding(1) var<storage, read> saliency: array<u32>;`,
  output: `@group(0) @binding(2) var<storage, read_write> tileMeans: array<f32>;`,
};

export const TILE_ANALYSIS = `
${STRUCT}
${BIND.params}
${BIND.input}
${BIND.output}

var<workgroup> wgSum: array<f32, 256>;
var<workgroup> wgCnt: array<u32, 256>;

@compute @workgroup_size(16, 16)
fn tileAnalysis(
  @builtin(workgroup_id) wgId: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tileX = wgId.x;
  let tileY = wgId.y;
  if (tileX >= params.tilesX || tileY >= params.tilesY) { return; }

  let blockW = (params.tileSize + 15u) / 16u;
  let blockH = (params.tileSize + 15u) / 16u;

  let baseX = tileX * params.tileSize;
  let baseY = tileY * params.tileSize;

  var sum = 0.0;
  var cnt = 0u;

  for (var dy = 0u; dy < blockH; dy++) {
    for (var dx = 0u; dx < blockW; dx++) {
      let px = baseX + lid.x * blockW + dx;
      let py = baseY + lid.y * blockH + dy;
      if (px < params.imgW && py < params.imgH) {
        let idx = py * params.imgW + px;
        let rgba = saliency[idx];
        let gray = f32(rgba & 0xFFu);
        let bw = select(0.0, 255.0, gray > params.threshold);
        sum += bw;
        cnt++;
      }
    }
  }

  let tid = lid.y * 16u + lid.x;
  wgSum[tid] = sum;
  wgCnt[tid] = cnt;
  workgroupBarrier();

  var s = 128u;
  while (s > 0u) {
    if (tid < s) {
      wgSum[tid] = wgSum[tid] + wgSum[tid + s];
      wgCnt[tid] = wgCnt[tid] + wgCnt[tid + s];
    }
    workgroupBarrier();
    s = s >> 1u;
  }

  if (tid == 0u) {
    let tileIdx = tileY * params.tilesX + tileX;
    tileMeans[tileIdx] = wgSum[0] / f32(wgCnt[0]);
  }
}
`;

export const ALL_SHADERS = TILE_ANALYSIS;
