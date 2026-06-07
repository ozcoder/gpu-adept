import { buildAdaptiveTiles, buildUniformTiles } from "../src/quadtree.js";
import { snapQuality, findOptimalThreshold } from "../src/adept.js";

let passed = 0;
let failed = 0;
const failures = [];

function check(name, cond, detail = "") {
  if (cond) {
    passed++;
  } else {
    failed++;
    failures.push({ name, detail });
    console.log(`FAIL: ${name}${detail ? " — " + detail : ""}`);
  }
}

function checkEqual(name, actual, expected) {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  check(name, a === e, `got ${a}, want ${e}`);
}

function assertCovers(name, leaves, imgW, imgH) {
  const covered = new Uint8Array(imgW * imgH);
  for (const leaf of leaves) {
    for (let yy = 0; yy < leaf.h; yy++) {
      for (let xx = 0; xx < leaf.w; xx++) {
        const px = (leaf.y + yy) * imgW + (leaf.x + xx);
        if (covered[px]) {
          check(`${name}: no overlap at (${leaf.x + xx},${leaf.y + yy})`, false);
          return;
        }
        covered[px] = 1;
      }
    }
  }
  const sum = covered.reduce((s, v) => s + v, 0);
  check(`${name}: covers full image`, sum === imgW * imgH, `covered ${sum}/${imgW * imgH}`);
}

function sumSalience(leaves) {
  return leaves.reduce((s, l) => s + l.salience, 0);
}

// snapQuality
checkEqual("snapQ 64", snapQuality(64), 64);
checkEqual("snapQ 65", snapQuality(65), 66);
checkEqual("snapQ 64.9", snapQuality(64.9), 64);
checkEqual("snapQ 65.1", snapQuality(65.1), 66);
checkEqual("snapQ 80", snapQuality(80), 80);
checkEqual("snapQ 94", snapQuality(94), 94);
checkEqual("snapQ 95", snapQuality(95), 94);
checkEqual("snapQ 96", snapQuality(96), 96);
checkEqual("snapQ 100", snapQuality(100), 96);
checkEqual("snapQ 0", snapQuality(0), 64);
checkEqual("snapQ 50", snapQuality(50), 64);
checkEqual("snapQ 63", snapQuality(63), 64);

// findOptimalThreshold — empty histogram (all low)
{
  const hist = new Array(256).fill(0);
  hist[10] = 1000;
  const t = findOptimalThreshold(hist, 1000);
  check("threshold all-low: t <= 10", t <= 10, `got ${t}`);
}

// findOptimalThreshold — all-high: algorithm targets countAbove in 7.8%-15.7% range,
// but a unimodal histogram has 0% or 100% above any threshold — algorithm oscillates
// around the bin location. Verify it returns a finite value in [0,100].
{
  const hist = new Array(256).fill(0);
  hist[200] = 1000;
  const t = findOptimalThreshold(hist, 1000);
  check("threshold all-high: t in [0,100]", t >= 0 && t <= 100, `got ${t}`);
}

// findOptimalThreshold — split in the middle
{
  const hist = new Array(256).fill(0);
  for (let i = 100; i < 200; i++) hist[i] = 10;
  const total = 1000;
  const t = findOptimalThreshold(hist, total);
  const tVal = t * 2.55;
  let countAbove = 0;
  for (let g = Math.ceil(tVal); g < 256; g++) countAbove += hist[g];
  const mean = (255 * countAbove) / total;
  check("threshold split-mid: mean 20..40", mean >= 20 && mean <= 40, `got ${mean}`);
}

// buildAdaptiveTiles: 64x64 all-zero -> 4 leaves of 32x32
{
  const tilesX8 = 8, tilesY8 = 8;
  const means = new Uint8Array(tilesX8 * tilesY8);
  const leaves = buildAdaptiveTiles(means, tilesX8, tilesY8, 64, 64);
  checkEqual("64x64 all-zero: 4 leaves", leaves.length, 4);
  checkEqual("64x64 all-zero: all 32x32", leaves.every(l => l.w === 32 && l.h === 32), true);
  assertCovers("64x64 all-zero", leaves, 64, 64);
}

// buildAdaptiveTiles: 64x64 all-ones -> 64 leaves of 8x8
{
  const tilesX8 = 8, tilesY8 = 8;
  const means = new Uint8Array(tilesX8 * tilesY8).fill(255);
  const leaves = buildAdaptiveTiles(means, tilesX8, tilesY8, 64, 64);
  checkEqual("64x64 all-ones: 64 leaves", leaves.length, 64);
  checkEqual("64x64 all-ones: all 8x8", leaves.every(l => l.w === 8 && l.h === 8), true);
  assertCovers("64x64 all-ones", leaves, 64, 64);
}

// buildAdaptiveTiles: 64x64 top-left 32x32 zero, rest ones
{
  const tilesX8 = 8, tilesY8 = 8;
  const means = new Uint8Array(tilesX8 * tilesY8).fill(255);
  for (let ty = 0; ty < 4; ty++)
    for (let tx = 0; tx < 4; tx++)
      means[ty * tilesX8 + tx] = 0;
  const leaves = buildAdaptiveTiles(means, tilesX8, tilesY8, 64, 64);
  const c32 = leaves.filter(l => l.w === 32 && l.h === 32);
  const c8  = leaves.filter(l => l.w === 8 && l.h === 8);
  checkEqual("64x64 quadrant-zero: 1x32", c32.length, 1);
  checkEqual("64x64 quadrant-zero: 48x8", c8.length, 48);
  checkEqual("64x64 quadrant-zero: total 49", leaves.length, 49);
  checkEqual("64x64 quadrant-zero: 32-leaf at (0,0)", c32[0].x === 0 && c32[0].y === 0, true);
  assertCovers("64x64 quadrant-zero", leaves, 64, 64);
}

// buildAdaptiveTiles: 64x64 single 8x8 salient inside otherwise-zero image
// 32x32 at (0,0) has one salient 8x8, the other three 32x32 are clean.
// Affected 32x32 subdivides to 4 16x16; the 16x16 containing the salient 8x8
// subdivides to 4 8x8 (3 non-salient + 1 salient). Other 3 16x16 stay whole.
{
  const tilesX8 = 8, tilesY8 = 8;
  const means = new Uint8Array(tilesX8 * tilesY8);
  means[0] = 200;
  const leaves = buildAdaptiveTiles(means, tilesX8, tilesY8, 64, 64);
  const c32 = leaves.filter(l => l.w === 32 && l.h === 32);
  const c16 = leaves.filter(l => l.w === 16 && l.h === 16);
  const c8  = leaves.filter(l => l.w === 8 && l.h === 8);
  checkEqual("single-8x8: 3x32 (other quadrants)", c32.length, 3);
  checkEqual("single-8x8: 3x16 (clean 16x16 children of affected 32x32)", c16.length, 3);
  checkEqual("single-8x8: 4x8 (16x16 with salient 8x8 subdivides to 4 8x8)", c8.length, 4);
  const salient = c8.find(l => l.salience === 200);
  check("single-8x8: salient leaf exists with salience=200", !!salient);
  check("single-8x8: salient at (0,0)", salient && salient.x === 0 && salient.y === 0, salient ? `at (${salient.x},${salient.y})` : "");
  check("single-8x8: other 8x8 have salience 0",
    c8.filter(l => l.salience !== 200).every(l => l.salience === 0), true);
  assertCovers("single-8x8", leaves, 64, 64);
}

// buildAdaptiveTiles: 80x80 all-zero, edge truncation
{
  const tilesX8 = 10, tilesY8 = 10;
  const means = new Uint8Array(tilesX8 * tilesY8);
  const leaves = buildAdaptiveTiles(means, tilesX8, tilesY8, 80, 80);
  // 3x3 grid of 32x32 cells. All w32<32 or h32<32 except the 4 corners.
  // (0,0): 32x32, (32,0): 32x32, (64,0): 16x32
  // (0,32): 32x32, (32,32): 32x32, (64,32): 16x32
  // (0,64): 32x16, (32,64): 32x16, (64,64): 16x16
  checkEqual("80x80 all-zero: 9 leaves (no subdivision due to edge)", leaves.length, 9);
  const sizes = leaves.map(l => `${l.w}x${l.h}`).sort();
  checkEqual("80x80 all-zero: size mix",
    sizes.join(","),
    ["16x16","16x32","16x32","32x16","32x16","32x32","32x32","32x32","32x32"].sort().join(","));
  assertCovers("80x80 all-zero", leaves, 80, 80);
}

// buildAdaptiveTiles: 96x96 is exactly 3x3 of 32x32 (no edge truncation).
// Put one salient 8x8 at tile (5,5) which is inside the center 32x32 (tiles 4..7).
// Center 32x32 has 1 salient 8x8 -> allBelow32 false -> subdivide to 4 16x16.
// 3 of those 16x16 are clean -> stay 16x16. 1 (containing the salient 8x8) -> subdivide to 4 8x8.
// Other 8 of 9 32x32 cells are clean -> stay 32x32.
// Expected: 8x 32x32, 3x 16x16, 4x 8x8.
{
  const tilesX8 = 12, tilesY8 = 12;
  const means = new Uint8Array(tilesX8 * tilesY8);
  means[5 * tilesX8 + 5] = 100;
  const leaves = buildAdaptiveTiles(means, tilesX8, tilesY8, 96, 96);
  const c32 = leaves.filter(l => l.w === 32 && l.h === 32);
  const c16 = leaves.filter(l => l.w === 16 && l.h === 16);
  const c8  = leaves.filter(l => l.w === 8 && l.h === 8);
  checkEqual("96x96 mixed: 8x 32x32 (clean 32x32 cells)", c32.length, 8);
  checkEqual("96x96 mixed: 3x 16x16 (clean 16x16 children of center 32x32)", c16.length, 3);
  checkEqual("96x96 mixed: 4x 8x8 (16x16 with salient 8x8 subdivides)", c8.length, 4);
  const salient = c8.find(l => l.salience === 100);
  check("96x96 mixed: salient 8x8 at (40,40)", salient && salient.x === 40 && salient.y === 40, salient ? `at (${salient.x},${salient.y})` : "");
  assertCovers("96x96 mixed", leaves, 96, 96);
}

// buildAdaptiveTiles: 8x8 image (single 8x8)
{
  const leaves = buildAdaptiveTiles(new Uint8Array([42]), 1, 1, 8, 8);
  checkEqual("8x8 image: 1 leaf", leaves.length, 1);
  checkEqual("8x8 image: at (0,0) 8x8 sal=42", leaves[0], { x: 0, y: 0, w: 8, h: 8, salience: 42 });
}

// buildAdaptiveTiles: 7x7 image (still produces a 1x1 8x8 grid, but image is 7x7)
{
  const leaves = buildAdaptiveTiles(new Uint8Array([99]), 1, 1, 7, 7);
  checkEqual("7x7 image: 1 leaf", leaves.length, 1);
  // 8x8 in grid covers beyond image, but leaf should be clamped to 7x7? Let's check.
  // Looking at code: w8 = Math.min(8, imgW - xBase8) = Math.min(8, 7) = 7
  checkEqual("7x7 image: leaf size 7x7", leaves[0].w, 7);
  checkEqual("7x7 image: leaf size h=7", leaves[0].h, 7);
}

// buildAdaptiveTiles: salience values preserved for full subdivisions
// All 16 8x8 tiles nonzero -> 16 leaves of 8x8, each with its own salience.
{
  const tilesX8 = 4, tilesY8 = 4;
  const means = new Uint8Array(16);
  for (let i = 0; i < 16; i++) means[i] = i * 16;
  const leaves = buildAdaptiveTiles(means, tilesX8, tilesY8, 32, 32);
  checkEqual("32x32 all-nonzero: 16 leaves", leaves.length, 16);
  for (let ty = 0; ty < 4; ty++) {
    for (let tx = 0; tx < 4; tx++) {
      const expected = (ty * 4 + tx) * 16;
      const leaf = leaves.find(l => l.x === tx * 8 && l.y === ty * 8);
      check(`32x32 leaf (${tx},${ty}) salience=${expected}`,
        leaf && leaf.salience === expected && leaf.w === 8 && leaf.h === 8,
        leaf ? `got salience=${leaf.salience}` : "leaf not found");
    }
  }
}

// buildUniformTiles: invalid tileSize throws
{
  let threw = false;
  try { buildUniformTiles(new Uint8Array(0), 0, 0, 16, 16, 7); } catch { threw = true; }
  check("buildUniformTiles: rejects tileSize=7", threw);

  threw = false;
  try { buildUniformTiles(new Uint8Array(0), 0, 0, 16, 16, 0); } catch { threw = true; }
  check("buildUniformTiles: rejects tileSize=0", threw);

  threw = false;
  try { buildUniformTiles(new Uint8Array(0), 0, 0, 16, 16, 10); } catch { threw = true; }
  check("buildUniformTiles: rejects tileSize=10 (not multiple of 8)", threw);
}

// buildUniformTiles: 64x64 tileSize=8, all zero -> 64 leaves
{
  const tilesX8 = 8, tilesY8 = 8;
  const means = new Uint8Array(tilesX8 * tilesY8);
  const leaves = buildUniformTiles(means, tilesX8, tilesY8, 64, 64, 8);
  checkEqual("uniform 8x64: 64 leaves", leaves.length, 64);
  checkEqual("uniform 8x64: all 8x8", leaves.every(l => l.w === 8 && l.h === 8), true);
  checkEqual("uniform 8x64: all salience 0", leaves.every(l => l.salience === 0), true);
  assertCovers("uniform 8x64", leaves, 64, 64);
}

// buildUniformTiles: 64x64 tileSize=16, all 255 -> 16 leaves with salience 255
{
  const tilesX8 = 8, tilesY8 = 8;
  const means = new Uint8Array(tilesX8 * tilesY8).fill(255);
  const leaves = buildUniformTiles(means, tilesX8, tilesY8, 64, 64, 16);
  checkEqual("uniform 16x64: 16 leaves", leaves.length, 16);
  checkEqual("uniform 16x64: all 16x16", leaves.every(l => l.w === 16 && l.h === 16), true);
  checkEqual("uniform 16x64: all salience 255", leaves.every(l => l.salience === 255), true);
  assertCovers("uniform 16x64", leaves, 64, 64);
}

// buildUniformTiles: 64x64 tileSize=32 -> 4 leaves
{
  const tilesX8 = 8, tilesY8 = 8;
  const means = new Uint8Array(tilesX8 * tilesY8).fill(100);
  const leaves = buildUniformTiles(means, tilesX8, tilesY8, 64, 64, 32);
  checkEqual("uniform 32x64: 4 leaves", leaves.length, 4);
  checkEqual("uniform 32x64: all 32x32", leaves.every(l => l.w === 32 && l.h === 32), true);
  checkEqual("uniform 32x64: all salience 100", leaves.every(l => l.salience === 100), true);
  assertCovers("uniform 32x64", leaves, 64, 64);
}

// buildUniformTiles: 80x80 tileSize=32 -> 9 leaves with edge truncation
{
  const tilesX8 = 10, tilesY8 = 10;
  const means = new Uint8Array(tilesX8 * tilesY8);
  const leaves = buildUniformTiles(means, tilesX8, tilesY8, 80, 80, 32);
  checkEqual("uniform 32x80: 9 leaves", leaves.length, 9);
  const sizes = leaves.map(l => `${l.w}x${l.h}`).sort();
  checkEqual("uniform 32x80: size mix",
    sizes.join(","),
    ["16x16","16x32","16x32","32x16","32x16","32x32","32x32","32x32","32x32"].sort().join(","));
  assertCovers("uniform 32x80", leaves, 80, 80);
}

// buildUniformTiles: aggregates 8x8 means (mixed salience in a 32x32)
{
  const tilesX8 = 8, tilesY8 = 8;
  const means = new Uint8Array(tilesX8 * tilesY8);
  // 32x32 at (0,0): top-left 8x8 = 200, rest = 0
  means[0] = 200;
  const leaves = buildUniformTiles(means, tilesX8, tilesY8, 64, 64, 32);
  checkEqual("uniform mixed: 4 leaves", leaves.length, 4);
  const affected = leaves.find(l => l.x === 0 && l.y === 0);
  check("uniform mixed: affected leaf exists at (0,0)", !!affected);
  // 1 of 16 8x8 is 200, 15 are 0, average = 12.5
  check("uniform mixed: salience averaged from 8x8 means",
    affected && Math.abs(affected.salience - 12.5) < 1e-6, affected ? `got ${affected.salience}` : "");
  // others should be 0
  const others = leaves.filter(l => !(l.x === 0 && l.y === 0));
  checkEqual("uniform mixed: other 3 leaves salience 0",
    others.every(l => l.salience === 0), true);
}

// buildUniformTiles: tileSize=64 on 64x64 -> 1 leaf
{
  const tilesX8 = 8, tilesY8 = 8;
  const means = new Uint8Array(tilesX8 * tilesY8).fill(50);
  const leaves = buildUniformTiles(means, tilesX8, tilesY8, 64, 64, 64);
  checkEqual("uniform 64x64: 1 leaf", leaves.length, 1);
  checkEqual("uniform 64x64: 64x64", leaves[0].w, 64);
  checkEqual("uniform 64x64: h=64", leaves[0].h, 64);
  checkEqual("uniform 64x64: salience 50", leaves[0].salience, 50);
}

// buildUniformTiles: tileSize=256 on 64x64 -> 1 truncated 64x64 leaf
{
  const tilesX8 = 8, tilesY8 = 8;
  const means = new Uint8Array(tilesX8 * tilesY8);
  const leaves = buildUniformTiles(means, tilesX8, tilesY8, 64, 64, 256);
  checkEqual("uniform 256x64: 1 leaf", leaves.length, 1);
  checkEqual("uniform 256x64: clamped to 64x64", leaves[0], { x: 0, y: 0, w: 64, h: 64, salience: 0 });
}

// buildUniformTiles: 100x100 tileSize=64 -> 4 leaves (2x2), some truncated
{
  const tilesX8 = 13, tilesY8 = 13;
  const means = new Uint8Array(tilesX8 * tilesY8).fill(0);
  // Make top-left 8x8 salient
  means[0] = 200;
  const leaves = buildUniformTiles(means, tilesX8, tilesY8, 100, 100, 64);
  checkEqual("uniform 64x100: 4 leaves", leaves.length, 4);
  // TL: 64x64, TR: 36x64, BL: 64x36, BR: 36x36
  const tl = leaves.find(l => l.x === 0 && l.y === 0);
  const tr = leaves.find(l => l.x === 64 && l.y === 0);
  const bl = leaves.find(l => l.x === 0 && l.y === 64);
  const br = leaves.find(l => l.x === 64 && l.y === 64);
  check("uniform 64x100: TL 64x64", tl && tl.w === 64 && tl.h === 64);
  check("uniform 64x100: TR 36x64", tr && tr.w === 36 && tr.h === 64);
  check("uniform 64x100: BL 64x36", bl && bl.w === 64 && bl.h === 36);
  check("uniform 64x100: BR 36x36", br && br.w === 36 && br.h === 36);
  // TL aggregates 8x8 means: 1 is 200, 63 are 0, average = 200/64
  check("uniform 64x100: TL salience averaged",
    tl && Math.abs(tl.salience - (200 / 64)) < 1e-6, tl ? `got ${tl.salience}` : "");
  assertCovers("uniform 64x100", leaves, 100, 100);
}

console.log(`\n${passed} passed, ${failed} failed`);
if (failed > 0) {
  console.log("\nfailures:");
  for (const f of failures) console.log("  -", f.name, f.detail);
  process.exit(1);
}
