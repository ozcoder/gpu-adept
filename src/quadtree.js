// Build a quadtree of variable-size tiles (8, 16, or 32) from 8×8 tile means.
// Subdivision rule: keep a 32×32 region as one tile only if all 16 of its
// 8×8 children are non-salient (binarized mean = 0). Otherwise subdivide
// into four 16×16, and subdivide each of those into four 8×8 if any of
// its 8×8 children is salient. Yields non-overlapping leaves that tile
// the image exactly. Truncated 32×32 regions at the image edge are kept
// as leaves (cannot subdivide into valid 16×16 children).

export function buildAdaptiveTiles(tileMeans8, tilesX8, tilesY8, imgW, imgH) {
  const tilesX32 = Math.ceil(tilesX8 / 4);
  const tilesY32 = Math.ceil(tilesY8 / 4);

  const leaves = [];

  for (let y32 = 0; y32 < tilesY32; y32++) {
    for (let x32 = 0; x32 < tilesX32; x32++) {
      const xBase = x32 * 32;
      const yBase = y32 * 32;
      const w32 = Math.min(32, imgW - xBase);
      const h32 = Math.min(32, imgH - yBase);

      let allBelow32 = true;
      let sum32 = 0, cnt32 = 0;
      for (let dy = 0; dy < 4; dy++) {
        for (let dx = 0; dx < 4; dx++) {
          const tx = x32 * 4 + dx;
          const ty = y32 * 4 + dy;
          if (tx < tilesX8 && ty < tilesY8) {
            const m = tileMeans8[ty * tilesX8 + tx];
            sum32 += m;
            cnt32++;
            if (m > 0) allBelow32 = false;
          }
        }
      }
      const mean32 = cnt32 > 0 ? sum32 / cnt32 : 0;

      if (w32 < 32 || h32 < 32 || allBelow32) {
        leaves.push({ x: xBase, y: yBase, w: w32, h: h32, salience: mean32 });
        continue;
      }

      for (let y16 = 0; y16 < 2; y16++) {
        for (let x16 = 0; x16 < 2; x16++) {
          const xBase16 = xBase + x16 * 16;
          const yBase16 = yBase + y16 * 16;

          let allBelow16 = true;
          let sum16 = 0, cnt16 = 0;
          for (let dy = 0; dy < 2; dy++) {
            for (let dx = 0; dx < 2; dx++) {
              const tx = x32 * 4 + x16 * 2 + dx;
              const ty = y32 * 4 + y16 * 2 + dy;
              if (tx < tilesX8 && ty < tilesY8) {
                const m = tileMeans8[ty * tilesX8 + tx];
                sum16 += m;
                cnt16++;
                if (m > 0) allBelow16 = false;
              }
            }
          }
          const mean16 = cnt16 > 0 ? sum16 / cnt16 : 0;

          if (allBelow16) {
            leaves.push({ x: xBase16, y: yBase16, w: 16, h: 16, salience: mean16 });
          } else {
            for (let dy = 0; dy < 2; dy++) {
              for (let dx = 0; dx < 2; dx++) {
                const xBase8 = xBase16 + dx * 8;
                const yBase8 = yBase16 + dy * 8;
                const tx = x32 * 4 + x16 * 2 + dx;
                const ty = y32 * 4 + y16 * 2 + dy;
                let m8 = 0;
                if (tx < tilesX8 && ty < tilesY8) {
                  m8 = tileMeans8[ty * tilesX8 + tx];
                }
                const w8 = Math.min(8, imgW - xBase8);
                const h8 = Math.min(8, imgH - yBase8);
                if (w8 <= 0 || h8 <= 0) continue;
                leaves.push({ x: xBase8, y: yBase8, w: w8, h: h8, salience: m8 });
              }
            }
          }
        }
      }
    }
  }

  return leaves;
}

export function buildUniformTiles(tileMeans8, tilesX8, tilesY8, imgW, imgH, tileSize) {
  if (tileSize < 8 || tileSize % 8 !== 0) {
    throw new Error(`Invalid tileSize: ${tileSize} (must be a positive multiple of 8)`);
  }
  const sx = tileSize / 8;
  const tilesX = Math.ceil(imgW / tileSize);
  const tilesY = Math.ceil(imgH / tileSize);
  const leaves = [];

  for (let ty = 0; ty < tilesY; ty++) {
    for (let tx = 0; tx < tilesX; tx++) {
      let sum = 0, cnt = 0;
      for (let dy = 0; dy < sx; dy++) {
        for (let dx = 0; dx < sx; dx++) {
          const ttx = tx * sx + dx;
          const tty = ty * sx + dy;
          if (ttx < tilesX8 && tty < tilesY8) {
            sum += tileMeans8[tty * tilesX8 + ttx];
            cnt++;
          }
        }
      }
      const x = tx * tileSize;
      const y = ty * tileSize;
      const w = Math.min(tileSize, imgW - x);
      const h = Math.min(tileSize, imgH - y);
      const salience = cnt > 0 ? sum / cnt : 0;
      leaves.push({ x, y, w, h, salience });
    }
  }

  return leaves;
}
