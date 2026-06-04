# gpu-adept

Adaptive JPEG compressor using WebGPU.

## Commands

```
npm install          # installs Vite + gpu-msss (GitHub dep)
npm run dev          # Vite dev server, opens demo/index.html
npm run build        # production build
```

Open `/demo/` for the demo, `/test/validate.html` for validation against reference.

## Architecture

Browser-only. Uses Vite, WebGPU compute, and Canvas 2D APIs.

```
Input ImageData ──► MSSS.compute() ──► raw saliency
                    (gpu-msss, GPU)
                         │
                    Binary search for optimal b/w threshold (CPU)
                         │
                    TileAnalysisPipeline.analyze() ──► per-tile means
                    (WGSL compute shader, one workgroup per tile)
                         │
                    classify: mean < 0.825 → low complexity
                         │
              ┌──────────┴──────────┐
              │                     │
         low complexity        high complexity
              │                     │
     OffscreenCanvas           keep original
     → toBlob(JPEG, Q=69)      pixels
     → createImageBitmap
     → drawImage
              │                     │
              └──────────┬──────────┘
                         ▼
              Output canvas → toBlob('image/jpeg', quality)
```

## Dependencies

- **gpu-msss** (GitHub: `ozcoder/gpu-msss`) — WebGPU MSS saliency. Imported as `import { MSSS } from "gpu-msss"`. Uses `vite.config.js` resolve alias to point at `node_modules/gpu-msss/src/msss.js`.
- **Vite** — dev server and bundler.

## Files

| Path | Role |
|---|---|
| `src/adept.js` | `AdeptJPEG` class — `create()`, `compress(imageData, opts)`, `destroy()`. Owns device + MSSS. |
| `src/tile-analysis.js` | `TileAnalysisPipeline` — WebGPU pipeline for per-tile saliency mean. Dispatch `tilesX × tilesY` workgroups. |
| `src/shaders.js` | WGSL `tileAnalysis` shader. Workgroup size 16×16, tree reduction in workgroup shared memory. |
| `src/tile-processor.js` | `processTiles()` — CPU-side tile encode/decode/composite. Uses OffscreenCanvas per tile. |
| `src/utils.js` | `imageDataFromFile`, `autoTileSize`, `tileCount`, `padTo8`, `downloadBlob`. |
| `demo/demo.js` | Demo controller — drag-and-drop, tile size selector, progress, download. |
| `demo/index.html` | Demo page (referenced from root `index.html`). |
| `test/validate.html` | Validation page — compares gpu-adept output to reference ImageMagick output. |
| `test/validate.js` | Validation logic — loads test images, runs pipeline, computes MSE + size ratio. |
| `original/` | Reference bash script + test images. Untouched. |

## Key implementation details

- **Binary search**: CPU-side on raw saliency data. Matches reference: target b/w mean between 20–40 (0–255 scale). Uses `msss.compute()` output directly, no extra GPU reads.
- **Tile analysis shader**: Workgroup (16,16) = 256 threads per tile. Each thread handles `ceil(tileSize/16)²` pixels. Tree reduction via `var<workgroup>` shared memory.
- **Per-tile JPEG**: Low-complexity tiles encoded via `OffscreenCanvas.convertToBlob({type:'image/jpeg', quality: 0.69})`. Decoded via `createImageBitmap(blob)`. High-complexity tiles skip encode/decode entirely (better than reference which re-encodes all tiles).
- **Edge tiles**: Tiles < 8px in any dimension are mirror-padded to 8×8 before JPEG encode, cropped after decode.
- **Quality defaults**: High-quality = inherit from input JPEG (parse quantization table), or 100 for PNG inputs. Low-quality = 69.
- **Tile size**: Auto-detect: ≤512px → 8, ≤1024px → 16, >1024px → 32. User override via `opts.tileSize`.

## Reference algorithm

`original/adept-jpeg.sh` — same logic using ImageMagick + MSS saliency binary. Test images in `original/` — Lena variants in PNG and JPEG.

## Limits

- **WebGPU required** (Chrome 113+, Edge 113+).
- **No Node.js support** — browser-only (Canvas, OffscreenCanvas, WebGPU).
- **gpu-msss** limits: 4K images need ~700 MB GPU memory. 8K exceeds most consumer GPU budgets.
