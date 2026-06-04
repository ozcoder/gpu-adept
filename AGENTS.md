# gpu-adept

Adaptive JPEG compressor using WebGPU.

## Commands

```
npm install          # installs Vite
npm run dev          # Vite dev server, opens demo/index.html
npm run build        # production build
```

Open `/demo/` for the demo, `/test/validate.html` for validation against reference.

## Architecture

Browser-only. Uses Vite, WebGPU compute, and Canvas 2D APIs.

```
                  Input ImageData
                        │
              SaliencyPipeline.setup()
              (upload RGBA to GPU)
                        │
           ┌─ computeSaliency() ──────────────────────┐
           │  RGB→Lab → Gaussian blur → integral      │
           │  images → saliency → normalize →          │
           │  histogram (GPU, 256-bin atomic)          │
           └──────────────┬───────────────────────────┘
                          │ readback 1 KB histogram
                          ▼
              Binary search for optimal b/w threshold
              (CPU on histogram data, 256 bins only)
                          │
           ┌─ analyzeTiles(threshold) ───────────────┐
           │  per-tile mean via tree reduction (GPU)  │
           └──────────────┬───────────────────────────┘
                          │ readback tile means
                          ▼
              snap quality to 16 discrete levels
              (64, 66, …, 94, 96)
                          │
               ┌──────────┴──────────┐
               │                     │
          quality < 96           quality = 96
               │                     │
      batch encode by         draw original
      quality level           pixels directly
      (≤16 toBlob calls)
               │                     │
               └──────────┬──────────┘
                          ▼
               Output canvas → toBlob('image/jpeg', Q=96)
```

## Dependencies

- **Vite** — dev server and bundler.

## Files

| Path | Role |
|---|---|
| `src/adept.js` | `AdeptJPEG` class — `create()`, `compress(imageData, opts)`, `destroy()`. Owns device + `SaliencyPipeline`. |
| `src/pipeline.js` | `SaliencyPipeline` — unified GPU pipeline. `setup()`, `computeSaliency()` (histogram readback), `analyzeTiles()` (tile means readback). |
| `src/shaders.js` | All WGSL shaders: saliency pipeline (RGB→Lab, Gaussian blur, integral images, saliency, reduce, normalize) + histogram + tile analysis. |
| `src/tile-processor.js` | `processTiles()` — groups tiles by quality level, encodes each group as a batch OffscreenCanvas (≤16 `toBlob` calls total). Edge tiles padded individually. |
| `src/utils.js` | `imageDataFromFile`, `autoTileSize`, `tileCount`, `padTo8`, `downloadBlob`. |
| `demo/demo.js` | Demo controller — drag-and-drop, tile size selector, progress, download. |
| `demo/index.html` | Demo page (referenced from root `index.html`). |
| `test/validate.html` | Validation page — compares gpu-adept output to reference ImageMagick output. |
| `test/validate.js` | Validation logic — loads test images, runs pipeline, computes MSE + size ratio. |
| `original/` | Reference bash script + test images. Untouched. |

## Key implementation details

- **Saliency pipeline**: RGB→Lab, 3×3 Gaussian blur, integral images via segmented scan (256-wide segments), saliency = squared Lab difference from local mean, tree-reduction for global min/max, normalize to 0–255.
- **Histogram**: 256-bin atomic counters on the normalized saliency output. Readback is 1 KB regardless of image size.
- **Binary search**: CPU-side on histogram bins (not full image). Target b/w mean between 20–40 (0–255 scale).
- **Tile analysis shader**: Workgroup (16,16) = 256 threads per tile. Each thread handles `ceil(tileSize/16)²` pixels. Tree reduction via `var<workgroup>` shared memory.
- **Batch encode**: Tiles grouped by quality level. All tiles at the same quality are packed into one OffscreenCanvas → `toBlob` once → decoded → distributed to output. At most 16 encodes for any image size.
- **Edge tiles**: Tiles < 8px in any dimension are mirror-padded to 8×8 before JPEG encode, cropped after decode. Handled individually (at most `tilesX + tilesY - 1` such tiles).
- **Quality levels**: 16 discrete levels from 64 to 94 (step 2). Level 96 means "skip encode" — original pixels used directly.
- **Output**: Final composite encoded at Q96.
- **Tile size**: Auto-detect: ≤512px → 8, ≤1024px → 16, >1024px → 32. User override via `opts.tileSize`.

## Reference algorithm

`original/adept-jpeg.sh` — same logic using ImageMagick + MSS saliency binary. Test images in `original/` — Lena variants in PNG and JPEG.

## Limits

- **WebGPU required** (Chrome 113+, Edge 113+).
- **No Node.js support** — browser-only (Canvas, OffscreenCanvas, WebGPU).
- **gpu-msss** has been removed — all saliency shaders are now in `src/shaders.js`.
- 4K images need ~700 MB GPU memory. 8K exceeds most consumer GPU budgets.
