# gpu-adept test results — lena.q100.jpg

**Source:** `original/lena.q100.jpg` (512×512, Q100 JPEG, 394.8 KB)

| Mode            | Tiles                             | Low % | Output   | Saved |
|-----------------|-----------------------------------|-------|----------|-------|
| Auto (adaptive) | 1453 (145×32, 156×16, 1152×8)    | 86%   | 61.0 KB  | 84.6% |
| Uniform 8×8     | 4096                              | 95%   | 66.0 KB  | 83.3% |
| Uniform 16×16   | 1024                              | 98%   | 57.2 KB  | 85.5% |
| Uniform 32×32   | 256                               | 100%  | 55.7 KB  | 85.9% |
| Uniform 64×64   | 64                                | 100%  | 54.1 KB  | 86.3% |
| Uniform 128×128 | 16                                | 100%  | 53.2 KB  | 86.5% |
| Uniform 256×256 | 4                                 | 100%  | 52.6 KB  | 86.7% |

Tested 2025-06-06 via Chrome DevTools MCP with WebGPU.
