import { AdeptJPEG } from "../src/adept.js";
import { imageDataFromFile, downloadBlob, autoTileSize } from "../src/utils.js";

const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const preview = document.getElementById("preview");
const controls = document.getElementById("controls");
const compressBtn = document.getElementById("compress-btn");
const tileSizeSelect = document.getElementById("tile-size");
const statusEl = document.getElementById("status");
const result = document.getElementById("result");
const resultImg = document.getElementById("result-img");
const statOriginal = document.getElementById("stat-original");
const statCompressed = document.getElementById("stat-compressed");
const statRatio = document.getElementById("stat-ratio");
const downloadBtn = document.getElementById("download-btn");
const debugCheck = document.getElementById("debug-check");
const debugTable = document.getElementById("debug-table");

let currentFile = null;
let currentImageData = null;
let adept = null;

dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) {
    loadImage(file);
  }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) {
    loadImage(fileInput.files[0]);
  }
});

async function loadImage(file) {
  currentFile = file;
  currentImageData = await imageDataFromFile(file);

  const url = URL.createObjectURL(file);
  preview.src = url;
  preview.style.display = "block";
  dropZone.classList.add("has-image");
  controls.classList.add("visible");
  compressBtn.disabled = false;
  result.classList.remove("visible");
  debugTable.classList.remove("visible");

  statusEl.textContent = `Loaded ${file.name} (${currentImageData.width}×${currentImageData.height}, ${formatSize(file.size)})`;
}

compressBtn.addEventListener("click", async () => {
  if (!currentImageData || !currentFile) return;

  compressBtn.disabled = true;
  statusEl.textContent = "Initializing WebGPU...";
  result.classList.remove("visible");

  try {
    if (!adept) {
      statusEl.textContent = "Creating WebGPU device...";
      adept = await AdeptJPEG.create();
    }

    let tileSize = tileSizeSelect.value;
    if (tileSize === "auto") {
      tileSize = autoTileSize(currentImageData.width, currentImageData.height);
    } else {
      tileSize = parseInt(tileSize, 10);
    }

    const inputQuality = await detectJPEGQuality(currentFile);

    statusEl.textContent = "Compressing...";

    const info = await adept.compress(currentImageData, {
      tileSize,
      inputQuality,
      onProgress: (msg) => {
        statusEl.textContent = msg;
      },
    });

    const resultURL = URL.createObjectURL(info.blob);
    resultImg.src = resultURL;
    result.classList.add("visible");

    const originalSize = currentFile.size;
    const compressedSize = info.blob.size;
    const ratio = ((1 - compressedSize / originalSize) * 100).toFixed(1);

    statOriginal.textContent = `Original: ${formatSize(originalSize)}`;
    const pctLow = ((info.lowComplexityCount / info.totalTiles) * 100).toFixed(0);
    statCompressed.textContent =
      `${info.tileSize}×${info.tileSize} tiles · ` +
      `${info.tilesX}×${info.tilesY} grid · ` +
      `${pctLow}% tiles @ Q${info.lowQuality} (low-complexity), ` +
      `rest kept original · output @ Q${info.highQuality} · ` +
      `${formatSize(compressedSize)}`;
    statRatio.textContent = `Saved: ${ratio}%`;

    if (compressedSize >= originalSize) {
      statusEl.textContent = "Warning: output is not smaller than input. Try a smaller tile size or lower quality.";
    } else {
      statusEl.textContent = "Done!";
    }

    if (debugCheck.checked) {
      renderDebugTable(info);
    }

    downloadBtn.onclick = () => {
      const name = currentFile.name.replace(/\.[^.]+$/, "_adept.jpg");
      downloadBlob(info.blob, name);
    };
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
    console.error(err);
  } finally {
    compressBtn.disabled = false;
  }
});

async function detectJPEGQuality(file) {
  if (!file.type || file.type === "image/jpeg") {
    try {
      const buf = await file.slice(0, Math.min(file.size, 65536)).arrayBuffer();
      const view = new DataView(buf);
      let offset = 2;

      while (offset + 1 < view.byteLength) {
        if (view.getUint8(offset) !== 0xFF) break;
        const marker = view.getUint8(offset + 1);
        if (marker === 0xDB) {
          offset += 4;
          if (offset + 64 <= view.byteLength) {
            let sum = 0;
            for (let i = 0; i < 64; i++) {
              sum += view.getUint8(offset + i);
            }
            const q = Math.max(1, Math.min(100, Math.round(5000 / sum * 64)));
            return q;
          }
          break;
        }
        if (marker === 0xDA) break;
        if (marker === 0xD9) break;
        if (marker >= 0xD0 && marker <= 0xD8) {
          offset += 2;
          continue;
        }
        if (offset + 3 > view.byteLength) break;
        const segLen = view.getUint16(offset + 2, false);
        offset += 2 + segLen;
      }
    } catch {}
  }
  return 100;
}

function renderDebugTable(info) {
  debugTable.classList.add("visible");
  const qs = info.tileQualities;
  const cols = info.tilesX;
  const rows = info.tilesY;
  const mask = info.lowQuality;

  let html = "<table>";
  for (let r = 0; r < rows; r++) {
    html += "<tr>";
    for (let c = 0; c < cols; c++) {
      const q = qs[r * cols + c];
      const cls = q === mask ? "low" : "high";
      html += `<td class="${cls}">${q}</td>`;
    }
    html += "</tr>";
  }
  html += "</table>";
  debugTable.innerHTML = html;
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
