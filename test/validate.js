import { AdeptJPEG } from "../src/adept.js";
import { imageDataFromURL, downloadBlob } from "../src/utils.js";

const runBtn = document.getElementById("run-btn");
const testImageSelect = document.getElementById("test-image");
const tileSizeSelect = document.getElementById("tile-size");
const statusEl = document.getElementById("status");

const originalImg = document.getElementById("original-img");
const referenceImg = document.getElementById("reference-img");
const resultImg = document.getElementById("result-img");
const diffCanvas = document.getElementById("diff-canvas");
const originalInfo = document.getElementById("original-info");
const referenceInfo = document.getElementById("reference-info");
const resultInfo = document.getElementById("result-info");
const diffInfo = document.getElementById("diff-info");

let adept = null;
let inputImageData = null;
const ORIG_PATH = "/original/lena.q100.jpg";

async function loadReferenceImages() {
  const inputPath = testImageSelect.value;
  const refPath = inputPath.replace(/\.jpg$/, "_adept_compress_imagemagick.jpg");

  statusEl.textContent = "Loading reference images...";

  const [inputData, refResp] = await Promise.all([
    imageDataFromURL(ORIG_PATH),
    fetch(refPath),
  ]);

  inputImageData = inputData;

  const refBlob = await refResp.blob();
  const refURL = URL.createObjectURL(refBlob);

  originalImg.src = ORIG_PATH;
  referenceImg.src = refURL;
  originalImg.onload = () => {
    originalInfo.textContent = `${inputData.width}×${inputData.height}`;
  };
  referenceImg.onload = () => {
    referenceInfo.textContent = `${refBlob.size} B`;
  };

  runBtn.disabled = false;
  statusEl.textContent = "Ready. Click 'Run Validation' to test.";
}

runBtn.addEventListener("click", async () => {
  if (!inputImageData) return;
  runBtn.disabled = true;
  statusEl.textContent = "Initializing WebGPU...";

  try {
    if (!adept) {
      adept = await AdeptJPEG.create();
    }

    const { width, height } = inputImageData;

    let tileSize = tileSizeSelect.value;
    if (tileSize !== "auto") tileSize = parseInt(tileSize, 10);
    const tileDesc = tileSize === "auto" ? "adaptive tiles" : `uniform ${tileSize}×${tileSize} tiles`;

    statusEl.textContent = `Compressing with ${tileDesc}...`;

    const start = performance.now();
    const result = await adept.compress(inputImageData, {
      tileSize,
      inputQuality: 100,
      onProgress: (done, total) => {
        if (typeof done === "number") {
          statusEl.textContent = `Encoding tiles... ${Math.round(done / total * 100)}%`;
        } else {
          statusEl.textContent = done;
        }
      },
    });
    const elapsed = ((performance.now() - start) / 1000).toFixed(1);

    const resultURL = URL.createObjectURL(result.blob);
    resultImg.src = resultURL;
    resultImg.onload = () => {
      resultInfo.textContent = `${result.blob.size} B (${elapsed}s)`;
    };

    // Compute difference
    const resultData = await imageDataFromURL(resultURL);
    const diffCtx = diffCanvas.getContext("2d");
    diffCanvas.width = width;
    diffCanvas.height = height;

    const refData = await imageDataFromURL(referenceImg.src);
    const diffImageData = diffCtx.createImageData(width, height);
    const diff = new Uint8ClampedArray(diffImageData.data);

    let mse = 0;
    for (let i = 0; i < diff.length; i += 4) {
      const dr = Math.abs(resultData.data[i] - refData.data[i]);
      const dg = Math.abs(resultData.data[i + 1] - refData.data[i + 1]);
      const db = Math.abs(resultData.data[i + 2] - refData.data[i + 2]);
      mse += (dr * dr + dg * dg + db * db) / 3;
      diff[i] = Math.min(255, dr * 10);
      diff[i + 1] = Math.min(255, dg * 10);
      diff[i + 2] = Math.min(255, db * 10);
      diff[i + 3] = 255;
    }
    mse /= width * height;
    diffImageData.data.set(diff);
    diffCtx.putImageData(diffImageData, 0, 0);

    const inputResp = await fetch(ORIG_PATH);
    const inputBlob = await inputResp.blob();
    const refResp = await fetch(referenceImg.src);
    const refBlob2 = await refResp.blob();
    const ratio = ((1 - result.blob.size / refBlob2.size) * 100).toFixed(1);

    diffInfo.innerHTML = `
      MSE: ${mse.toFixed(1)} |
      Size ratio (adept / ref): ${ratio}% |
      Time: ${elapsed}s
    `;
    statusEl.innerHTML = `<span class="pass">Validation complete.</span>`;

    // Add download button
    const downloadBtn = document.createElement("button");
    downloadBtn.textContent = "Download result";
    downloadBtn.style.marginTop = "1rem";
    downloadBtn.onclick = () => downloadBlob(resultBlob, "adept-output.jpg");
    statusEl.after(downloadBtn);

  } catch (err) {
    statusEl.innerHTML = `<span class="fail">Error: ${err.message}</span>`;
    console.error(err);
  } finally {
    runBtn.disabled = false;
  }
});

loadReferenceImages();
