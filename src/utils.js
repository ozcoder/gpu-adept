export function imageDataFromFile(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0);
      resolve(ctx.getImageData(0, 0, img.width, img.height));
    };
    img.onerror = reject;
    img.src = URL.createObjectURL(file);
  });
}

export function imageDataFromURL(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0);
      resolve(ctx.getImageData(0, 0, img.width, img.height));
    };
    img.onerror = reject;
    img.src = url;
  });
}

export function downloadBlob(blob, filename = "adept-compressed.jpg") {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function autoTileSize(w, h) {
  const d = Math.min(w, h);
  if (d <= 512) return 8;
  if (d <= 1024) return 16;
  return 32;
}

export function tileCount(dim, tileSize) {
  return Math.ceil(dim / tileSize);
}

export function padTo8(w, h) {
  return {
    w: Math.max(w + (w % 8 === 0 ? 0 : 8 - (w % 8)), 8),
    h: Math.max(h + (h % 8 === 0 ? 0 : 8 - (h % 8)), 8),
  };
}
