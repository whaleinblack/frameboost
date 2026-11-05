// Ground-truth check: render frames at t=0 and t=1, synthesize t=0.5, compare to the true t=0.5.
// Baselines are what the viewer would get without us: frame repeat and a plain cross-fade.
import { readFileSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";

const here = dirname(fileURLToPath(import.meta.url));
const outDir = join(here, ".output");
mkdirSync(outDir, { recursive: true });

const browser = await chromium.launch({
  channel: "msedge",
  headless: true,
  args: ["--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist"],
});
const page = await browser.newPage();
page.on("pageerror", (error) => console.error("page error:", error.message));
await page.setContent("<!doctype html><canvas id=out></canvas>");
await page.addScriptTag({ content: readFileSync(join(here, "..", "src", "engine.js"), "utf8") });

const results = await page.evaluate(async () => {
  const W = 1280;
  const H = 720;
  const rng = (seed) => () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);

  function texture(seed, size) {
    const c = new OffscreenCanvas(size, size);
    const g = c.getContext("2d");
    const r = rng(seed);
    g.fillStyle = "#6b7f8f";
    g.fillRect(0, 0, size, size);
    for (let i = 0; i < 900; i++) {
      g.fillStyle = `hsl(${r() * 360} ${20 + r() * 60}% ${15 + r() * 70}%)`;
      const s = 6 + r() * 70;
      g.fillRect(r() * size, r() * size, s, s * (0.3 + r()));
    }
    return c;
  }
  const background = texture(11, 3000);
  const sprite = texture(29, 400);
  const sprite2 = texture(53, 300);

  // Sub-pixel positions on purpose: real footage never moves by whole pixels.
  const scenes = {
    "慢速平移 4px": (g, t) => g.drawImage(background, -300 - t * 4.3, -200 - t * 1.7),
    "中速平移 16px": (g, t) => g.drawImage(background, -300 - t * 16.3, -200 - t * 5.2),
    "快速平移 48px": (g, t) => g.drawImage(background, -300 - t * 48.6, -200 - t * 9.4),
    "前景物体 + 静止背景": (g, t) => {
      g.drawImage(background, -300, -200);
      g.drawImage(sprite, 150 + t * 28.5, 150 + t * 6.5);
      g.drawImage(sprite2, 800 - t * 20.5, 300 + t * 11.5);
    },
    "前景物体 + 平移背景": (g, t) => {
      g.drawImage(background, -300 - t * 12.3, -200 + t * 3.6);
      g.drawImage(sprite, 150 + t * 24.5, 150 - t * 8.5);
    },
  };

  const draw = (scene, t) => {
    const c = new OffscreenCanvas(W, H);
    const g = c.getContext("2d", { willReadFrequently: true });
    g.filter = "blur(0.6px)"; // cameras do not produce perfectly sharp edges
    scene(g, t);
    return c;
  };
  const pixels = (c) => c.getContext("2d").getImageData(0, 0, W, H).data;
  const psnr = (a, b) => {
    let sum = 0;
    for (let i = 0; i < a.length; i += 4) {
      for (let k = 0; k < 3; k++) sum += (a[i + k] - b[i + k]) ** 2;
    }
    return 10 * Math.log10((255 * 255) / (sum / ((a.length / 4) * 3)));
  };

  const output = document.getElementById("out");
  output.width = W;
  output.height = H;
  const compileStart = performance.now();
  const engine = new FrameBoostEngine.FlowInterpolator(output, { quality: "medium" });
  const compileMs = performance.now() - compileStart;
  const gl = engine.gl;
  const viewport = { x: 0, y: 0, width: W, height: H };
  const crop = { x: 0, y: 0, width: 1, height: 1 };

  const readOutput = () => {
    const raw = new Uint8Array(W * H * 4);
    gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, raw);
    const flipped = new Uint8ClampedArray(raw.length);
    for (let y = 0; y < H; y++) {
      flipped.set(raw.subarray((H - 1 - y) * W * 4, (H - y) * W * 4), y * W * 4);
    }
    return flipped;
  };
  const toPng = async (data) => {
    const c = new OffscreenCanvas(W, H);
    c.getContext("2d").putImageData(new ImageData(data, W, H), 0, 0);
    const blob = await c.convertToBlob({ type: "image/png" });
    const bytes = new Uint8Array(await blob.arrayBuffer());
    return btoa(Array.from(bytes, (b) => String.fromCharCode(b)).join(""));
  };

  const rows = [];
  const images = {};
  for (const [name, scene] of Object.entries(scenes)) {
    // Three frames: the first pair seeds the temporal predictor, as in real playback.
    const f0 = draw(scene, -1);
    const a = draw(scene, 0);
    const mid = draw(scene, 0.5);
    const b = draw(scene, 1);
    engine.reset();
    engine.pushFrame(f0, W, H, -1);
    engine.pushFrame(a, W, H, 0);
    engine.pushFrame(b, W, H, 1);
    engine.render(0.5, viewport, crop);
    const synth = readOutput();

    const pa = pixels(a);
    const pb = pixels(b);
    const pm = pixels(mid);
    const blend = new Uint8ClampedArray(pa.length);
    for (let i = 0; i < pa.length; i++) blend[i] = (pa[i] + pb[i]) / 2;
    rows.push({
      name,
      engine: +psnr(synth, pm).toFixed(2),
      crossfade: +psnr(blend, pm).toFixed(2),
      repeat: +psnr(pa, pm).toFixed(2),
    });
    images[name] = await toPng(synth);
  }

  // A hard cut must stay a clean cut: nearest real frame, no warp soup.
  engine.reset();
  const cutA = draw(scenes["中速平移 16px"], 0);
  const cutB = (() => {
    const c = new OffscreenCanvas(W, H);
    const g = c.getContext("2d", { willReadFrequently: true });
    g.drawImage(sprite, 0, 0, W, H);
    return c;
  })();
  engine.pushFrame(cutA, W, H, 0);
  engine.pushFrame(cutB, W, H, 1);
  engine.render(0.25, viewport, crop);
  const cutEarly = psnr(readOutput(), pixels(cutA));
  engine.render(0.75, viewport, crop);
  const cutLate = psnr(readOutput(), pixels(cutB));

  // Submit + GPU completion for one frame, at 1080p source.
  const bigA = new OffscreenCanvas(1920, 1080);
  bigA.getContext("2d").drawImage(background, 0, 0);
  const bigB = new OffscreenCanvas(1920, 1080);
  bigB.getContext("2d").drawImage(background, -13, -4);
  const timing = {};
  for (const quality of ["low", "medium", "high"]) {
    engine.setQuality(quality);
    engine.pushFrame(bigA, 1920, 1080, 0);
    engine.pushFrame(bigB, 1920, 1080, 1);
    gl.finish();
    const samples = [];
    for (let i = 0; i < 20; i++) {
      const start = performance.now();
      engine.pushFrame(i % 2 ? bigB : bigA, 1920, 1080, 2 + i);
      const px = new Uint8Array(4);
      gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
      samples.push(performance.now() - start);
    }
    samples.sort((x, y) => x - y);
    timing[quality] = +samples[10].toFixed(1);
  }
  const debug = gl.getExtension("WEBGL_debug_renderer_info");
  return {
    rows,
    images,
    cutEarly: +cutEarly.toFixed(1),
    cutLate: +cutLate.toFixed(1),
    compileMs: +compileMs.toFixed(0),
    timing,
    renderer: debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : "unknown",
  };
});
await browser.close();

for (const [name, data] of Object.entries(results.images)) {
  writeFileSync(join(outDir, `${name.replace(/[^\w一-龥]+/g, "_")}.png`), Buffer.from(data, "base64"));
}
console.log("renderer:", results.renderer);
console.log("shader compile:", results.compileMs, "ms");
console.table(results.rows);
console.log("hard cut PSNR vs nearest real frame:", results.cutEarly, results.cutLate);
console.log("flow per source frame at 1080p (ms, submit + GPU finish):", results.timing);

const failures = results.rows.filter((row) => row.engine <= row.crossfade || row.engine <= row.repeat);
if (failures.length) console.error("FAIL: not better than baseline:", failures.map((r) => r.name));
if (results.cutEarly < 35 || results.cutLate < 35) console.error("FAIL: hard cut was not clean");
process.exit(failures.length || results.cutEarly < 35 || results.cutLate < 35 ? 1 : 0);
