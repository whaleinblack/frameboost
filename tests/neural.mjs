// Neural engine on its own: ground truth at the middle, real frames at the ends, clean hard cuts.
import { createReadStream, readFileSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");
const server = createServer((req, res) => {
  const url = req.url.split("?")[0];
  if (url === "/") {
    res.writeHead(200, { "content-type": "text/html" });
    return res.end("<!doctype html><canvas id=out width=1280 height=720></canvas>");
  }
  if (!url.startsWith("/models/")) {
    res.writeHead(404);
    return res.end();
  }
  const file = join(root, url);
  res.writeHead(200, { "content-length": statSync(file).size });
  createReadStream(file).pipe(res);
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const browser = await chromium.launch({ channel: "msedge", headless: true, args: ["--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist"] });
const page = await browser.newPage();
page.on("pageerror", (error) => console.error("page error:", error.message));
await page.goto(`http://127.0.0.1:${server.address().port}/`);
await page.addScriptTag({ content: readFileSync(join(root, "src/rife.js"), "utf8") });
await page.addScriptTag({ content: readFileSync(join(root, "src/repair.js"), "utf8") });
await page.addScriptTag({ content: readFileSync(join(root, "src/neural.js"), "utf8") });

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
  // Neural networks fall on these: shrunk, the stripes turn grey, and the inbetweens come out inverted.
  const stripes = (period, speed) => (g, t) => {
    g.fillStyle = "#101010";
    g.fillRect(0, 0, W, H);
    g.fillStyle = "#f0f0f0";
    for (let x = -period * 2; x < W + period * 2; x += period) g.fillRect(x + (t * speed) % period, 0, period / 2, H);
  };
  const scenes = {
    "中速平移 16px": (g, t) => g.drawImage(background, -300 - t * 16.3, -200 - t * 5.2),
    "前景物体 + 平移背景": (g, t) => {
      g.drawImage(background, -300 - t * 12.3, -200 + t * 3.6);
      g.drawImage(sprite, 150 + t * 24.5, 150 - t * 8.5);
    },
    "黑白条纹 周期 16 / 5px": stripes(16, 5.3),
    "黑白条纹 周期 8 / 3px": stripes(8, 3.1),
  };
  const draw = (scene, t) => {
    const c = new OffscreenCanvas(W, H);
    const g = c.getContext("2d", { willReadFrequently: true });
    g.filter = "blur(0.6px)";
    scene(g, t);
    return c;
  };
  const pixels = (c) => c.getContext("2d").getImageData(0, 0, W, H).data;
  const psnr = (a, b) => {
    let sum = 0;
    for (let i = 0; i < a.length; i += 4) {
      for (let k = 0; k < 3; k++) sum += (a[i + k] - b[i + k]) ** 2;
    }
    return +(10 * Math.log10((255 * 255) / (sum / ((a.length / 4) * 3)))).toFixed(2);
  };

  const canvas = document.getElementById("out");
  const engine = new FrameBoostNeural.NeuralInterpolator(canvas, { modelFiles: { bin: "/models/rife49.bin", json: "/models/rife49.json" } });
  await engine.loaded;
  const viewport = { x: 0, y: 0, width: W, height: H };
  const crop = { x: 0, y: 0, width: 1, height: 1 };
  const shot = (time) => {
    engine.render(time, viewport, crop);
    const c = new OffscreenCanvas(W, H);
    const g = c.getContext("2d", { willReadFrequently: true });
    g.drawImage(canvas, 0, 0);
    return g.getImageData(0, 0, W, H).data;
  };
  const settle = async () => {
    while (!engine.frames[engine.frames.length - 1].pair?.ready) {
      engine.pump();
      await new Promise((r) => setTimeout(r, 2));
    }
  };

  const rows = [];
  const ends = [];
  for (const [name, scene] of Object.entries(scenes)) {
    const a = draw(scene, 0);
    const b = draw(scene, 1);
    const mid = pixels(draw(scene, 0.5));
    const quarter = pixels(draw(scene, 0.25));
    engine.reset();
    engine.pushFrame(a, W, H, 0);
    engine.pushFrame(b, W, H, 1);
    await settle();
    const pa = pixels(a);
    const pb = pixels(b);
    const blend = new Uint8ClampedArray(pa.length);
    for (let i = 0; i < pa.length; i++) blend[i] = (pa[i] + pb[i]) / 2;
    const quarterBlend = new Uint8ClampedArray(pa.length);
    for (let i = 0; i < pa.length; i++) quarterBlend[i] = pa[i] * 0.75 + pb[i] * 0.25;
    rows.push({
      name,
      "t=0.5": psnr(shot(0.5), mid),
      "t=0.5 淡化": psnr(blend, mid),
      "t=0.25": psnr(shot(0.25), quarter),
      "t=0.25 淡化": psnr(quarterBlend, quarter),
    });
    ends.push({ name, start: psnr(shot(0), pa), end: psnr(shot(1), pb), nearStart: psnr(shot(0.001), pa), nearEnd: psnr(shot(0.999), pb) });
  }

  const cutA = draw(scenes["中速平移 16px"], 0);
  const cutB = new OffscreenCanvas(W, H);
  cutB.getContext("2d", { willReadFrequently: true }).drawImage(sprite, 0, 0, W, H);
  engine.reset();
  engine.pushFrame(cutA, W, H, 0);
  engine.pushFrame(cutB, W, H, 1);
  await settle();
  const pair = engine.frames[1].pair;
  const cut = { detected: pair.cut, residual: +pair.residual.toFixed(3), early: psnr(shot(0.25), pixels(cutA)), late: psnr(shot(0.75), pixels(cutB)) };
  return { rows, ends, cut, stats: engine.stats() };
});
await browser.close();
server.close();

console.table(results.rows);
console.table(results.ends);
console.log("hard cut:", results.cut);
console.log("stats:", results.stats);

const failures = [];
for (const row of results.rows) {
  if (row["t=0.5"] <= row["t=0.5 淡化"] || row["t=0.25"] <= row["t=0.25 淡化"]) failures.push(`not better than a cross-fade: ${row.name}`);
  if (row.name.includes("条纹") && row["t=0.5"] < 18) failures.push(`stripes not repaired: ${row.name}`);
}
for (const row of results.ends) {
  if (row.start < 45 || row.end < 45) failures.push(`real frames not exact at the ends: ${row.name}`);
  if (row.nearStart < 35 || row.nearEnd < 35) failures.push(`picture jumps next to a real frame: ${row.name}`);
}
if (!results.cut.detected || results.cut.early < 35 || results.cut.late < 35) failures.push("hard cut was not clean");
if (failures.length) console.error("FAIL:", failures);
process.exit(failures.length ? 1 : 0);
