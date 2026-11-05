// Loads the unpacked extension in a throwaway Edge profile and plays real H.264 video.
import { execFileSync } from "node:child_process";
import { createReadStream, existsSync, mkdirSync, mkdtempSync, rmSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import ffmpeg from "ffmpeg-static";
import { chromium } from "playwright-core";

const here = dirname(fileURLToPath(import.meta.url));
const extension = resolve(here, "..");
const fixtures = join(here, ".fixtures");
const outDir = join(here, ".output");
mkdirSync(fixtures, { recursive: true });
mkdirSync(outDir, { recursive: true });

function makeClip(name, fps, filter) {
  const file = join(fixtures, name);
  if (existsSync(file)) return file;
  execFileSync(ffmpeg, [
    "-y", "-loglevel", "error", "-f", "lavfi", "-i", filter,
    "-t", "20", "-r", String(fps), "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "veryfast",
    "-crf", "20", "-movflags", "+faststart", file,
  ]);
  return file;
}
const clip24 = makeClip("clip24.mp4", 24, "testsrc2=size=1280x720:rate=24");
const clip60 = makeClip("clip60.mp4", 60, "testsrc2=size=1280x720:rate=60");
const sound24 = join(fixtures, "sound24.mp4");
if (!existsSync(sound24)) {
  execFileSync(ffmpeg, [
    "-y", "-loglevel", "error", "-i", clip24, "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
    "-t", "20", "-c:v", "copy", "-c:a", "aac", "-movflags", "+faststart", sound24,
  ]);
}

const pages = {
  "/basic.html": `<!doctype html><body style="margin:0;background:#000"><video id=v src="/clip24.mp4" muted loop autoplay playsinline width=1280 height=720></video>`,
  "/basic-sound.html": `<!doctype html><body style="margin:0;background:#000"><video id=v src="/sound24.mp4" loop autoplay playsinline width=1280 height=720></video>`,
  "/own-audio.html": `<!doctype html><body style="margin:0;background:#000"><video id=v src="/sound24.mp4" loop autoplay playsinline width=1280 height=720></video><script>
    const ctx = new AudioContext();
    ctx.createMediaElementSource(document.getElementById("v")).connect(ctx.destination);
  </script>`,
  "/contain.html": `<!doctype html><body style="margin:0;background:#222"><div style="position:relative;width:1000px;height:800px"><video id=v src="/clip24.mp4" muted loop autoplay style="position:absolute;inset:0;width:100%;height:100%;object-fit:contain"></video></div>`,
  "/high-fps.html": `<!doctype html><video id=v src="/clip60.mp4" muted loop autoplay width=1280 height=720></video>`,
  "/small.html": `<!doctype html><video id=v src="/clip24.mp4" muted loop autoplay width=200 height=112></video>`,
  "/controls.html": `<!doctype html><video id=v src="/clip24.mp4" controls muted loop autoplay width=1280 height=720></video>`,
  "/captions.html": `<!doctype html><body style="margin:0"><video id=v src="/clip24.mp4" muted loop autoplay width=1280 height=720><track kind=subtitles src="/cues.vtt" default></video>`,
  "/captions-letterbox.html": `<!doctype html><body style="margin:0;background:#222"><video id=v src="/clip24.mp4" muted loop autoplay style="width:1000px;height:800px;object-fit:contain"><track kind=subtitles src="/cues.vtt" default></video>`,
  "/captions-shadow.html": `<!doctype html><body style="margin:0"><div id=host></div><script>
    document.getElementById("host").attachShadow({ mode: "open" }).innerHTML =
      '<video id=v src="/clip24.mp4" muted loop autoplay style="width:1000px;height:800px;object-fit:contain"><track kind=subtitles src="/cues.vtt" default></video>';
  </script>`,
  "/cross-origin.html": (crossPort) => `<!doctype html><video id=v src="http://127.0.0.1:${crossPort}/clip24.mp4" muted loop autoplay width=1280 height=720></video>`,
  "/fullscreen.html": `<!doctype html><body style="margin:0"><div id=box style="position:relative;background:#000"><video id=v src="/clip24.mp4" muted loop autoplay style="width:100%;display:block"></video></div>
    <button id=box-fs onclick="document.getElementById('box').requestFullscreen()">box</button>
    <button id=video-fs onclick="document.getElementById('v').requestFullscreen()">video</button>`,
  "/mse.html": `<!doctype html><video id=v muted loop autoplay width=1280 height=720></video><script>
    const v = document.getElementById("v");
    const ms = new MediaSource();
    v.src = URL.createObjectURL(ms);
    ms.addEventListener("sourceopen", async () => {
      const sb = ms.addSourceBuffer('video/mp4; codecs="avc1.640020"');
      sb.mode = "sequence";
      sb.appendBuffer(await (await fetch("/frag24.mp4")).arrayBuffer());
    });
  </script>`,
};
const fragmented = join(fixtures, "frag24.mp4");
if (!existsSync(fragmented)) {
  execFileSync(ffmpeg, ["-y", "-loglevel", "error", "-i", clip24, "-c", "copy", "-movflags", "frag_keyframe+empty_moov+default_base_moof", fragmented]);
}

function serve(crossPort, allowCors = false) {
  return createServer((req, res) => {
    const path = req.url.split("?")[0];
    if (allowCors) res.setHeader("access-control-allow-origin", "*");
    if (pages[path]) {
      const page = pages[path];
      res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
      return res.end(typeof page === "function" ? page(crossPort()) : page);
    }
    if (path === "/cues.vtt") {
      res.writeHead(200, { "content-type": "text/vtt; charset=utf-8" });
      return res.end(
        "WEBVTT\n\n" +
          "00:00.000 --> 00:10.000\n<i>第一句</i> first line\n\n" +
          "00:00.000 --> 00:10.000 line:0 position:10% align:start size:40%\ntop left\n\n" +
          "00:10.000 --> 00:20.000\n第二句 second line\n",
      );
    }
    // Same page behind a strict CSP: the rule hiding native captions must still apply.
    if (path === "/captions-csp.html") {
      res.writeHead(200, { "content-type": "text/html; charset=utf-8", "content-security-policy": "default-src 'self'; style-src 'none'; script-src 'none'" });
      // Attributes, not style="": this CSP would block inline styles and undo the letterbox.
      return res.end(`<!doctype html><video id=v src="/clip24.mp4" muted loop autoplay width=1000 height=800><track kind=subtitles src="/cues.vtt" default></video>`);
    }
    const file = join(fixtures, path.slice(1));
    if (!path.endsWith(".mp4") || !existsSync(file)) {
      res.writeHead(404);
      return res.end();
    }
    const { size } = statSync(file);
    const range = /bytes=(\d+)-(\d*)/.exec(req.headers.range ?? "");
    if (range) {
      const start = Number(range[1]);
      const end = range[2] ? Number(range[2]) : size - 1;
      res.writeHead(206, {
        "content-type": "video/mp4",
        "accept-ranges": "bytes",
        "content-range": `bytes ${start}-${end}/${size}`,
        "content-length": end - start + 1,
      });
      return createReadStream(file, { start, end }).pipe(res);
    }
    res.writeHead(200, { "content-type": "video/mp4", "accept-ranges": "bytes", "content-length": size });
    createReadStream(file).pipe(res);
  });
}

let crossServer;
const server = serve(() => crossServer.address().port);
crossServer = serve(() => 0);
await new Promise((r) => server.listen(0, "127.0.0.1", r));
await new Promise((r) => crossServer.listen(0, "127.0.0.1", r));
const base = `http://127.0.0.1:${server.address().port}`;

const profile = mkdtempSync(join(tmpdir(), "frameboost-e2e-"));
const context = await chromium.launchPersistentContext(profile, {
  channel: "msedge",
  headless: process.env.HEADED !== "1",
  viewport: { width: 1400, height: 900 },
  ignoreDefaultArgs: ["--disable-extensions", "--disable-component-extensions-with-background-pages"],
  args: [
    `--disable-extensions-except=${extension}`,
    `--load-extension=${extension}`,
    "--disable-features=DisableLoadExtensionCommandLineSwitch",
    "--use-angle=d3d11",
    "--enable-gpu",
    "--ignore-gpu-blocklist",
    "--autoplay-policy=no-user-gesture-required",
    ...(process.env.UNCAPPED === "1" ? ["--disable-gpu-vsync", "--disable-frame-rate-limit"] : []),
  ],
});

const failures = [];
function check(name, ok, detail = "") {
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? `  (${detail})` : ""}`);
  if (!ok) failures.push(name);
}

try {
  // Edge runs its own component extensions too, so pick ours by manifest name.
  let worker = null;
  const deadline = Date.now() + 15000;
  while (!worker && Date.now() < deadline) {
    for (const candidate of context.serviceWorkers()) {
      const name = await candidate.evaluate(() => globalThis.chrome?.runtime?.getManifest?.().name).catch(() => null);
      if (name?.startsWith("FrameBoost")) worker = candidate;
    }
    if (!worker) await new Promise((r) => setTimeout(r, 250));
  }
  if (!worker) throw new Error("extension service worker did not start");

  const statusOf = (page) =>
    worker.evaluate(async (url) => {
      const [tab] = await chrome.tabs.query({ url });
      const results = await chrome.scripting.executeScript({
        target: { tabId: tab.id, allFrames: true },
        func: () => globalThis.__frameBoost?.status() ?? null,
      });
      return results.map((r) => r.result).find(Boolean) ?? null;
    }, page.url());
  const setSettings = (values) => worker.evaluate((v) => chrome.storage.sync.set(v), values);
  const open = async (path) => {
    const page = await context.newPage();
    page.on("pageerror", (error) => check(`no page errors on ${path}`, false, error.message));
    await page.goto(base + path);
    await page.bringToFront();
    return page;
  };
  const settled = async (page, state, ms = 12000) => {
    const deadline = Date.now() + ms;
    let status;
    while (Date.now() < deadline) {
      status = await statusOf(page);
      if (status?.state === state && (state !== "running" || status.outputFps > 0)) return status;
      await page.waitForTimeout(250);
    }
    return status;
  };
  // Counts distinct pictures that actually reach the canvas, from inside the frame they drew in.
  const distinctFps = (page, seconds = 3) =>
    page.evaluate(
      (seconds) =>
        new Promise((resolve) => {
          const canvas = document.querySelector("canvas[data-frameboost]");
          const probe = new OffscreenCanvas(64, 36);
          const g = probe.getContext("2d", { willReadFrequently: true });
          let last = null;
          let distinct = 0;
          let ticks = 0;
          const start = performance.now();
          const tick = () => {
            ticks++;
            g.drawImage(canvas, 0, 0, 64, 36);
            const data = g.getImageData(0, 0, 64, 36).data;
            let hash = 0;
            for (let i = 0; i < data.length; i += 4) hash = (hash * 31 + data[i] + data[i + 1] * 3) | 0;
            if (hash !== last) distinct++;
            last = hash;
            const elapsed = performance.now() - start;
            if (elapsed > seconds * 1000) resolve({ distinct: distinct / (elapsed / 1000), raf: ticks / (elapsed / 1000) });
            else requestAnimationFrame(tick);
          };
          requestAnimationFrame(tick);
        }),
      seconds,
    );

  await setSettings({ quality: "medium" });

  // --- Basic playback ---
  let page = await open("/basic.html");
  let status = await settled(page, "running");
  check("24 fps file interpolates", status?.state === "running", JSON.stringify(status));
  check("source rate detected as 24", Math.abs((status?.sourceFps ?? 0) - 24) < 1.5, `source ${status?.sourceFps}`);
  // Measured before the probes below, whose readback and screenshots stall the GPU.
  const metrics = await page.evaluate(async () => {
    const video = document.getElementById("v");
    const before = video.getVideoPlaybackQuality();
    await new Promise((r) => setTimeout(r, 4000));
    const after = video.getVideoPlaybackQuality();
    return { dropped: after.droppedVideoFrames - before.droppedVideoFrames, total: after.totalVideoFrames - before.totalVideoFrames };
  });
  check("video element not starved by overlay", metrics.dropped <= 5 && metrics.total > 85, JSON.stringify(metrics));
  const measured = await distinctFps(page);
  console.log(`      distinct pictures ${measured.distinct.toFixed(1)}/s at ${measured.raf.toFixed(1)} rAF/s, refresh ${status?.refreshRate} Hz`);
  check("distinct output well above source rate", measured.distinct > 45, measured.distinct.toFixed(1));
  await page.screenshot({ path: join(outDir, "basic.png") });


  await setSettings({ targetFps: 60 });
  await page.waitForTimeout(1500);
  status = await statusOf(page);
  if (status.refreshRate > 70) {
    const expected = status.refreshRate / Math.floor(status.refreshRate / 60 + 0.05);
    check("target 60 uses an even divisor on a fast display", Math.abs(status.outputFps - expected) < 6, `output ${status.outputFps}, expected ${expected.toFixed(0)}`);
  }
  await setSettings({ targetFps: 120 });

  // Pause hands the screen back to the real frame.
  await page.evaluate(() => document.getElementById("v").pause());
  await page.waitForTimeout(300);
  check("pause hides overlay", (await page.evaluate(() => getComputedStyle(document.querySelector("canvas[data-frameboost]")).display)) === "none");
  await page.evaluate(() => document.getElementById("v").play());
  status = await settled(page, "running");
  check("resume interpolates again", status?.state === "running");

  // Seek must flush: no synthesis across the jump, and it recovers.
  await page.evaluate(() => (document.getElementById("v").currentTime = 12));
  status = await settled(page, "running");
  check("recovers after seek", status?.state === "running");

  // Toggle off and on.
  await setSettings({ enabled: false });
  status = await settled(page, "off", 3000);
  check("disable shows real video", status?.state === "off");
  await setSettings({ enabled: true });
  status = await settled(page, "running");
  check("enable resumes", status?.state === "running");

  // Quality change reallocates cleanly while playing.
  await setSettings({ quality: "high" });
  status = await settled(page, "running");
  check("quality change while playing", status?.state === "running");
  await setSettings({ quality: "medium" });
  await page.close();

  // --- Layout ---
  page = await open("/contain.html");
  status = await settled(page, "running");
  const letterbox = await page.evaluate(() => {
    const video = document.getElementById("v").getBoundingClientRect();
    const canvas = document.querySelector("canvas[data-frameboost]").getBoundingClientRect();
    return { match: Math.abs(video.left - canvas.left) < 1 && Math.abs(video.top - canvas.top) < 1 && Math.abs(video.width - canvas.width) < 1 && Math.abs(video.height - canvas.height) < 1 };
  });
  check("overlay matches letterboxed video box", letterbox.match);
  await page.screenshot({ path: join(outDir, "contain.png") });
  await page.close();

  // --- Fullscreen: containers keep working, a bare <video> falls back to the real picture ---
  page = await open("/fullscreen.html");
  await settled(page, "running");
  await page.evaluate(() => document.getElementById("v").scrollIntoView());
  await page.click("#box-fs");
  await page.waitForTimeout(500);
  status = await settled(page, "running");
  const boxFull = await page.evaluate(() => {
    const canvas = document.querySelector("canvas[data-frameboost]");
    return { full: document.fullscreenElement?.id, inside: document.fullscreenElement?.contains(canvas), width: canvas.getBoundingClientRect().width, screen: innerWidth };
  });
  check("container fullscreen keeps interpolating", status?.state === "running" && boxFull.inside && Math.abs(boxFull.width - boxFull.screen) < 2, JSON.stringify(boxFull));
  await page.evaluate(() => document.exitFullscreen());
  await page.waitForTimeout(300);
  await page.click("#video-fs");
  status = await settled(page, "fullscreen", 3000);
  check("native video fullscreen falls back", status?.state === "fullscreen", status?.state);
  await page.evaluate(() => document.exitFullscreen());
  status = await settled(page, "running");
  check("leaving native fullscreen resumes", status?.state === "running", status?.state);

  // Losing the GPU context (driver reset, sleep) must show the real video, then recover.
  const lostOk = await page.evaluate(async () => {
    const extension = document.querySelector("canvas[data-frameboost]").getContext("webgl2").getExtension("WEBGL_lose_context");
    extension.loseContext();
    await new Promise((r) => setTimeout(r, 400));
    const hidden = getComputedStyle(document.querySelector("canvas[data-frameboost]")).display === "none";
    extension.restoreContext();
    return hidden;
  });
  check("context loss hides overlay", lostOk);
  status = await settled(page, "running");
  check("context restore resumes", status?.state === "running", status?.message);
  await page.close();

  // --- Protected media must be left completely alone ---
  page = await open("/basic.html");
  await settled(page, "running");
  await page.evaluate(async () => {
    const config = [{ initDataTypes: ["cenc"], videoCapabilities: [{ contentType: 'video/mp4; codecs="avc1.640020"' }] }];
    const access = await navigator.requestMediaKeySystemAccess("org.w3.clearkey", config);
    await document.getElementById("v").setMediaKeys(await access.createMediaKeys());
  });
  status = await settled(page, "drm", 3000);
  check("EME video reports DRM and stays untouched", status?.state === "drm", status?.message);
  const drmHidden = await page.evaluate(() => getComputedStyle(document.querySelector("canvas[data-frameboost]")).display === "none");
  check("no overlay over protected video", drmHidden);
  await page.close();

  // --- MSE, the delivery used by most streaming sites ---
  page = await open("/mse.html");
  status = await settled(page, "running");
  check("MSE (blob URL) video interpolates", status?.state === "running", status?.message);
  await page.close();

  // --- States that must not interpolate ---
  page = await open("/high-fps.html");
  await page.waitForTimeout(2500);
  status = await statusOf(page);
  if (status.refreshRate <= 65) check("60 fps source passes through on 60 Hz", status?.state === "passthrough", status?.state);
  else console.log(`      60 fps source on ${status.refreshRate} Hz: ${status.state} ${status.outputFps} fps`);
  await page.close();

  page = await open("/small.html");
  await page.waitForTimeout(2500);
  status = await statusOf(page);
  check("small preview video ignored", status?.state === "idle", status?.state);
  await page.close();

  // --- Native captions: interpolation keeps running and exactly one copy stays visible ---
  const captionState = (p) =>
    p.evaluate(() => {
      const box = document.querySelector("[data-frameboost-captions]");
      const video = document.querySelector("video");
      return {
        shown: !!box && getComputedStyle(box).display !== "none",
        text: box?.innerText ?? "",
        italic: !!box?.querySelector("i"),
        nativeHidden: video.hasAttribute("data-frameboost-hide-native-captions"),
      };
    });
  // Edge draws native cues across the whole video box, so in letterboxed video they would show
  // through the transparent bars of the canvas. Compare the top bar with captions fully off.
  const nativeLeaks = async (p, videoIn) => {
    const clip = { x: 0, y: 0, width: 1000, height: 100 };
    await videoIn(p, (v) => (v.currentTime = 1));
    await settled(p, "running");
    await videoIn(p, (v) => (v.textTracks[0].mode = "disabled"));
    await p.waitForTimeout(300);
    const clean = await p.screenshot({ clip });
    await videoIn(p, (v) => (v.textTracks[0].mode = "showing"));
    await p.waitForTimeout(300);
    const withOverlay = await p.screenshot({ clip });
    await videoIn(p, (v) => {
      const box = v.getRootNode().querySelector("[data-frameboost-captions]");
      box.style.visibility = "hidden";
    });
    await p.waitForTimeout(300);
    const overlayOff = await p.screenshot({ clip });
    await videoIn(p, (v) => {
      v.getRootNode().querySelector("[data-frameboost-captions]").style.visibility = "";
    });
    return { overlayDraws: !withOverlay.equals(clean), nativeLeaks: !overlayOff.equals(clean), clean, clip };
  };
  const inDocument = (p, fn) => p.evaluate(`(${fn})(document.getElementById("v"))`);
  const inShadow = (p, fn) => p.evaluate(`(${fn})(document.getElementById("host").shadowRoot.querySelector("video"))`);

  page = await open("/captions.html");
  status = await settled(page, "running");
  check("native captions no longer stop interpolation", status?.state === "running", status?.state);
  await page.evaluate(() => (document.getElementById("v").currentTime = 1));
  await settled(page, "running");
  await page.waitForTimeout(300);
  let captions = await captionState(page);
  check("overlay shows the active cues", captions.shown && captions.text.includes("第一句") && captions.text.includes("top left"), JSON.stringify(captions));
  check("cue markup survives", captions.italic);
  const layoutOfCues = await page.evaluate(() => {
    const box = document.querySelector("[data-frameboost-captions]");
    const video = document.querySelector("video").getBoundingClientRect();
    const span = (text) => [...box.querySelectorAll("span")].find((s) => s.innerText.includes(text)).getBoundingClientRect();
    const bottom = span("first line");
    const top = span("top left");
    return {
      bottomCueNearBottom: bottom.bottom > video.top + video.height * 0.8,
      bottomCueCentered: Math.abs((bottom.left + bottom.right) / 2 - video.width / 2) < 4,
      topCueAtTop: top.top < video.top + video.height * 0.1,
      topCueLeft: top.left < video.width * 0.2,
      aboveCanvas: document.elementsFromPoint(0, 0).length >= 0 && box.compareDocumentPosition(document.querySelector("canvas[data-frameboost]")) === Node.DOCUMENT_POSITION_PRECEDING,
    };
  });
  check("cue placement follows WebVTT settings", Object.values(layoutOfCues).every(Boolean), JSON.stringify(layoutOfCues));
  await page.screenshot({ path: join(outDir, "captions.png") });

  await page.evaluate(() => (document.getElementById("v").currentTime = 11));
  await settled(page, "running");
  await page.waitForTimeout(300);
  captions = await captionState(page);
  check("overlay follows cue changes", captions.text.includes("第二句") && !captions.text.includes("第一句"), captions.text);

  await page.evaluate(() => document.getElementById("v").pause());
  await page.waitForTimeout(300);
  captions = await captionState(page);
  check("pause hands captions back to the browser", !captions.shown && !captions.nativeHidden, JSON.stringify(captions));
  await page.screenshot({ path: join(outDir, "captions-paused.png") });
  await page.evaluate(() => document.getElementById("v").play());
  await settled(page, "running");

  await page.evaluate(() => (document.getElementById("v").textTracks[0].mode = "disabled"));
  await page.waitForTimeout(300);
  captions = await captionState(page);
  check("turning captions off removes them", !captions.shown && !captions.nativeHidden, JSON.stringify(captions));
  await page.evaluate(() => (document.getElementById("v").textTracks[0].mode = "showing"));
  await page.waitForTimeout(300);
  captions = await captionState(page);
  check("turning captions on shows them", captions.shown && captions.nativeHidden, JSON.stringify(captions));

  await setSettings({ enabled: false });
  await settled(page, "off", 3000);
  captions = await captionState(page);
  check("disabling the extension restores native captions", !captions.nativeHidden && !captions.shown, JSON.stringify(captions));
  await setSettings({ enabled: true });
  await page.close();

  page = await open("/captions-letterbox.html");
  await settled(page, "running");
  await page.evaluate(() => (document.getElementById("v").currentTime = 1));
  await settled(page, "running");
  await page.waitForTimeout(300);
  await page.screenshot({ path: join(outDir, "captions-letterbox.png") });
  let leak = await nativeLeaks(page, inDocument);
  check("exactly one copy of captions in letterbox bars", leak.overlayDraws && !leak.nativeLeaks, JSON.stringify({ overlayDraws: leak.overlayDraws, nativeLeaks: leak.nativeLeaks }));
  await inDocument(page, (v) => v.pause());
  await page.waitForTimeout(300);
  const nativeBack = await page.screenshot({ clip: leak.clip });
  check("native captions return in the bars on pause", !nativeBack.equals(leak.clean));
  await page.close();

  page = await open("/captions-csp.html");
  status = await settled(page, "running");
  leak = await nativeLeaks(page, inDocument);
  check("native copy hidden under strict CSP", status?.state === "running" && !leak.nativeLeaks && leak.overlayDraws, JSON.stringify({ state: status?.state, overlayDraws: leak.overlayDraws, nativeLeaks: leak.nativeLeaks }));
  await page.close();

  page = await open("/captions-shadow.html");
  status = await settled(page, "running");
  leak = await nativeLeaks(page, inShadow);
  check("captions work inside shadow DOM", status?.state === "running" && !leak.nativeLeaks && leak.overlayDraws, JSON.stringify({ state: status?.state, overlayDraws: leak.overlayDraws, nativeLeaks: leak.nativeLeaks }));
  await page.close();

  // --- Cross-origin file: blocked first, works after the explicit reload ---
  page = await open("/cross-origin.html");
  status = await settled(page, "tainted");
  check("cross-origin video reports tainted", status?.state === "tainted", status?.state);
  check("cross-origin offers reload", status?.canReloadCors === true);
  await worker.evaluate(async (url) => {
    const [tab] = await chrome.tabs.query({ url });
    await chrome.scripting.executeScript({ target: { tabId: tab.id }, func: () => globalThis.__frameBoost.reloadWithCors() });
  }, page.url());
  status = await settled(page, "running");
  check("cross-origin works after reload", status?.state === "running", status?.message);
  await page.close();

  // --- Popup renders the same status ---
  page = await open("/basic.html");
  await settled(page, "running");
  const extensionId = worker.url().split("/")[2];
  const videoTabId = await worker.evaluate(async (url) => (await chrome.tabs.query({ url }))[0].id, page.url());
  // The popup as a tab, pointed at the video tab, since a popup opened as a tab is its own active tab.
  const popupEngine = async (videoPage) => {
    const tabId = await worker.evaluate(async (url) => (await chrome.tabs.query({ url }))[0].id, videoPage.url());
    const popup = await context.newPage();
    await popup.goto(`chrome-extension://${extensionId}/popup.html`);
    await popup.evaluate((id) => {
      chrome.tabs.query = async () => [{ id }];
    }, tabId);
    await popup.waitForTimeout(2500);
    const engine = await popup.evaluate(() => {
      const line = document.getElementById("engine");
      return { hidden: line.hidden, kind: line.dataset.kind, text: line.textContent, note: document.getElementById("engine-note").textContent };
    });
    await popup.close();
    return engine;
  };
  for (const scheme of ["light", "dark"]) {
    const popup = await context.newPage();
    await popup.emulateMedia({ colorScheme: scheme });
    await popup.setViewportSize({ width: 320, height: 500 });
    // A popup opened as a tab would be its own active tab, so point it at the video tab.
    await popup.goto(`chrome-extension://${extensionId}/popup.html`);
    await popup.evaluate((id) => {
      chrome.tabs.query = async () => [{ id }];
    }, videoTabId);
    await page.bringToFront();
    await popup.waitForFunction(() => document.getElementById("rate").textContent.includes("→"), null, { timeout: 8000 }).catch(() => {});
    check(`popup shows live rate (${scheme})`, (await popup.locator("#rate").textContent()).includes("→"), await popup.locator("#rate").textContent());
    check(`popup loads settings (${scheme})`, await popup.locator('input[name=targetFps][value="120"]').isChecked());
    const flowLine = await popup.locator("#engine").textContent();
    check(`popup names the running engine (${scheme})`, await popup.locator("#engine").isVisible() && flowLine.includes("补帧"), flowLine);
    await popup.bringToFront();
    await popup.screenshot({ path: join(outDir, `popup-${scheme}.png`) });
    await popup.close();
  }
  await page.close();

  // --- AI engine, held on even if the GPU is busy ---
  let gpuUse = 0;
  try {
    gpuUse = Number(execFileSync("nvidia-smi", ["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]).toString().trim().split("\n")[0]);
  } catch {}
  const gpuBusy = gpuUse > 50;
  // A size that fits the budget: pacing is what is judged here, not how large a model can run.
  await worker.evaluate(() => chrome.storage.local.set({ modelHeight: 512 }));
  await setSettings({ quality: "neural", keepModel: true });
  page = await open("/basic-sound.html");
  status = await settled(page, "running", 20000);
  check("AI quality starts", status?.state === "running" && status?.engine === "neural", JSON.stringify(status));
  let engineLine = await popupEngine(page);
  check("popup says AI is running", engineLine.kind === "ai" && engineLine.text.includes("AI 补帧生效中"), JSON.stringify(engineLine));
  const audioThrough = await worker.evaluate(async (url) => {
    const [tab] = await chrome.tabs.query({ url });
    const [{ result }] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: async () => {
        const node = globalThis.__frameBoost.audioNode();
        if (!node) return { routed: false };
        const analyser = node.context.createAnalyser();
        node.connect(analyser);
        const data = new Float32Array(analyser.fftSize);
        let peak = 0;
        for (let i = 0; i < 20; i++) {
          await new Promise((r) => setTimeout(r, 50));
          analyser.getFloatTimeDomainData(data);
          for (const v of data) peak = Math.max(peak, Math.abs(v));
        }
        return { routed: true, peak, delay: node.delayTime.value, context: node.context.state };
      },
    });
    return result;
  }, page.url());
  check("sound keeps playing through the delay line", audioThrough.routed && audioThrough.peak > 0.01 && audioThrough.context === "running", JSON.stringify(audioThrough));
  status = await statusOf(page);
  console.log(`      engine ${status.engine}, ${status.model || "no model"}, skipped ${status.stats?.skipRate?.toFixed(2)}, audio delay ${status.audioDelay}`);
  check("audio is held back with the picture", Math.abs(audioThrough.delay - status.audioDelay) < 0.02 && status.audioDelay > 0.1, JSON.stringify(audioThrough));
  const output = await distinctFps(page);
  console.log(`      AI: ${output.distinct.toFixed(1)} distinct pictures/s at ${output.raf.toFixed(1)} rAF/s`);
  status = await statusOf(page);
  // A game holding the GPU starves every client, the browser's own compositor included, and the
  // adaptive path below handles that. Pacing can only be judged on a GPU left to this test.
  if (gpuBusy) {
    console.log(`      GPU busy before this section (${gpuUse}%): pacing not judged, output ${status?.outputFps} fps`);
  } else {
    check("network does not block the picture", output.raf > 60, `${output.raf.toFixed(1)} rAF/s`);
    check("AI output keeps its rate", status?.outputFps > 60, `${status?.outputFps} fps`);
    check("AI interpolates on a free GPU", output.distinct > 45, `${output.distinct.toFixed(1)}/s, ${status?.model}`);
  }
  await page.screenshot({ path: join(outDir, "neural.png") });
  await page.evaluate(() => document.getElementById("v").pause());
  status = await settled(page, "holding", 3000);
  check("pause holds the delayed picture", status?.state === "holding", status?.state);
  await page.evaluate(() => document.getElementById("v").play());
  status = await settled(page, "running", 15000);
  check("resume after hold", status?.state === "running", status?.state);
  await page.evaluate(() => (document.getElementById("v").currentTime = 8));
  status = await settled(page, "running", 15000);
  check("AI recovers after seek", status?.state === "running", status?.state);
  await page.close();

  // Adaptive from here: it runs, or has moved to the classic engine and says why.
  await setSettings({ keepModel: false });
  page = await open("/basic-sound.html");
  status = await settled(page, "running", 20000);
  await page.waitForTimeout(6000);
  status = await settled(page, "running", 10000);
  console.log(`      adaptive: engine ${status?.engine}, ${status?.model || "no model"}, note "${status?.note}"`);
  check(
    "busy GPU settles on a model size or the classic engine",
    status?.state === "running" && (status.engine === "neural" ? status.audioDelay > 0.1 : status.note.length > 0 && status.audioDelay === 0),
    JSON.stringify(status),
  );
  await setSettings({ quality: "medium" });
  status = await settled(page, "running", 8000);
  check("leaving AI quality uses the classic engine", status?.engine === "flow" && status?.audioDelay === 0, JSON.stringify(status));
  await setSettings({ quality: "neural" });
  await page.waitForTimeout(500);
  status = await statusOf(page);
  check("returning to AI quality tries it again", status?.engine === "neural" && !status?.note, JSON.stringify(status));
  await page.close();

  // A page that routes the sound itself leaves no way to delay it: AI stands down and says so.
  page = await open("/own-audio.html");
  status = await settled(page, "running", 20000);
  check("page-owned audio falls back to the classic engine", status?.engine === "flow" && status?.note.length > 0, JSON.stringify(status));
  const toast = await page.locator("[data-frameboost-toast]").textContent({ timeout: 3000 }).catch(() => "");
  check("the video says AI is not running", toast.includes("AI 补帧暂不可用"), toast);
  engineLine = await popupEngine(page);
  check("popup says AI is not running and why", engineLine.kind === "fallback" && engineLine.note.includes("原因"), JSON.stringify(engineLine));
  await setSettings({ quality: "medium" });
  await page.close();
} finally {
  await context.close();
  server.close();
  crossServer.close();
  rmSync(profile, { recursive: true, force: true });
}

if (failures.length) {
  console.error(`\n${failures.length} failed:`, failures);
  process.exit(1);
}
console.log("\nall passed");
process.exit(0);
