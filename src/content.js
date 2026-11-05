// FrameBoost page integration: finds the main video, overlays a canvas, keeps the clock.
(() => {
  "use strict";
  if (globalThis.__frameBoost) return;

  const DEFAULTS = { enabled: true, targetFps: 120, quality: "neural", hud: false, keepModel: false };
  const MIN_WIDTH = 400;
  const STATE_TEXT = {
    idle: "没有检测到正在播放的视频",
    off: "补帧已关闭",
    running: "补帧中",
    paused: "视频暂停或缓冲中",
    drm: "这个视频受 DRM 版权保护，浏览器不允许读取画面，无法补帧",
    tainted: "视频来自其他域名，浏览器禁止读取画面。可重新加载视频来解除限制",
    corsFailed: "视频服务器拒绝了跨域读取，无法补帧",
    passthrough: "视频帧率已达到目标，无需补帧",
    fullscreen: "视频元素处于原生全屏，无法叠加画面，已暂停补帧",
    controls: "正在显示视频自带控制条，补帧暂时让开",
    error: "GPU 初始化失败",
    loading: "正在加载 AI 模型",
    audioWait: "点一下页面后开始 AI 补帧：需要同步延后声音，浏览器要求先有用户操作",
    holding: "视频暂停或缓冲中",
  };
  const RETRY_NEURAL_MS = 60000;

  let settings = { ...DEFAULTS };
  let session = null;
  let rememberedModelHeight;

  class VideoSession {
    constructor(video) {
      this.video = video;
      this.state = "idle";
      this.detail = "";
      this.deltas = [];
      this.lastMediaTime = null;
      this.clockOffset = null;
      this.refreshRate = 60;
      this.rafDeltas = [];
      this.lastRaf = 0;
      this.outputFps = 0;
      this.drawCount = 0;
      this.drawWindowStart = performance.now();
      this.layout = null;
      this.revealedUntil = 0;
      this.alive = true;
      this.notified = new Set();
      this.fellBack = false;

      this.captions = new FrameBoostCaptions.CaptionOverlay(video);
      this.createEngine();

      this.onDiscontinuity = () => this.flush();
      this.onFullscreen = () => this.syncLayout();
      this.onPointer = (event) => this.handlePointer(event);
      video.addEventListener("seeking", this.onDiscontinuity);
      video.addEventListener("emptied", this.onDiscontinuity);
      video.addEventListener("ratechange", this.onDiscontinuity);
      document.addEventListener("fullscreenchange", this.onFullscreen);
      document.addEventListener("pointermove", this.onPointer, { passive: true, capture: true });
      this.resizeObserver = new ResizeObserver(() => this.syncLayout());
      this.resizeObserver.observe(video);
      this.layoutTimer = setInterval(() => this.syncLayout(), 500);

      this.syncLayout();
      this.frameHandle = video.requestVideoFrameCallback((now, meta) => this.onVideoFrame(now, meta));
      this.rafHandle = requestAnimationFrame((now) => this.onAnimationFrame(now));
    }

    // A canvas keeps its first context type for life, so every engine gets a new one.
    createCanvas() {
      this.canvas?.remove();
      const canvas = document.createElement("canvas");
      canvas.dataset.frameboost = "";
      canvas.setAttribute("aria-hidden", "true");
      canvas.style.setProperty("display", "none", "important");
      canvas.style.setProperty("pointer-events", "none", "important");
      canvas.style.setProperty("position", "absolute", "important");
      canvas.style.setProperty("margin", "0", "important");
      canvas.style.setProperty("padding", "0", "important");
      canvas.style.setProperty("border", "0", "important");
      canvas.style.setProperty("max-width", "none", "important");
      canvas.style.setProperty("max-height", "none", "important");
      // Replacing an engine also loses its old context; only the current canvas counts.
      canvas.addEventListener("webglcontextlost", (event) => {
        event.preventDefault();
        if (canvas !== this.canvas) return;
        this.engine = null;
        this.fail("GPU 上下文丢失，正在恢复");
      });
      canvas.addEventListener("webglcontextrestored", () => {
        if (canvas === this.canvas) this.createEngine();
      });
      this.canvas = canvas;
    }

    get neural() {
      return this.engine instanceof FrameBoostNeural.NeuralInterpolator;
    }

    createEngine() {
      this.engine?.dispose();
      this.createCanvas();
      this.state = "idle";
      this.detail = "";
      try {
        if (settings.quality === "neural" && navigator.gpu && !this.neuralBlocked) {
          const engine = new FrameBoostNeural.NeuralInterpolator(this.canvas, {
            modelFiles: { bin: chrome.runtime.getURL("models/rife49.bin"), json: chrome.runtime.getURL("models/rife49.json") },
            modelHeight: rememberedModelHeight,
            adaptive: !settings.keepModel,
            onLost: (message) => this.dropNeural(message),
          });
          engine.loaded.catch((error) => {
            if (this.engine === engine) this.dropNeural(error.message);
          });
          this.engine = engine;
          this.audio = FrameBoostAudio.forVideo(this.video);
        } else {
          if (settings.quality === "neural" && !navigator.gpu && !this.note) {
            this.note = isSecureContext
              ? "浏览器没有提供 WebGPU。可在 edge://gpu 或 chrome://gpu 查看 WebGPU 一项的状态"
              : "这个页面不是 HTTPS，浏览器不给它 WebGPU";
            this.announceFallback();
          }
          const quality = settings.quality === "neural" ? "high" : settings.quality;
          this.engine = new FrameBoostEngine.FlowInterpolator(this.canvas, { quality });
        }
        this.flush();
        this.syncLayout();
      } catch (error) {
        this.engine = null;
        this.fail(error.message);
      }
    }

    // Permanent reasons stay until reload. Overload can pass, so that one is tried again later.
    dropNeural(reason, retry = false) {
      this.neuralBlocked = true;
      this.note = reason;
      this.retryNeuralAt = retry ? performance.now() + RETRY_NEURAL_MS : 0;
      this.audio?.set(0);
      this.announceFallback();
      this.createEngine();
    }

    // The quality setting still reads AI after a fallback, so the switch is said on the video itself.
    // Each message once per video: a busy GPU retries every minute and would otherwise repeat it.
    announceFallback() {
      this.fellBack = true;
      this.notify(`FrameBoost：AI 补帧暂不可用，已换用光流补帧。${this.note}`);
    }

    notify(text) {
      if (!settings.enabled || this.notified.has(text)) return;
      this.notified.add(text);
      this.toast?.remove();
      const toast = document.createElement("div");
      toast.dataset.frameboostToast = "";
      toast.setAttribute("role", "status");
      toast.style.cssText =
        "position:absolute;pointer-events:none;z-index:2147483647;max-width:min(480px,80%);padding:8px 12px;border-radius:8px;" +
        "background:rgba(8,17,31,.86);color:#e5edf7;font:14px/1.5 'Segoe UI','Microsoft YaHei',sans-serif";
      toast.textContent = text;
      // Before the first layout the canvas is not in the page yet.
      (this.canvas.isConnected ? this.canvas : this.video).after(toast);
      const { video } = this;
      toast.style.left = `${video.offsetLeft + video.offsetWidth - 12}px`;
      toast.style.top = `${video.offsetTop + 12}px`;
      toast.style.transform = "translateX(-100%)";
      this.toast = toast;
      setTimeout(() => toast.remove(), 8000);
    }

    retryNeural() {
      this.neuralBlocked = false;
      this.note = "";
      this.retryNeuralAt = 0;
      this.createEngine();
    }

    fail(detail) {
      this.state = "error";
      this.detail = detail;
      this.hide();
    }

    flush() {
      this.engine?.reset();
      this.lastShown = -Infinity;
      this.lastMediaTime = null;
      this.clockOffset = null;
      this.deltas.length = 0;
    }

    destroy() {
      this.alive = false;
      const { video } = this;
      video.cancelVideoFrameCallback(this.frameHandle);
      cancelAnimationFrame(this.rafHandle);
      clearInterval(this.layoutTimer);
      this.resizeObserver.disconnect();
      video.removeEventListener("seeking", this.onDiscontinuity);
      video.removeEventListener("emptied", this.onDiscontinuity);
      video.removeEventListener("ratechange", this.onDiscontinuity);
      document.removeEventListener("fullscreenchange", this.onFullscreen);
      document.removeEventListener("pointermove", this.onPointer, { capture: true });
      this.engine?.dispose();
      this.audio?.set(0);
      this.canvas.remove();
      this.captions.destroy();
      this.hud?.remove();
      this.toast?.remove();
    }

    get sourceFps() {
      if (this.deltas.length < 4) return 0;
      const sorted = [...this.deltas].sort((a, b) => a - b);
      return (this.video.playbackRate || 1) / sorted[sorted.length >> 1];
    }

    get effectiveTarget() {
      return Math.min(settings.targetFps || Infinity, this.refreshRate);
    }

    // State that stops interpolation regardless of frames arriving. Null means free to run.
    blockedState() {
      const { video } = this;
      if (!settings.enabled) return "off";
      if (video.mediaKeys) return "drm";
      if (this.state === "error") return "error";
      if (this.corsState === "tainted") return "tainted";
      if (this.corsState === "failed") return "corsFailed";
      if (document.fullscreenElement === video) return "fullscreen";
      if (video.controls && performance.now() < this.revealedUntil) return "controls";
      return null;
    }

    onVideoFrame(now, meta) {
      if (!this.alive) return;
      this.frameHandle = this.video.requestVideoFrameCallback((n, m) => this.onVideoFrame(n, m));
      if (this.blockedState() || !this.engine) return;

      const { mediaTime, expectedDisplayTime } = meta;
      const rate = this.video.playbackRate || 1;
      if (this.lastMediaTime !== null) {
        const delta = mediaTime - this.lastMediaTime;
        if (delta <= 0 || delta > 0.5) {
          this.flush();
        } else {
          this.deltas.push(delta);
          if (this.deltas.length > 24) this.deltas.shift();
        }
      }
      this.lastMediaTime = mediaTime;

      const offset = mediaTime - (rate * expectedDisplayTime) / 1000;
      if (this.clockOffset === null || Math.abs(offset - this.clockOffset) > 0.1) this.clockOffset = offset;
      else this.clockOffset += (offset - this.clockOffset) * 0.1;

      // No point paying for flow on a source already at the target rate.
      if (this.sourceFps >= this.effectiveTarget * 0.92) return;
      try {
        this.engine.pushFrame(this.video, this.video.videoWidth, this.video.videoHeight, mediaTime);
      } catch (error) {
        if (error.name === "SecurityError") this.corsState = "tainted";
        else this.fail(error.message);
        return;
      }
    }

    onAnimationFrame(now) {
      if (!this.alive) return;
      this.rafHandle = requestAnimationFrame((n) => this.onAnimationFrame(n));
      this.measureRefresh(now);
      if (this.neural && this.engine.overloaded) {
        this.dropNeural("显卡负载过高（例如正在玩游戏），AI 模型跟不上，每 60 秒自动重试一次", true);
      } else if (this.retryNeuralAt && now > this.retryNeuralAt) {
        this.retryNeural();
      }
      this.state = this.nextState();
      if (this.neural) {
        // Each page would otherwise spend its first seconds finding the size this GPU can run.
        const height = this.engine.settledHeight;
        if (height && height !== rememberedModelHeight) {
          rememberedModelHeight = height;
          chrome.storage.local.set({ modelHeight: height });
        }
        if (height && this.fellBack) {
          this.fellBack = false;
          this.notify("FrameBoost：AI 补帧已恢复");
        }
        const delayed = this.state === "running" || this.state === "holding" || this.state === "paused";
        this.audio.set(delayed ? FrameBoostNeural.NeuralInterpolator.DISPLAY_DELAY : 0);
      }
      if (this.state !== "running" && this.state !== "holding") {
        this.hide();
      } else {
        this.draw(now);
      }
      if (this.neural) this.engine.pump(now);
      this.updateHud();
    }

    nextState() {
      const blocked = this.blockedState();
      if (blocked) return blocked;
      const { video } = this;
      if (this.neural) {
        if (!this.engine.net) return "loading";
        if (!this.audio.ensure()) {
          if (!this.audio.failed) return "audioWait";
          this.dropNeural("这个页面自己接管了声音，无法同步延后");
          return "paused";
        }
        // The picture is behind the element, so it plays out to the last frame and holds there.
        if (!video.seeking && (video.paused || video.ended || video.readyState < 3)) return this.engine.ready ? "holding" : "paused";
      }
      if (video.paused || video.ended || video.seeking || video.readyState < 3) return "paused";
      if (this.deltas.length >= 4 && this.sourceFps >= this.effectiveTarget * 0.92) return "passthrough";
      if (!this.engine?.ready || this.deltas.length < 4) return "paused";
      return "running";
    }

    measureRefresh(now) {
      if (this.lastRaf) {
        const delta = now - this.lastRaf;
        if (delta > 2 && delta < 60) this.rafDeltas.push(delta);
        if (this.rafDeltas.length > 60) this.rafDeltas.shift();
        if (this.rafDeltas.length >= 30) {
          // Mean, not median: rAF timestamps are rounded, so single deltas read 143 or 145 on 144 Hz.
          const mean = this.rafDeltas.reduce((sum, value) => sum + value, 0) / this.rafDeltas.length;
          this.refreshRate = Math.round(1000 / mean);
        }
      }
      this.lastRaf = now;
    }

    draw(now) {
      // Whole divisors of the refresh rate only: 120 fps on a 144 Hz screen would repeat every
      // sixth refresh and judder, so the output rounds up to the next rate that paces evenly.
      const every = Math.max(1, Math.floor(this.refreshRate / this.effectiveTarget + 0.05));
      this.tick = ((this.tick ?? 0) + 1) % every;
      if (this.tick !== 0) return;

      if (!this.canvas.isConnected) this.syncLayout();
      if (!this.layout) return;

      // Interpolating needs the next real frame, so the picture runs one source frame behind.
      // The network also needs time, and it gets a fixed delay that the audio shares.
      const period = this.percentileDelta();
      const rate = this.video.playbackRate || 1;
      const clock = (rate * now) / 1000 + this.clockOffset;
      const margin = (rate * (1000 / this.refreshRate + 4)) / 1000;
      let time = clock - period - margin;
      if (this.neural) {
        time = Math.max(this.lastShown, clock - margin - rate * FrameBoostNeural.NeuralInterpolator.DISPLAY_DELAY);
        this.lastShown = time;
      }
      const drawn = this.engine.render(time, this.layout.viewport, this.layout.crop);
      if (!drawn) return;

      this.canvas.style.setProperty("display", "block", "important");
      this.captions.show(this.canvas, this.neural ? time : undefined);
      this.drawCount++;
      const elapsed = now - this.drawWindowStart;
      if (elapsed >= 500) {
        this.outputFps = (this.drawCount * 1000) / elapsed;
        this.drawCount = 0;
        this.drawWindowStart = now;
      }
    }

    percentileDelta() {
      const sorted = [...this.deltas].sort((a, b) => a - b);
      return Math.min(0.1, sorted[Math.floor(sorted.length * 0.9)] ?? 1 / 24);
    }

    hide() {
      this.canvas.style.setProperty("display", "none", "important");
      this.captions?.hide();
      this.outputFps = 0;
      this.drawCount = 0;
      this.drawWindowStart = performance.now();
    }

    syncLayout() {
      const { video, canvas } = this;
      if (!this.alive || !video.isConnected) return;
      if (!canvas.isConnected || canvas.previousElementSibling !== video) video.after(canvas);

      const style = getComputedStyle(video);
      const fixed = style.position === "fixed" || !video.offsetParent;
      const rect = fixed ? video.getBoundingClientRect() : null;
      const width = fixed ? rect.width : video.offsetWidth;
      const height = fixed ? rect.height : video.offsetHeight;
      const set = (name, value) => canvas.style.setProperty(name, value, "important");
      set("position", fixed ? "fixed" : "absolute");
      set("left", `${fixed ? rect.left : video.offsetLeft}px`);
      set("top", `${fixed ? rect.top : video.offsetTop}px`);
      set("width", `${width}px`);
      set("height", `${height}px`);
      set("transform", style.transform);
      set("transform-origin", style.transformOrigin);
      set("filter", style.filter);
      set("border-radius", style.borderRadius);
      if (style.zIndex !== "auto") set("z-index", style.zIndex);
      else canvas.style.removeProperty("z-index");

      // The AI engine draws at most at source size: a 1080p video on a 4K screen would otherwise
      // cost four times the pixels at every refresh, taking GPU time from the network.
      let dpr = devicePixelRatio || 1;
      if (this.neural && video.videoWidth) {
        const fit = Math.min(width / video.videoWidth, height / video.videoHeight) * dpr;
        if (fit > 1) dpr /= fit;
      }
      const pixelWidth = Math.max(1, Math.round(width * dpr));
      const pixelHeight = Math.max(1, Math.round(height * dpr));
      if (canvas.width !== pixelWidth) canvas.width = pixelWidth;
      if (canvas.height !== pixelHeight) canvas.height = pixelHeight;
      this.layout = video.videoWidth
        ? fitContent(style.objectFit, style.objectPosition, video.videoWidth, video.videoHeight, pixelWidth, pixelHeight)
        : null;
      this.captions.sync(canvas, height);
    }

    handlePointer(event) {
      if (!this.video.controls) return;
      const rect = this.video.getBoundingClientRect();
      const over =
        event.clientX >= rect.left && event.clientX <= rect.right && event.clientY >= rect.top && event.clientY <= rect.bottom;
      if (over) this.revealedUntil = performance.now() + 2500;
    }

    updateHud() {
      if (!settings.hud) {
        this.hud?.remove();
        this.hud = null;
        return;
      }
      if (!this.hud) {
        this.hud = document.createElement("div");
        this.hud.dataset.frameboostHud = "";
        this.hud.setAttribute("aria-hidden", "true");
        this.hud.style.cssText =
          "position:absolute;pointer-events:none;z-index:2147483647;padding:4px 8px;border-radius:8px;" +
          "background:rgba(8,17,31,.72);color:#e5edf7;font:12px/1.4 'Segoe UI','Microsoft YaHei',sans-serif;" +
          "font-variant-numeric:tabular-nums;white-space:nowrap";
      }
      if (!this.hud.isConnected && this.canvas.isConnected) this.canvas.after(this.hud);
      this.hud.style.left = `${this.video.offsetLeft + 8}px`;
      this.hud.style.top = `${this.video.offsetTop + 8}px`;
      const now = performance.now();
      if (now - (this.hudUpdated || 0) < 250) return;
      this.hudUpdated = now;
      const model = this.neural ? this.modelText() : settings.quality === "neural" ? " · 光流（AI 未生效）" : " · 光流";
      this.hud.textContent =
        this.state === "running"
          ? `FrameBoost ${this.sourceFps.toFixed(0)} → ${this.outputFps.toFixed(0)} fps · ${this.refreshRate} Hz${model}`
          : `FrameBoost · ${STATE_TEXT[this.state] ?? this.state}`;
    }

    modelText() {
      const { modelHeight, inferMs } = this.engine.stats();
      return modelHeight ? ` · AI ${modelHeight}p ${Math.round(inferMs)} ms` : "";
    }

    status() {
      const { video } = this;
      return {
        model: this.neural ? this.modelText().replace(" · ", "") : "",
        stats: this.neural ? this.engine.stats() : null,
        engine: this.engine ? (this.neural ? "neural" : "flow") : "",
        audioDelay: this.neural ? this.audio.target : 0,
        note: this.note ?? "",
        state: this.state,
        message: this.detail || STATE_TEXT[this.state] || this.state,
        sourceFps: Math.round(this.sourceFps * 10) / 10,
        outputFps: Math.round(this.outputFps),
        refreshRate: this.refreshRate,
        width: video.videoWidth,
        height: video.videoHeight,
        area: video.clientWidth * video.clientHeight,
        canReloadCors: this.corsState === "tainted",
      };
    }

    // Reloads the same URL with CORS enabled; the service worker adds the response header.
    async reloadWithCors() {
      const { video } = this;
      const src = video.currentSrc;
      if (!/^https?:/.test(src)) return;
      const time = video.currentTime;
      const wasPlaying = !video.paused;
      await chrome.runtime.sendMessage({ type: "allow-media-cors" });
      const restore = (crossOrigin) => {
        if (crossOrigin) video.crossOrigin = crossOrigin;
        else video.removeAttribute("crossorigin");
        video.src = src;
        video.load();
        video.addEventListener(
          "loadedmetadata",
          () => {
            video.currentTime = time;
            if (wasPlaying) video.play().catch(() => {});
          },
          { once: true },
        );
      };
      const onError = () => {
        this.corsState = "failed";
        restore(null);
      };
      video.addEventListener("error", onError, { once: true });
      video.addEventListener("loadeddata", () => video.removeEventListener("error", onError), { once: true });
      this.corsState = null;
      this.flush();
      restore("anonymous");
    }
  }

  function fitContent(fit, position, videoWidth, videoHeight, boxWidth, boxHeight) {
    const crop = { x: 0, y: 0, width: 1, height: 1 };
    const viewport = { x: 0, y: 0, width: boxWidth, height: boxHeight };
    if (fit === "fill") return { viewport, crop };
    const parts = position.split(" ").map((part) => (part.endsWith("%") ? parseFloat(part) / 100 : 0.5));
    const px = parts[0] ?? 0.5;
    const py = parts[1] ?? 0.5;
    const videoAspect = videoWidth / videoHeight;
    const boxAspect = boxWidth / boxHeight;
    if (fit === "cover") {
      if (videoAspect > boxAspect) {
        crop.width = boxAspect / videoAspect;
        crop.x = (1 - crop.width) * px;
      } else {
        crop.height = videoAspect / boxAspect;
        crop.y = (1 - crop.height) * py;
      }
      return { viewport, crop };
    }
    // contain (and "none"/"scale-down", which behave the same for full-size players)
    if (videoAspect > boxAspect) {
      viewport.height = Math.round(boxWidth / videoAspect);
      viewport.y = Math.round((boxHeight - viewport.height) * (1 - py));
    } else {
      viewport.width = Math.round(boxHeight * videoAspect);
      viewport.x = Math.round((boxWidth - viewport.width) * px);
    }
    return { viewport, crop };
  }

  function collectVideos(root, found = []) {
    root.querySelectorAll("video").forEach((video) => found.push(video));
    root.querySelectorAll("*").forEach((element) => {
      if (element.shadowRoot) collectVideos(element.shadowRoot, found);
    });
    return found;
  }

  function isCandidate(video) {
    return (
      video.isConnected &&
      !video.paused &&
      !video.ended &&
      video.videoWidth > 0 &&
      video.clientWidth >= MIN_WIDTH &&
      video.getClientRects().length > 0
    );
  }

  function scan() {
    if (session && !session.video.isConnected) {
      session.destroy();
      session = null;
    }
    if (session && (!session.video.paused || session.video.seeking)) return;
    const videos = collectVideos(document).filter(isCandidate);
    if (!videos.length) return;
    const largest = videos.reduce((a, b) => (b.clientWidth * b.clientHeight > a.clientWidth * a.clientHeight ? b : a));
    if (session?.video === largest) return;
    session?.destroy();
    session = new VideoSession(largest);
  }

  function applySettings(next) {
    const before = settings.quality;
    settings = { ...DEFAULTS, ...next };
    if (session?.neural) session.engine.adaptive = !settings.keepModel;
    if (settings.quality === before || !session) return;
    if (settings.quality === "neural" || before === "neural") {
      session.audio?.set(0);
      session.retryNeural();
    } else {
      session.engine?.setQuality(settings.quality);
      session.flush();
    }
  }

  // Nothing starts before the settings are in, or the first engine could be the wrong one.
  Promise.all([chrome.storage.sync.get(DEFAULTS), chrome.storage.local.get("modelHeight")]).then(([stored, local]) => {
    rememberedModelHeight = local.modelHeight;
    applySettings(stored);
    document.addEventListener("playing", scan, true);
    setInterval(scan, 2000);
    scan();
  });
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== "sync") return;
    const next = { ...settings };
    for (const [key, change] of Object.entries(changes)) next[key] = change.newValue;
    applySettings(next);
  });

  // Media events do not bubble but do pass through capture, so new players are noticed at once.
  globalThis.__frameBoost = {
    status: () => (session ? session.status() : { state: "idle", message: STATE_TEXT.idle, area: 0 }),
    reloadWithCors: () => session?.reloadWithCors(),
    audioNode: () => session?.audio?.delayNode ?? null,
  };
})();
