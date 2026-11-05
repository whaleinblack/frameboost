// FrameBoost neural engine: one RIFE inference per source pair, then any number of cheap output
// frames from its mid-time flow, so 24 → 120 fps costs no more network time than 24 → 48.
// Same interface as FlowInterpolator, plus a load promise and a display delay.
(() => {
  "use strict";

  const RING = 12; // source frames kept: the display delay plus what is still on the GPU
  const MAX_JOBS = 2;
  const SLICE_MARGIN = 1.05; // just ahead of the next pair: a larger slice would overrun its refresh
  const MAX_SLICES_QUEUED = 2;
  const MAX_SLICES = 64; // per pair, for timing; a screen refresh takes one at most
  const MODEL_HEIGHTS = [1088, 960, 896, 832, 768, 704, 640, 576, 512, 448, 384];
  const MODEL_STEP = 64; // the network halves its input six times at most
  const NETWORK_SCALE = 1; // on real footage 0.5 costs 3.4 dB, worse than a quarter of the pixels at 1
  const WINDOW = 36; // pairs between decisions on the model size
  const FAST_WINDOW = 6; // sooner when it is hopeless, because waiting costs visible stutter
  // Share of the time between source frames the network may use. Measured at 144 Hz on 1080p 24 fps:
  // 64% keeps every refresh, 80% falls to 92 fps, 92% to 55 fps. The rest is drawing and compositing.
  const GPU_SHARE = 0.72;
  const CUT_RESIDUAL = 0.09; // after repair: real motion stayed under 0.07 even at 12.5 fps, a hard cut measured 0.116
  const GRID = [64, 36];
  const FORMAT_FLOW = "rgba16float";

  // Frame texture into the rgb tensor at model size. Stretching by under 1% to a multiple of 64
  // beats padding: the model never sees a fake black border, and flow is stored in uv units.
  const TENSOR_SHADER = /* wgsl */ `
struct Dim { w: i32, h: i32, radius: f32 }
@group(0) @binding(0) var<uniform> d: Dim;
@group(0) @binding(1) var frame: texture_2d<f32>;
@group(0) @binding(2) var linear: sampler;
@group(0) @binding(3) var<storage, read_write> dst: array<vec4f>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let x = i32(g.x);
  let y = i32(g.y);
  if (x >= d.w || y >= d.h) { return; }
  let size = vec2f(f32(d.w), f32(d.h));
  let uv = (vec2f(f32(x), f32(y)) + 0.5) / size;
  let o = d.radius / size;
  var c = textureSampleLevel(frame, linear, uv + vec2f(-o.x, -o.y), 0.0) + textureSampleLevel(frame, linear, uv + vec2f(o.x, -o.y), 0.0);
  c += textureSampleLevel(frame, linear, uv + vec2f(-o.x, o.y), 0.0) + textureSampleLevel(frame, linear, uv + vec2f(o.x, o.y), 0.0);
  dst[y * d.w + x] = vec4f(c.rgb * 0.25, 0.0);
}`;

  // Network result into filterable textures. The flow only has detail at half resolution.
  const FLOW_SHADER = /* wgsl */ `
struct Dim { w: i32, h: i32, step: i32 }
@group(0) @binding(0) var<uniform> d: Dim;
@group(0) @binding(1) var<storage, read> flow: array<vec4f>;
@group(0) @binding(2) var<storage, read> mask: array<f32>;
@group(0) @binding(3) var flowOut: texture_storage_2d<${FORMAT_FLOW}, write>;
@group(0) @binding(4) var maskOut: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let size = vec2i(textureDimensions(flowOut));
  let p = vec2i(g.xy);
  if (p.x >= size.x || p.y >= size.y) { return; }
  var f = vec4f(0.0);
  var m = 0.0;
  for (var y = 0; y < d.step; y++) {
    for (var x = 0; x < d.step; x++) {
      let i = min(p.y * d.step + y, d.h - 1) * d.w + min(p.x * d.step + x, d.w - 1);
      f += flow[i];
      m += mask[i];
    }
  }
  let n = f32(d.step * d.step);
  textureStore(flowOut, p, f / n / vec4f(f32(d.w), f32(d.h), f32(d.w), f32(d.h)));
  textureStore(maskOut, p, vec4f(1.0 / (1.0 + exp(-m / n))));
}`;

  // What the flow cannot explain, on a coarse grid. Across a cut the warped frames stay different.
  const RESIDUAL_SHADER = /* wgsl */ `
struct Dim { w: i32, h: i32, gridW: i32, gridH: i32 }
@group(0) @binding(0) var<uniform> d: Dim;
@group(0) @binding(1) var<storage, read> a: array<vec4f>;
@group(0) @binding(2) var<storage, read> b: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let x = i32(g.x);
  let y = i32(g.y);
  if (x >= d.gridW || y >= d.gridH) { return; }
  let sx = x * d.w / d.gridW;
  let sy = y * d.h / d.gridH;
  let i = (sy * d.w + sx) * 2;
  let e = abs(a[i].xyz - b[i].xyz);
  dst[y * d.gridW + x] = (e.x + e.y + e.z) / 3.0;
}`;

  const VERTEX = /* wgsl */ `
struct Out { @builtin(position) position: vec4f, @location(0) uv: vec2f }
@vertex fn vs(@builtin(vertex_index) i: u32) -> Out {
  let p = vec2f(f32((i & 1u) << 2u), f32((i & 2u) << 1u)) - 1.0;
  return Out(vec4f(p, 0.0, 1.0), vec2f(p.x + 1.0, 1.0 - p.y) * 0.5);
}`;

  // Flow is anchored at the middle: a scene point q sits at q + F0(q) in frame A and q + F1(q) in
  // frame B. It moves in a straight line to q and on, so at t = 0 and t = 1 the real frames come
  // back exactly. Finding q for a screen position is a fixed-point search that keeps its best step.
  const SYNTH_SHADER = /* wgsl */ `
${VERTEX}
struct Params { crop: vec4f, t: f32 }
@group(0) @binding(0) var<uniform> u: Params;
@group(0) @binding(1) var frameA: texture_2d<f32>;
@group(0) @binding(2) var frameB: texture_2d<f32>;
@group(0) @binding(3) var flow: texture_2d<f32>;
@group(0) @binding(4) var mask: texture_2d<f32>;
@group(0) @binding(5) var linear: sampler;

// Catmull-Rom from five bilinear taps. A plain bilinear warp softens every inserted frame, and a
// sharp real frame next to a soft one reads as pulsing.
fn sharp(frame: texture_2d<f32>, uv: vec2f) -> vec3f {
  let size = vec2f(textureDimensions(frame));
  let p = uv * size;
  let base = floor(p - 0.5) + 0.5;
  let f = p - base;
  let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
  let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
  let w3 = f * f * (-0.5 + 0.5 * f);
  let w12 = w1 + w2;
  let o12 = w2 / w12;
  let t0 = (base - 1.0) / size;
  let t3 = (base + 2.0) / size;
  let t12 = (base + o12) / size;
  var c = textureSampleLevel(frame, linear, vec2f(t12.x, t0.y), 0.0).rgb * w12.x * w0.y;
  c += textureSampleLevel(frame, linear, vec2f(t0.x, t12.y), 0.0).rgb * w0.x * w12.y;
  c += textureSampleLevel(frame, linear, t12, 0.0).rgb * w12.x * w12.y;
  c += textureSampleLevel(frame, linear, vec2f(t3.x, t12.y), 0.0).rgb * w3.x * w12.y;
  c += textureSampleLevel(frame, linear, vec2f(t12.x, t3.y), 0.0).rgb * w12.x * w3.y;
  let norm = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
  return clamp(c / norm, vec3f(0.0), vec3f(1.0));
}

fn inside(uv: vec2f) -> f32 {
  return step(0.0, uv.x) * step(uv.x, 1.0) * step(0.0, uv.y) * step(uv.y, 1.0);
}

fn solve(x: vec2f, t: f32) -> vec2f {
  let before = t < 0.5;
  let s = select(2.0 * t - 1.0, 1.0 - 2.0 * t, before);
  var q = x;
  var best = x;
  var bestError = 1e9;
  for (var i = 0; i < 6; i++) {
    let f = textureSampleLevel(flow, linear, q, 0.0);
    let d = select(f.zw, f.xy, before);
    let error = length(q + s * d - x);
    if (error < bestError) {
      bestError = error;
      best = q;
    }
    q = x - s * d;
  }
  return best;
}

@fragment fn fs(@location(0) screen: vec2f) -> @location(0) vec4f {
  let x = screen * u.crop.zw + u.crop.xy;
  let q = solve(x, u.t);
  let f = textureSampleLevel(flow, linear, q, 0.0);
  let m = textureSampleLevel(mask, linear, q, 0.0).r;
  // On its own side of the middle a frame is reached from x directly, not through the solved q:
  // it stays exact at the real frame however rough the search was, so nothing jumps there.
  let pa = select(q + f.xy, x + 2.0 * u.t * f.xy, u.t < 0.5);
  let pb = select(x + 2.0 * (1.0 - u.t) * f.zw, q + f.zw, u.t < 0.5);
  let a = sharp(frameA, pa);
  let b = sharp(frameB, pb);
  // A flow leaving the picture has no real pixel there, only a stretched edge: use the other frame.
  let wa = (1.0 - u.t) * (m + 0.01) * (inside(pa) + 0.001);
  let wb = u.t * (1.01 - m) * (inside(pb) + 0.001);
  return vec4f((a * wa + b * wb) / (wa + wb), 1.0);
}`;

  const BLIT_SHADER = /* wgsl */ `
${VERTEX}
struct Params { crop: vec4f, t: f32 }
@group(0) @binding(0) var<uniform> u: Params;
@group(0) @binding(1) var frame: texture_2d<f32>;
@group(0) @binding(2) var linear: sampler;
@fragment fn fs(@location(0) screen: vec2f) -> @location(0) vec4f {
  return vec4f(textureSampleLevel(frame, linear, screen * u.crop.zw + u.crop.xy, 0.0).rgb, 1.0);
}`;

  function modelSize(width, height, limit) {
    const h = Math.min(limit, Math.max(MODEL_STEP * 2, Math.round(height / MODEL_STEP) * MODEL_STEP));
    const w = Math.max(MODEL_STEP * 2, Math.round((width * (h / height)) / MODEL_STEP) * MODEL_STEP);
    return { w, h };
  }

  class NeuralInterpolator {
    /** Delay between a frame arriving and it being on screen. Audio is held back by the same. */
    static DISPLAY_DELAY = 0.3;

    constructor(canvas, options) {
      this.canvas = canvas;
      this.modelFiles = options.modelFiles;
      this.modelHeight = options.modelHeight ?? MODEL_HEIGHTS[0];
      this.networkScale = options.networkScale ?? NETWORK_SCALE;
      this.onLost = options.onLost;
      this.adaptive = options.adaptive ?? true;
      this.useRepair = options.repair ?? true;
      this.ring = options.ring ?? RING;
      this.frames = [];
      this.overloaded = false;
      this.inferMs = 0;
      this.pairs = 0;
      this.skipped = 0;
      this.jobs = [];
      this.slicesQueued = 0;
      this.pumpChannel = new MessageChannel();
      this.pumpChannel.port1.onmessage = () => {
        this.pumpQueued = false;
        if (!this.disposed && this.net) this.#pumpNow();
      };
      this.framePeriod = 1000 / 24;
      this.loaded = this.#load();
    }

    async #load() {
      const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
      if (!adapter) throw new Error("WebGPU 不可用");
      const limits = adapter.limits;
      this.timed = adapter.features.has("timestamp-query");
      this.device = await adapter.requestDevice({
        requiredFeatures: this.timed ? ["timestamp-query"] : [],
        requiredLimits: {
          maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
          maxBufferSize: limits.maxBufferSize,
          maxStorageBuffersPerShaderStage: Math.min(10, limits.maxStorageBuffersPerShaderStage),
          maxStorageTexturesPerShaderStage: Math.min(4, limits.maxStorageTexturesPerShaderStage),
        },
      });
      this.device.lost.then((info) => {
        if (info.reason !== "destroyed") this.onLost?.(info.message);
      });
      const [bin, index] = await Promise.all([
        fetch(this.modelFiles.bin).then((r) => r.arrayBuffer()),
        fetch(this.modelFiles.json).then((r) => r.json()),
      ]);
      this.net = new FrameBoostRife.RifeNetwork(this.device, new Float32Array(bin), index);
      this.context = this.canvas.getContext("webgpu");
      this.format = navigator.gpu.getPreferredCanvasFormat();
      this.context.configure({ device: this.device, format: this.format, alphaMode: "premultiplied" });
      const compute = (code) =>
        this.device.createComputePipeline({ layout: "auto", compute: { module: this.device.createShaderModule({ code }), entryPoint: "main" } });
      const render = (code) => {
        const module = this.device.createShaderModule({ code });
        return this.device.createRenderPipeline({
          layout: "auto",
          vertex: { module, entryPoint: "vs" },
          fragment: { module, entryPoint: "fs", targets: [{ format: this.format }] },
        });
      };
      this.tensorPipeline = compute(TENSOR_SHADER);
      this.flowPipeline = compute(FLOW_SHADER);
      this.residualPipeline = compute(RESIDUAL_SHADER);
      this.synthPipeline = render(SYNTH_SHADER);
      this.blitPipeline = render(BLIT_SHADER);
      this.sampler = this.device.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" });
      this.params = this.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      if (this.useRepair) this.repair = new FrameBoostRepair.FlowRepair(this.device, this.sampler);
      if (this.timed) {
        this.querySet = this.device.createQuerySet({ type: "timestamp", count: MAX_SLICES * 2 });
        this.queryResolve = this.device.createBuffer({ size: MAX_SLICES * 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      }
    }

    /** Forget all history, e.g. after a seek, so no flow crosses the discontinuity. */
    reset() {
      this.frames.length = 0;
      this.generation = (this.generation ?? 0) + 1;
      this.tensorOwners = [null, null];
      this.jobs = [];
      this.warmup = 2;
    }

    setQuality() {}

    pushFrame(source, width, height, mediaTime) {
      if (!this.net) return;
      if (width !== this.sourceWidth || height !== this.sourceHeight || this.needsAllocate) this.#allocate(width, height);
      const slot = this.slots[this.nextSlot];
      this.nextSlot = (this.nextSlot + 1) % this.ring;
      // Work this far behind is useless. Dropping it leaves the tensors in an unknown state.
      if (this.jobs.some((job) => job.slot === slot || job.previous === slot)) {
        this.jobs = [];
        this.tensorOwners = [null, null];
      }
      this.tensorOwners = this.tensorOwners.map((owner) => (owner === slot ? null : owner));
      this.device.queue.copyExternalImageToTexture({ source }, { texture: slot.frame, colorSpace: "srgb" }, [width, height]);
      slot.mediaTime = mediaTime;
      slot.pair = null;
      slot.generation = this.generation;
      const arrived = performance.now();
      if (this.lastArrival && arrived - this.lastArrival < 500) this.framePeriod += (arrived - this.lastArrival - this.framePeriod) * 0.1;
      this.lastArrival = arrived;

      const previous = this.frames[this.frames.length - 1];
      if (previous) this.#schedule(previous, slot);
      this.frames.push(slot);
      if (this.frames.length > this.ring - 1) this.frames.shift();
    }

    /** Draws mediaTime. A pair still on the GPU, or across a cut, shows its nearest real frame. */
    render(mediaTime, viewport, crop) {
      if (this.frames.length < 2) return false;
      let i = 1;
      while (i < this.frames.length - 1 && mediaTime >= this.frames[i].mediaTime) i++;
      const a = this.frames[i - 1];
      const b = this.frames[i];
      const t = Math.min(1, Math.max(0, (mediaTime - a.mediaTime) / (b.mediaTime - a.mediaTime)));
      const pair = b.pair;
      const interpolate = pair?.ready && !pair.cut && t > 0 && t < 1;

      this.device.queue.writeBuffer(this.params, 0, new Float32Array([crop.x, crop.y, crop.width, crop.height, t]));
      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: this.context.getCurrentTexture().createView(), clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "store" }],
      });
      pass.setViewport(viewport.x, this.canvas.height - viewport.y - viewport.height, viewport.width, viewport.height, 0, 1);
      if (interpolate) {
        pass.setPipeline(this.synthPipeline);
        pass.setBindGroup(0, pair.synthGroup);
      } else {
        pass.setPipeline(this.blitPipeline);
        pass.setBindGroup(0, (t < 0.5 ? a : b).blitGroup);
      }
      pass.draw(3);
      pass.end();
      this.device.queue.submit([encoder.finish()]);
      return true;
    }

    get ready() {
      return this.frames.length >= 2; // a pair still on the GPU shows its nearest real frame
    }

    stats() {
      return { inferMs: Math.round(this.inferMs * 10) / 10, modelHeight: this.model?.h ?? 0, skipRate: this.pairs ? this.skipped / this.pairs : 0 };
    }

    /**
     * Gives the network its slice of the GPU. Call once per screen refresh, after drawing. The queue
     * runs in order, so a whole inference would block every picture behind it. One slice at a time
     * keeps the picture ahead of the work, and a busy GPU simply gets less work.
     */
    pump() {
      // After this task: the browser queues the canvas for display when the task ends, and work
      // submitted before that would make every picture wait for its slice.
      if (this.pumpQueued) return;
      this.pumpQueued = true;
      this.pumpChannel.port2.postMessage(0);
    }

    #pumpNow(now = performance.now()) {
      const job = this.jobs[0];
      if (!job) {
        this.lastPump = now;
        return;
      }
      if (this.slicesQueued >= MAX_SLICES_QUEUED) return;
      const elapsed = Math.min(100, now - (this.lastPump ?? now - 7));
      this.lastPump = now;
      const hurry = this.jobs.length > 1 ? 1.3 : 1;
      let quota = (job.cost * elapsed * SLICE_MARGIN * hurry) / this.framePeriod;
      const steps = [];
      while (job.next < job.steps.length && (quota > 0 || !steps.length)) {
        const step = job.steps[job.next++];
        quota -= step.cost;
        steps.push(step);
      }
      const encoder = this.device.createCommandEncoder();
      const timed = this.timed && job.slices < MAX_SLICES;
      const pass = encoder.beginComputePass(
        timed ? { timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: job.slices * 2, endOfPassWriteIndex: job.slices * 2 + 1 } } : {},
      );
      if (timed) job.slices++;
      this.net.run(pass, steps);
      pass.end();
      const done = job.next >= job.steps.length;
      if (done) {
        encoder.copyBufferToBuffer(this.residual, 0, job.slot.residualRead, 0, GRID[0] * GRID[1] * 4);
        if (job.slices) {
          encoder.resolveQuerySet(this.querySet, 0, job.slices * 2, this.queryResolve, 0);
          encoder.copyBufferToBuffer(this.queryResolve, 0, job.slot.timingRead, 0, job.slices * 16);
        }
      }
      this.device.queue.submit([encoder.finish()]);

      this.slicesQueued++;
      this.device.queue.onSubmittedWorkDone().then(() => {
        this.slicesQueued--;
      });
      if (done) {
        this.jobs.shift();
        this.#finish(job);
      }
    }

    dispose() {
      this.disposed = true;
      this.device?.destroy();
    }

    #schedule(previous, slot) {
      const pair = { ready: false, cut: false };
      slot.pair = pair;
      this.pairs++;
      if (this.jobs.length >= MAX_JOBS) {
        this.skipped++;
        this.#judgeLoad();
        return;
      }
      pair.flowTexture = slot.flowTexture;
      pair.synthGroup = this.device.createBindGroup({
        layout: this.synthPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.params } },
          { binding: 1, resource: previous.frame.createView() },
          { binding: 2, resource: slot.frame.createView() },
          { binding: 3, resource: slot.flowTexture.createView() },
          { binding: 4, resource: slot.maskTexture.createView() },
          { binding: 5, resource: this.sampler },
        ],
      });

      const steps = [...this.#planEncode(previous), ...this.#planEncode(slot)];
      const [tensorA, tensorB] = [previous, slot].map((frame) => this.tensors[this.tensorOwners.indexOf(frame)]);
      steps.push(...this.net.planInfer(tensorA, tensorB));
      steps.push(this.#step(this.flowPipeline, slot.flowGroup, this.flowSize.w, this.flowSize.h));
      if (this.repair) steps.push(...this.repair.plan(previous.frame, slot.frame, this.rawFlow, this.rawMask, slot.flowTexture, slot.maskTexture, this.residual, GRID));
      else steps.push(this.#step(this.residualPipeline, this.residualGroup, GRID[0], GRID[1]));
      this.jobs.push({ steps, next: 0, cost: steps.reduce((sum, step) => sum + step.cost, 0), slices: 0, pair, slot, previous, generation: this.generation });
    }

    #finish(job) {
      const { slot, pair, generation } = job;
      const started = performance.now();
      Promise.all([slot.residualRead.mapAsync(GPUMapMode.READ), job.slices ? slot.timingRead.mapAsync(GPUMapMode.READ, 0, job.slices * 16) : null])
        .then(() => {
          let gpuMs = performance.now() - started;
          if (job.slices) {
            const stamps = new BigInt64Array(slot.timingRead.getMappedRange(0, job.slices * 16));
            gpuMs = 0;
            for (let i = 0; i < job.slices; i++) gpuMs += Number(stamps[i * 2 + 1] - stamps[i * 2]) / 1e6;
            slot.timingRead.unmap();
          }
          const values = new Float32Array(slot.residualRead.getMappedRange());
          let sum = 0;
          for (const v of values) sum += v;
          const residual = sum / values.length;
          slot.residualRead.unmap();
          if (generation !== this.generation || slot.pair !== pair) return;
          pair.residual = residual;
          pair.cut = residual > CUT_RESIDUAL;
          pair.ready = true;
          // The first pairs after a resize also pay for shader compilation.
          if (this.warmup > 0) {
            this.warmup--;
            return;
          }
          this.inferMs += (gpuMs - this.inferMs) * (this.inferMs ? 0.15 : 1);
          this.#judgeLoad();
        })
        .catch(() => {});
    }

    // Too slow for this video: fall to a smaller model, and past the smallest report overload so
    // the page can switch to the classic engine.
    #judgeLoad() {
      if (!this.adaptive) return;
      const period = this.framePeriod;
      const hopeless = this.inferMs > period * 2 || this.skipped / Math.max(1, this.pairs) > 0.5;
      if (this.pairs < (hopeless ? FAST_WINDOW : WINDOW)) return;
      const tooSlow = this.skipped / this.pairs > 0.15 || this.inferMs > period * GPU_SHARE;
      this.pairs = 0;
      this.skipped = 0;
      // Only a size chosen for the source's own limit stays out of the memory, not a cap by video size.
      if (!tooSlow && this.model.h >= this.modelHeight) this.settledHeight = this.modelHeight;
      if (tooSlow) {
        // Cost goes with the area, so jump straight to a size that could fit.
        const fits = MODEL_HEIGHTS.find((h) => h < this.model.h && this.inferMs * (h / this.model.h) ** 2 < period * GPU_SHARE * 0.97);
        const smaller = MODEL_HEIGHTS.find((h) => h < this.model.h);
        if (hopeless && !fits) this.overloaded = true;
        else if (fits ?? smaller) this.#resize(fits ?? smaller);
        else this.overloaded = true;
        return;
      }
      // Room to spare, e.g. a game just closed. The margin keeps it from bouncing between sizes.
      const larger = MODEL_HEIGHTS.findLast((h) => h > this.model.h);
      if (larger && this.inferMs * (larger / this.model.h) ** 2 < period * GPU_SHARE * 0.9) {
        if (modelSize(this.sourceWidth, this.sourceHeight, larger).h !== this.model.h) this.#resize(larger);
      }
    }

    #resize(height) {
      this.modelHeight = height;
      this.needsAllocate = true;
    }

    #planEncode(slot) {
      if (this.tensorOwners.includes(slot)) return [];
      // Overwrite whichever tensor the current pair does not need.
      const index = this.tensorOwners[0] === this.frames[this.frames.length - 1] ? 1 : 0;
      this.tensorOwners[index] = slot;
      return [this.#step(this.tensorPipeline, slot.tensorGroup, this.model.w, this.model.h), ...this.net.planEncode(this.rgb, this.tensors[index])];
    }

    #step(pipeline, group, w, h) {
      return { pipeline, group, x: Math.ceil(w / 8), y: Math.ceil(h / 8), z: 1, cost: w * h * 16 };
    }

    #allocate(width, height) {
      this.#releaseSlots();
      this.needsAllocate = false;
      this.sourceWidth = width;
      this.sourceHeight = height;
      const model = modelSize(width, height, this.modelHeight);
      this.model = model;
      this.net.configure(model.w, model.h, this.networkScale);
      this.net.setTime(0.5);
      const d = this.device;
      const uniform = (ints, floats = []) => {
        const buffer = d.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        d.queue.writeBuffer(buffer, 0, new Int32Array(ints));
        if (floats.length) d.queue.writeBuffer(buffer, ints.length * 4, new Float32Array(floats));
        this.owned.push(buffer);
        return buffer;
      };
      const storage = (size, extra = 0) => {
        const buffer = d.createBuffer({ size, usage: GPUBufferUsage.STORAGE | extra });
        this.owned.push(buffer);
        return buffer;
      };
      this.rgb = storage(model.w * model.h * 16, GPUBufferUsage.COPY_DST);
      this.tensors = [this.net.createFrameTensor(), this.net.createFrameTensor()];
      this.tensorOwners = [null, null];
      this.residual = storage(GRID[0] * GRID[1] * 4, GPUBufferUsage.COPY_SRC);
      this.residualGroup = d.createBindGroup({
        layout: this.residualPipeline.getBindGroupLayout(0),
        entries: [uniform([model.w, model.h, GRID[0], GRID[1]]), this.net.warpedA, this.net.warpedB, this.residual].map((buffer, binding) => ({
          binding,
          resource: { buffer },
        })),
      });
      const step = Math.round(1 / this.networkScale);
      this.flowSize = { w: Math.ceil(model.w / step), h: Math.ceil(model.h / step) };
      const tensorDim = uniform([model.w, model.h], [width > model.w * 1.3 ? 0.25 : 0]);
      const flowDim = uniform([model.w, model.h, step]);
      if (this.repair) {
        const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING;
        this.rawFlow = d.createTexture({ size: [this.flowSize.w, this.flowSize.h], format: FORMAT_FLOW, usage });
        this.rawMask = d.createTexture({ size: [this.flowSize.w, this.flowSize.h], format: "rgba8unorm", usage });
        this.textures.push(this.rawFlow, this.rawMask);
        this.repair.configure(width, height);
      }

      this.slots = Array.from({ length: this.ring }, () => {
        const frame = d.createTexture({
          size: [width, height],
          format: "rgba8unorm",
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
        });
        const flowTexture = d.createTexture({ size: [this.flowSize.w, this.flowSize.h], format: FORMAT_FLOW, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING });
        const maskTexture = d.createTexture({ size: [this.flowSize.w, this.flowSize.h], format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING });
        const residualRead = d.createBuffer({ size: GRID[0] * GRID[1] * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        this.owned.push(residualRead);
        const timingRead = d.createBuffer({ size: MAX_SLICES * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        this.owned.push(timingRead);
        this.textures.push(frame, flowTexture, maskTexture);
        return {
          frame,
          flowTexture,
          maskTexture,
          residualRead,
          timingRead,
          mediaTime: 0,
          pair: null,
          tensorGroup: d.createBindGroup({
            layout: this.tensorPipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: tensorDim } },
              { binding: 1, resource: frame.createView() },
              { binding: 2, resource: this.sampler },
              { binding: 3, resource: { buffer: this.rgb } },
            ],
          }),
          flowGroup: d.createBindGroup({
            layout: this.flowPipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: flowDim } },
              { binding: 1, resource: { buffer: this.net.flow } },
              { binding: 2, resource: { buffer: this.net.mask } },
              { binding: 3, resource: (this.rawFlow && this.repair ? this.rawFlow : flowTexture).createView() },
              { binding: 4, resource: (this.rawMask && this.repair ? this.rawMask : maskTexture).createView() },
            ],
          }),
          blitGroup: d.createBindGroup({
            layout: this.blitPipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: this.params } },
              { binding: 1, resource: frame.createView() },
              { binding: 2, resource: this.sampler },
            ],
          }),
        };
      });
      this.nextSlot = 0;
      this.reset();
    }

    #releaseSlots() {
      for (const buffer of this.owned ?? []) buffer.destroy();
      for (const texture of this.textures ?? []) texture.destroy();
      this.owned = [];
      this.textures = [];
      this.slots = null;
    }
  }

  globalThis.FrameBoostNeural = { NeuralInterpolator };
})();
