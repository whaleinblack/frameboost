// FrameBoost RIFE: the RIFE 4.9 flow network as WGSL compute shaders, no ML runtime involved.
// Input is two frames, output is bidirectional flow and a blend mask anchored at the middle time.
// Tensors are storage buffers of vec4f with channels grouped by four, weights are mat4x4f blocks.
// Tensors are planar, one whole plane per channel group: neighbouring threads then read neighbouring
// memory, which measured 3.4 times faster here than keeping a pixel's channels together.
(() => {
  "use strict";

  const BLOCK_SCALES = [8, 4, 2, 1];
  const BLOCK_CHANNELS = [192, 128, 96, 64];
  const WORKGROUP = 8;
  const FRAME_VEC4 = 2; // rgb + 4 feature channels per pixel

  const group4 = (channels) => Math.ceil(channels / 4);

  function convShader({ inChannels, outChannels, stride, activation, residual }) {
    const ic = group4(inChannels);
    const oc = group4(outChannels);
    return /* wgsl */ `
struct Dim { inW: i32, inH: i32, outW: i32, outH: i32 }
@group(0) @binding(0) var<uniform> d: Dim;
@group(0) @binding(1) var<storage, read> src: array<vec4f>;
@group(0) @binding(2) var<storage, read> wt: array<mat4x4f>;
@group(0) @binding(3) var<storage, read> extra: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> dst: array<vec4f>;
const IC = ${ic};
const OC = ${oc};
@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3u) {
  let ox = i32(g.x);
  let oy = i32(g.y);
  if (ox >= d.outW || oy >= d.outH) { return; }
  let o = i32(g.z);
  let plane = d.inW * d.inH;
  var acc = extra[o];
  var tap = 0;
  for (var ky = -1; ky <= 1; ky++) {
    let sy = oy * ${stride} + ky;
    for (var kx = -1; kx <= 1; kx++) {
      let sx = ox * ${stride} + kx;
      if (sx >= 0 && sy >= 0 && sx < d.inW && sy < d.inH) {
        let at = sy * d.inW + sx;
        let w = (o * 9 + tap) * IC;
        for (var i = 0; i < IC; i++) { acc += wt[w + i] * src[i * plane + at]; }
      }
      tap++;
    }
  }
  ${residual ? "acc = acc * extra[OC + o] + src[o * plane + oy * d.inW + ox];" : ""}
  ${activation ? "acc = max(acc, acc * 0.2);" : ""}
  dst[o * d.outW * d.outH + oy * d.outW + ox] = acc;
}`;
  }

  function transposeShader({ inChannels, outChannels }) {
    const ic = group4(inChannels);
    const oc = group4(outChannels);
    return /* wgsl */ `
struct Dim { inW: i32, inH: i32, outW: i32, outH: i32 }
@group(0) @binding(0) var<uniform> d: Dim;
@group(0) @binding(1) var<storage, read> src: array<vec4f>;
@group(0) @binding(2) var<storage, read> wt: array<mat4x4f>;
@group(0) @binding(3) var<storage, read> extra: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> dst: array<vec4f>;
const IC = ${ic};
const OC = ${oc};
@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3u) {
  let ox = i32(g.x);
  let oy = i32(g.y);
  if (ox >= d.outW || oy >= d.outH) { return; }
  // Kernel 4, stride 2, padding 1: output = 2 * input + k - 1, so two taps per axis contribute.
  let kx0 = (ox + 1) & 1;
  let ky0 = (oy + 1) & 1;
  let o = i32(g.z);
  let plane = d.inW * d.inH;
  var acc = extra[o];
  for (var ky = ky0; ky < 4; ky += 2) {
    let sy = (oy + 1 - ky) / 2;
    for (var kx = kx0; kx < 4; kx += 2) {
      let sx = (ox + 1 - kx) / 2;
      if (sx >= 0 && sy >= 0 && sx < d.inW && sy < d.inH) {
        let at = sy * d.inW + sx;
        let w = (o * 16 + ky * 4 + kx) * IC;
        for (var i = 0; i < IC; i++) { acc += wt[w + i] * src[i * plane + at]; }
      }
    }
  }
  dst[o * d.outW * d.outH + oy * d.outW + ox] = acc;
}`;
  }

  const FRAME_SAMPLERS = ["a", "b"]
    .map(
      (name) => /* wgsl */ `
fn sample_${name}(x: f32, y: f32, c: i32) -> vec4f {
  let px = clamp(x, 0.0, f32(d.w - 1));
  let py = clamp(y, 0.0, f32(d.h - 1));
  let x0 = i32(px);
  let y0 = i32(py);
  let x1 = min(x0 + 1, d.w - 1);
  let y1 = min(y0 + 1, d.h - 1);
  let fx = px - f32(x0);
  let fy = py - f32(y0);
  let top = mix(${name}[(y0 * d.w + x0) * ${FRAME_VEC4} + c], ${name}[(y0 * d.w + x1) * ${FRAME_VEC4} + c], fx);
  let bottom = mix(${name}[(y1 * d.w + x0) * ${FRAME_VEC4} + c], ${name}[(y1 * d.w + x1) * ${FRAME_VEC4} + c], fx);
  return mix(top, bottom, fy);
}`,
    )
    .join("\n");

  // Moves the rgb and feature channels of both frames along the current flow, at full resolution.
  const WARP_SHADER = /* wgsl */ `
struct Dim { w: i32, h: i32 }
@group(0) @binding(0) var<uniform> d: Dim;
@group(0) @binding(1) var<storage, read> flow: array<vec4f>;
@group(0) @binding(2) var<storage, read> a: array<vec4f>;
@group(0) @binding(3) var<storage, read> b: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> outA: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> outB: array<vec4f>;
${FRAME_SAMPLERS}
@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3u) {
  let x = i32(g.x);
  let y = i32(g.y);
  if (x >= d.w || y >= d.h) { return; }
  let i = y * d.w + x;
  let f = flow[i];
  let px = f32(x);
  let py = f32(y);
  outA[i * 2] = sample_a(px + f.x, py + f.y, 0);
  outA[i * 2 + 1] = sample_a(px + f.x, py + f.y, 1);
  outB[i * 2] = sample_b(px + f.z, py + f.w, 0);
  outB[i * 2 + 1] = sample_b(px + f.z, py + f.w, 1);
}`;

  // Network input of one block: the frames and the current estimate, resized to the block's scale.
  const prepShader = (first) => /* wgsl */ `
struct Dim { w: i32, h: i32, outW: i32, outH: i32, scale: f32, t: f32 }
@group(0) @binding(0) var<uniform> d: Dim;
@group(0) @binding(1) var<storage, read> a: array<vec4f>;
@group(0) @binding(2) var<storage, read> b: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> dst: array<vec4f>;
${first ? "" : "@group(0) @binding(4) var<storage, read> flow: array<vec4f>;\n@group(0) @binding(5) var<storage, read> mask: array<f32>;"}
${FRAME_SAMPLERS}
@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3u) {
  let ox = i32(g.x);
  let oy = i32(g.y);
  if (ox >= d.outW || oy >= d.outH) { return; }
  let x = (f32(ox) + 0.5) * d.scale - 0.5;
  let y = (f32(oy) + 0.5) * d.scale - 0.5;
  let a0 = sample_a(x, y, 0);
  let a1 = sample_a(x, y, 1);
  let b0 = sample_b(x, y, 0);
  let b1 = sample_b(x, y, 1);
  var m = 0.0;
  ${
    first
      ? ""
      : /* wgsl */ `
  let px = clamp(x, 0.0, f32(d.w - 1));
  let py = clamp(y, 0.0, f32(d.h - 1));
  let x0 = i32(px);
  let y0 = i32(py);
  let x1 = min(x0 + 1, d.w - 1);
  let y1 = min(y0 + 1, d.h - 1);
  let fx = px - f32(x0);
  let fy = py - f32(y0);
  m = mix(mix(mask[y0 * d.w + x0], mask[y0 * d.w + x1], fx), mix(mask[y1 * d.w + x0], mask[y1 * d.w + x1], fx), fy);
  let f = mix(mix(flow[y0 * d.w + x0], flow[y0 * d.w + x1], fx), mix(flow[y1 * d.w + x0], flow[y1 * d.w + x1], fx), fy);`
  }
  let i = oy * d.outW + ox;
  let plane = d.outW * d.outH;
  dst[i] = vec4f(a0.xyz, b0.x);
  dst[i + plane] = vec4f(b0.yz, a0.w, a1.x);
  dst[i + plane * 2] = vec4f(a1.yz, b0.w, b1.x);
  dst[i + plane * 3] = vec4f(b1.yz, d.t, m);
  ${first ? "" : "dst[i + plane * 4] = f / d.scale;"}
}`;

  // PixelShuffle by two, keeping the five channels the model uses: flow xyxy and the mask.
  const SHUFFLE_SHADER = /* wgsl */ `
struct Dim { inW: i32, inH: i32, outW: i32, outH: i32 }
@group(0) @binding(0) var<uniform> d: Dim;
@group(0) @binding(1) var<storage, read> src: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4f>;
@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3u) {
  let ox = i32(g.x);
  let oy = i32(g.y);
  if (ox >= d.outW || oy >= d.outH) { return; }
  let at = (oy / 2) * d.inW + ox / 2;
  let plane = d.inW * d.inH;
  let sub = (oy & 1) * 2 + (ox & 1);
  // Input channel c * 4 + sub sits in plane number c, at component sub.
  let i = oy * d.outW + ox;
  dst[i] = vec4f(src[at][sub], src[at + plane][sub], src[at + plane * 2][sub], src[at + plane * 3][sub]);
  dst[i + d.outW * d.outH] = vec4f(src[at + plane * 4][sub], 0.0, 0.0, 0.0);
}`;

  // Back to full resolution: flow rescales and accumulates, the mask is replaced.
  const MERGE_SHADER = /* wgsl */ `
struct Dim { w: i32, h: i32, inW: i32, inH: i32, scale: f32, keep: f32 }
@group(0) @binding(0) var<uniform> d: Dim;
@group(0) @binding(1) var<storage, read> src: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> flow: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> mask: array<f32>;
@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3u) {
  let ox = i32(g.x);
  let oy = i32(g.y);
  if (ox >= d.w || oy >= d.h) { return; }
  let x = clamp((f32(ox) + 0.5) / d.scale - 0.5, 0.0, f32(d.inW - 1));
  let y = clamp((f32(oy) + 0.5) / d.scale - 0.5, 0.0, f32(d.inH - 1));
  let x0 = i32(x);
  let y0 = i32(y);
  let x1 = min(x0 + 1, d.inW - 1);
  let y1 = min(y0 + 1, d.inH - 1);
  let fx = x - f32(x0);
  let fy = y - f32(y0);
  let i00 = y0 * d.inW + x0;
  let i10 = y0 * d.inW + x1;
  let i01 = y1 * d.inW + x0;
  let i11 = y1 * d.inW + x1;
  let plane = d.inW * d.inH;
  let f = mix(mix(src[i00], src[i10], fx), mix(src[i01], src[i11], fx), fy);
  let m = mix(mix(src[i00 + plane].x, src[i10 + plane].x, fx), mix(src[i01 + plane].x, src[i11 + plane].x, fx), fy);
  let i = oy * d.w + ox;
  flow[i] = flow[i] * d.keep + f * d.scale;
  mask[i] = m;
}`;

  // Joins rgb and encoder features into the two-vec4 frame layout the other passes read.
  const PACK_SHADER = /* wgsl */ `
struct Dim { w: i32, h: i32 }
@group(0) @binding(0) var<uniform> d: Dim;
@group(0) @binding(1) var<storage, read> rgb: array<vec4f>;
@group(0) @binding(2) var<storage, read> feat: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> dst: array<vec4f>;
@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3u) {
  let x = i32(g.x);
  let y = i32(g.y);
  if (x >= d.w || y >= d.h) { return; }
  let i = y * d.w + x;
  dst[i * 2] = vec4f(rgb[i].xyz, feat[i].x);
  dst[i * 2 + 1] = vec4f(feat[i].yzw, 0.0);
}`;

  // Conv weights are [out, in, h, w] and transposed ones [in, out, h, w]; both become
  // mat4x4 blocks indexed by output group, kernel tap and input group.
  function packWeights(weights, entry, transpose) {
    const [d0, d1, kh, kw] = entry.shape;
    const inC = transpose ? d0 : d1;
    const outC = transpose ? d1 : d0;
    const ic = group4(inC);
    const oc = group4(outC);
    const taps = kh * kw;
    const out = new Float32Array(oc * taps * ic * 16);
    const base = entry.offset;
    for (let o = 0; o < outC; o++) {
      for (let c = 0; c < inC; c++) {
        const head = (transpose ? c * outC + o : o * inC + c) * taps;
        for (let tap = 0; tap < taps; tap++) {
          const m = (((o >> 2) * taps + tap) * ic + (c >> 2)) * 16;
          out[m + (c & 3) * 4 + (o & 3)] = weights[base + head + tap];
        }
      }
    }
    return out;
  }

  function packExtra(weights, index, biasName, betaName, channels) {
    const oc = group4(channels);
    const out = new Float32Array(oc * 4 * (betaName ? 2 : 1));
    const bias = index[biasName].offset;
    for (let c = 0; c < channels; c++) out[c] = weights[bias + c];
    if (betaName) {
      const beta = index[betaName].offset;
      for (let c = 0; c < channels; c++) out[oc * 4 + c] = weights[beta + c];
    }
    return out;
  }

  class RifeNetwork {
    constructor(device, weights, index) {
      this.device = device;
      this.pipelines = new Map();
      this.layers = { encode0: null, encode1: null, blocks: [] };
      const buffer = (data) => this.#upload(data);
      const conv = (name, activation, residual, stride) => {
        const entry = index[`${name}.weight`];
        const [outChannels, inChannels] = entry.shape;
        return {
          inChannels,
          outChannels,
          weight: buffer(packWeights(weights, entry, false)),
          extra: buffer(packExtra(weights, index, `${name}.bias`, residual ? name.replace(/\.conv$/, ".beta") : null, outChannels)),
          pipeline: this.#pipeline(`conv${inChannels}-${outChannels}-${stride}-${activation}-${residual}`, () =>
            convShader({ inChannels, outChannels, stride, activation, residual }),
          ),
        };
      };
      const deconv = (name) => {
        const entry = index[`${name}.weight`];
        const [inChannels, outChannels] = entry.shape;
        return {
          inChannels,
          outChannels,
          weight: buffer(packWeights(weights, entry, true)),
          extra: buffer(packExtra(weights, index, `${name}.bias`, null, outChannels)),
          pipeline: this.#pipeline(`deconv${inChannels}-${outChannels}`, () => transposeShader({ inChannels, outChannels })),
        };
      };
      this.layers.encode0 = conv("encode.0", false, false, 2);
      this.layers.encode1 = deconv("encode.1");
      BLOCK_CHANNELS.forEach((_, b) => {
        const prefix = `block${b}`;
        this.layers.blocks.push({
          conv0: conv(`${prefix}.conv0.0.0`, true, false, 2),
          conv1: conv(`${prefix}.conv0.1.0`, true, false, 2),
          residual: Array.from({ length: 8 }, (_, r) => conv(`${prefix}.convblock.${r}.conv`, true, true, 1)),
          last: deconv(`${prefix}.lastconv.0`),
        });
      });
      this.warpPipeline = this.#pipeline("warp", () => WARP_SHADER);
      this.prepFirst = this.#pipeline("prep-first", () => prepShader(true));
      this.prepNext = this.#pipeline("prep-next", () => prepShader(false));
      this.shufflePipeline = this.#pipeline("shuffle", () => SHUFFLE_SHADER);
      this.mergePipeline = this.#pipeline("merge", () => MERGE_SHADER);
      this.packPipeline = this.#pipeline("pack", () => PACK_SHADER);
      this.owned = [];
    }

    /** Allocates every tensor for one input size. Both sides must be multiples of 16 * 4 / scale. */
    configure(width, height, scale = 1) {
      this.release();
      this.width = width;
      this.height = height;
      this.scale = scale;
      const tensor = (w, h, channels) => this.#tensor(w * h * group4(channels) * 16);
      const uniform = (values) => this.#uniform(values);

      this.flow = tensor(width, height, 4);
      this.mask = this.#tensor(width * height * 4);
      this.warpedA = tensor(width, height, 8);
      this.warpedB = tensor(width, height, 8);
      const halfW = Math.ceil(width / 2);
      const halfH = Math.ceil(height / 2);
      this.encodeHalf = tensor(halfW, halfH, 16);
      this.encodeFeat = tensor(width, height, 4);
      const { encode0, encode1 } = this.layers;
      this.encodeGroups = {
        dim0: uniform([width, height, halfW, halfH]),
        dim1: uniform([halfW, halfH, width, height]),
      };
      void encode0;
      void encode1;
      this.packDim = uniform([width, height]);
      this.warpDim = uniform([width, height]);
      this.warpGroups = new Map();

      this.blocks = this.layers.blocks.map((layer, b) => {
        const blockScale = BLOCK_SCALES[b] / scale;
        const w = Math.floor(width / blockScale);
        const h = Math.floor(height / blockScale);
        const w2 = Math.ceil(w / 2);
        const h2 = Math.ceil(h / 2);
        const w4 = Math.ceil(w2 / 2);
        const h4 = Math.ceil(h2 / 2);
        const inChannels = layer.conv0.inChannels;
        const input = tensor(w, h, inChannels);
        const half = tensor(w2, h2, layer.conv0.outChannels);
        const c = layer.conv1.outChannels;
        const res0 = tensor(w4, h4, c);
        const res1 = tensor(w4, h4, c);
        const raw = tensor(w2, h2, 24);
        const shuffled = this.#tensor(w * h * 2 * 16);
        const group = (l, src, dst, dim) => this.#convGroup(l, src, dim, dst);
        const steps = [];
        const conv = (l, src, dst, dim, outW, outH, taps = 9) =>
          steps.push(this.#step(l.pipeline, group(l, src, dst, dim), outW, outH, outW * outH * group4(l.inChannels) * group4(l.outChannels) * taps * 16, group4(l.outChannels)));
        conv(layer.conv0, input, half, uniform([w, h, w2, h2]), w2, h2);
        conv(layer.conv1, half, res0, uniform([w2, h2, w4, h4]), w4, h4);
        const resDim = uniform([w4, h4, w4, h4]);
        layer.residual.forEach((r, i) => conv(r, i % 2 ? res1 : res0, i % 2 ? res0 : res1, resDim, w4, h4));
        conv(layer.last, res0, raw, uniform([w4, h4, w2, h2]), w2, h2, 4);
        steps.push(this.#step(this.shufflePipeline, this.#group(this.shufflePipeline, [uniform([w2, h2, w, h]), raw, shuffled]), w, h));
        steps.push(
          this.#step(
            this.mergePipeline,
            this.#group(this.mergePipeline, [this.#mixedUniform([width, height, w, h], [blockScale, b === 0 ? 0 : 1]), shuffled, this.flow, this.mask]),
            width,
            height,
          ),
        );
        const prepDim = this.#mixedUniform([width, height, w, h], [blockScale, 0.5]);
        return { steps, input, prepDim, w, h, first: b === 0, prepGroups: new Map() };
      });
    }

    /** Set the time the flow is anchored at. Takes effect from the next infer call. */
    setTime(t) {
      for (const block of this.blocks) {
        const scale = BLOCK_SCALES[this.blocks.indexOf(block)] / this.scale;
        this.device.queue.writeBuffer(block.prepDim, 16, new Float32Array([scale, t]));
      }
    }

    createFrameTensor() {
      return this.#tensor(this.width * this.height * FRAME_VEC4 * 16);
    }

    /** Steps turning rgb, a vec4f tensor of the input size, into a frame tensor with features. */
    planEncode(rgb, frame) {
      const { encode0, encode1 } = this.layers;
      const w = this.width;
      const h = this.height;
      const halfW = Math.ceil(w / 2);
      const halfH = Math.ceil(h / 2);
      const groups = (this.encodeBindGroups ??= new Map());
      let bound = groups.get(rgb);
      if (!bound) {
        bound = {
          a: this.#convGroup(encode0, rgb, this.encodeGroups.dim0, this.encodeHalf),
          b: this.#convGroup(encode1, this.encodeHalf, this.encodeGroups.dim1, this.encodeFeat),
        };
        groups.set(rgb, bound);
      }
      const pack = this.device.createBindGroup({
        layout: this.packPipeline.getBindGroupLayout(0),
        entries: [this.packDim, rgb, this.encodeFeat, frame].map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
      return [
        this.#step(encode0.pipeline, bound.a, halfW, halfH, halfW * halfH * group4(16) * 9 * 16, group4(16)),
        this.#step(encode1.pipeline, bound.b, w, h, w * h * group4(16) * 4 * 16, group4(4)),
        this.#step(this.packPipeline, pack, w, h),
      ];
    }

    /** Steps filling this.flow (0.5→0 xy, 0.5→1 zw, in pixels of the input) and this.mask (before sigmoid). */
    planInfer(frameA, frameB) {
      const { width: w, height: h } = this;
      const plan = [];
      for (const block of this.blocks) {
        const a = block.first ? frameA : this.warpedA;
        const b = block.first ? frameB : this.warpedB;
        const pipeline = block.first ? this.prepFirst : this.prepNext;
        let byB = block.prepGroups.get(a);
        if (!byB) block.prepGroups.set(a, (byB = new Map()));
        let prep = byB.get(b);
        if (!prep) {
          prep = this.#group(pipeline, block.first ? [block.prepDim, a, b, block.input] : [block.prepDim, a, b, block.input, this.flow, this.mask]);
          byB.set(b, prep);
        }
        plan.push(this.#step(pipeline, prep, block.w, block.h), ...block.steps, this.#warpStep(frameA, frameB, w, h));
      }
      return plan;
    }

    /** Runs steps inside a compute pass. Order matters and the caller decides where to cut. */
    run(pass, steps) {
      for (const { pipeline, group, x, y, z } of steps) {
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(x, y, z);
      }
    }

    release() {
      for (const buffer of this.owned) buffer.destroy();
      this.owned = [];
      this.encodeBindGroups = null;
    }

    dispose() {
      this.release();
      for (const buffer of this.weightBuffers ?? []) buffer.destroy();
    }

    #warpStep(frameA, frameB, w, h) {
      let byB = this.warpGroups.get(frameA);
      if (!byB) this.warpGroups.set(frameA, (byB = new Map()));
      let group = byB.get(frameB);
      if (!group) {
        group = this.#group(this.warpPipeline, [this.warpDim, this.flow, frameA, frameB, this.warpedA, this.warpedB]);
        byB.set(frameB, group);
      }
      return this.#step(this.warpPipeline, group, w, h);
    }

    // Cost is in multiply-adds, for slicing. Passes without a network layer are small by comparison.
    #step(pipeline, group, w, h, cost = w * h * 16, z = 1) {
      return { pipeline, group, x: Math.ceil(w / WORKGROUP), y: Math.ceil(h / WORKGROUP), z, cost };
    }

    #convGroup(layer, src, dim, dst) {
      return this.#group(layer.pipeline, [dim, src, layer.weight, layer.extra, dst]);
    }

    #group(pipeline, buffers) {
      return this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
    }

    #pipeline(key, source) {
      if (!this.pipelines.has(key)) {
        const module = this.device.createShaderModule({ code: source() });
        this.pipelines.set(key, this.device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } }));
      }
      return this.pipelines.get(key);
    }

    #upload(data) {
      const buffer = this.device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(buffer, 0, data);
      (this.weightBuffers ??= []).push(buffer);
      return buffer;
    }

    #tensor(size) {
      const buffer = this.device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
      this.owned.push(buffer);
      return buffer;
    }

    #uniform(values) {
      const buffer = this.device.createBuffer({ size: Math.ceil((values.length * 4) / 16) * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(buffer, 0, new Int32Array(values));
      this.owned.push(buffer);
      return buffer;
    }

    #mixedUniform(ints, floats) {
      const buffer = this.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(buffer, 0, new Int32Array(ints));
      this.device.queue.writeBuffer(buffer, ints.length * 4, new Float32Array(floats));
      this.owned.push(buffer);
      return buffer;
    }
  }

  globalThis.FrameBoostRife = { RifeNetwork };
})();
