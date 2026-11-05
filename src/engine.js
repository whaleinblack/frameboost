// FrameBoost engine: GPU optical flow and motion-compensated frame synthesis (WebGL2).
// Knows nothing about the page: feed it frames with timestamps, ask it to draw a time.
(() => {
  "use strict";

  const QUALITY_HEIGHT = { low: 270, medium: 360, high: 540 };
  const HISTORY = 3;
  const FRAME_SLOTS = HISTORY + 1; // one extra so a new frame never overwrites one still on screen
  const CUT_WIDTH = 32;
  const CUT_HEIGHT = 18;

  const VERTEX = `#version 300 es
out vec2 vUv;
void main() {
  vec2 p = vec2(float((gl_VertexID & 1) << 2), float((gl_VertexID & 2) << 1));
  vUv = p * 0.5;
  gl_Position = vec4(p - 1.0, 0.0, 1.0);
}`;

  const HEADER = `#version 300 es
precision highp float;
precision highp sampler2D;
#define TEX(s, uv) textureLod(s, uv, 0.0)
in vec2 vUv;
`;

  // Full-resolution frame to luma at the top flow level, minified through its mip chain.
  const LUMA_FROM_FRAME = `${HEADER}
uniform sampler2D uFrame;
uniform float uLod;
out vec4 color;
void main() {
  vec3 c = textureLod(uFrame, vUv, uLod).rgb;
  color = vec4(dot(c, vec3(0.2126, 0.7152, 0.0722)));
}`;

  const LUMA_DOWN = `${HEADER}
uniform sampler2D uSrc;
uniform vec2 uTexel;
out vec4 color;
void main() {
  vec2 d = uTexel * 0.5;
  color = vec4(0.25 * (
    TEX(uSrc, vUv + vec2(-d.x, -d.y)).r + TEX(uSrc, vUv + vec2(d.x, -d.y)).r +
    TEX(uSrc, vUv + vec2(-d.x, d.y)).r + TEX(uSrc, vUv + vec2(d.x, d.y)).r));
}`;

  // One level of the coarse-to-fine search. Flow is stored in uv units so it needs no rescale
  // between levels. Candidates come from the coarser level, its neighbours and the previous
  // frame pair, then the winner is refined locally. Cost is zero-mean SAD (survives fades)
  // plus a smoothness term that keeps flat areas from picking random vectors.
  const FLOW_SEARCH = `${HEADER}
uniform sampler2D uRef, uTgt, uPrior, uTemporal;
uniform vec2 uTexel;
uniform float uTemporalSign;
uniform float uHasTemporal;
uniform int uFullRadius;
uniform float uFinestStep;
out vec4 color;

vec2 gPrior;

float cost(vec2 flow) {
  float d[9];
  float mean = 0.0;
  int i = 0;
  for (int y = -1; y <= 1; y++) {
    for (int x = -1; x <= 1; x++) {
      vec2 o = vec2(float(x), float(y)) * 2.0 * uTexel;
      float v = TEX(uRef, vUv + o).r - TEX(uTgt, vUv + o + flow).r;
      d[i++] = v;
      mean += v;
    }
  }
  mean /= 9.0;
  float sad = 0.0;
  for (int k = 0; k < 9; k++) sad += abs(d[k] - mean);
  vec2 px = flow / uTexel;
  vec2 pxPrior = (flow - gPrior) / uTexel;
  return sad / 9.0 + 0.25 * abs(mean) + 0.0012 * length(pxPrior) + 0.0002 * length(px);
}

void main() {
  gPrior = TEX(uPrior, vUv).xy;
  vec2 best = vec2(0.0);
  float bestCost = cost(best);

  #define TRY(candidate) { vec2 c_ = (candidate); float k_ = cost(c_); if (k_ < bestCost) { bestCost = k_; best = c_; } }

  TRY(gPrior);
  for (int y = -1; y <= 1; y += 2) {
    for (int x = -1; x <= 1; x += 2) {
      TRY(TEX(uPrior, vUv + vec2(float(x), float(y)) * 3.0 * uTexel).xy);
    }
  }
  if (uHasTemporal > 0.5) {
    TRY(uTemporalSign * TEX(uTemporal, vUv).xy);
  }
  if (uFullRadius > 0) {
    for (int y = -uFullRadius; y <= uFullRadius; y++) {
      for (int x = -uFullRadius; x <= uFullRadius; x++) {
        TRY(vec2(float(x), float(y)) * uTexel);
      }
    }
  }
  for (float stride = 2.0; stride >= uFinestStep; stride *= 0.5) {
    vec2 center = best;
    for (int y = -1; y <= 1; y++) {
      for (int x = -1; x <= 1; x++) {
        if (x != 0 || y != 0) TRY(center + vec2(float(x), float(y)) * stride * uTexel);
      }
    }
  }
  color = vec4(best, 0.0, 1.0);
}`;

  // 3x3 vector median: removes isolated bad vectors without blurring motion edges.
  const FLOW_MEDIAN = `${HEADER}
uniform sampler2D uFlow;
uniform vec2 uTexel;
out vec4 color;
void main() {
  vec2 v[9];
  int i = 0;
  for (int y = -1; y <= 1; y++) {
    for (int x = -1; x <= 1; x++) {
      v[i++] = TEX(uFlow, vUv + vec2(float(x), float(y)) * uTexel).xy;
    }
  }
  float best = 1e9;
  vec2 pick = v[4];
  for (int a = 0; a < 9; a++) {
    float sum = 0.0;
    for (int b = 0; b < 9; b++) sum += length((v[a] - v[b]) / uTexel);
    if (sum < best) { best = sum; pick = v[a]; }
  }
  color = vec4(pick, 0.0, 1.0);
}`;

  // Joint bilateral: smooths the field but stops at luma edges, so motion stays on objects.
  const FLOW_SMOOTH = `${HEADER}
uniform sampler2D uFlow, uGuide;
uniform vec2 uTexel;
out vec4 color;
void main() {
  float g0 = TEX(uGuide, vUv).r;
  vec2 f0 = TEX(uFlow, vUv).xy;
  vec2 sum = vec2(0.0);
  float total = 0.0;
  for (int y = -2; y <= 2; y++) {
    for (int x = -2; x <= 2; x++) {
      vec2 o = vec2(float(x), float(y)) * uTexel;
      vec2 f = TEX(uFlow, vUv + o).xy;
      float g = TEX(uGuide, vUv + o).r;
      float dg = (g - g0) / 0.06;
      vec2 df = (f - f0) / uTexel / 3.0;
      float w = exp(-0.5 * (dot(o / uTexel, o / uTexel) / 2.25 + dg * dg + dot(df, df)));
      sum += f * w;
      total += w;
    }
  }
  color = vec4(sum / total, 0.0, 1.0);
}`;

  // Confidence: a vector is trusted when the opposite flow returns to the same place and the
  // two pixels look alike. Low confidence marks occlusions, where only one frame is usable.
  const FLOW_CONFIDENCE = `${HEADER}
uniform sampler2D uFlow, uOpposite, uRef, uTgt;
uniform vec2 uTexel;
out vec4 color;
void main() {
  vec2 f = TEX(uFlow, vUv).xy;
  vec2 back = TEX(uOpposite, vUv + f).xy;
  float loop = length((f + back) / uTexel);
  float photo = abs(TEX(uRef, vUv).r - TEX(uTgt, vUv + f).r);
  vec2 target = vUv + f;
  float inside = step(0.0, target.x) * step(target.x, 1.0) * step(0.0, target.y) * step(target.y, 1.0);
  float conf = exp(-loop * loop / 4.0) * (1.0 - smoothstep(0.06, 0.22, photo)) * inside;
  color = vec4(f, conf, smoothstep(2.0, 6.0, loop));
}`;

  // Share of the picture the flow cannot explain. Averaged via mip chain, read without readback.
  // Across a hard cut block matching still finds look-alike patches, but they do not agree
  // between directions. Only the round trip counts: sharp detail lowers photometric confidence
  // under fast motion, and that must not read as a cut.
  const CUT_MEASURE = `${HEADER}
uniform sampler2D uForward, uBackward;
out vec4 color;
void main() {
  color = vec4(0.5 * (TEX(uForward, vUv).a + TEX(uBackward, vUv).a));
}`;

  // Synthesis. Each frame is warped to time t by solving x = p + t * F(p) for the source pixel p
  // with a few fixed-point steps, then the two warps blend by temporal distance and confidence.
  const SYNTHESIS = `${HEADER}
uniform sampler2D uFrameA, uFrameB, uFlowA, uFlowB, uCut;
uniform float uT, uLod;
uniform vec4 uCrop;
out vec4 color;

vec3 warp(vec2 uv, float t, sampler2D flow, out vec2 src) {
  vec3 f = TEX(flow, uv).xyz;
  for (int i = 0; i < 4; i++) f = TEX(flow, uv - t * f.xy).xyz;
  src = uv - t * f.xy;
  return f;
}

void main() {
  vec2 uv = vec2(vUv.x, 1.0 - vUv.y) * uCrop.zw + uCrop.xy;
  vec2 pa, pb;
  float ca = warp(uv, uT, uFlowA, pa).z;
  float cb = warp(uv, 1.0 - uT, uFlowB, pb).z;
  vec3 a = textureLod(uFrameA, pa, uLod).rgb;
  vec3 b = textureLod(uFrameB, pb, uLod).rgb;
  float wa = (1.0 - uT) * (ca + 0.02);
  float wb = uT * (cb + 0.02);
  vec3 blended = (a * wa + b * wb) / (wa + wb);

  float unexplained = textureLod(uCut, vec2(0.5), 16.0).r;
  float cut = smoothstep(0.35, 0.55, unexplained);
  vec3 nearest = uT < 0.5 ? textureLod(uFrameA, uv, uLod).rgb : textureLod(uFrameB, uv, uLod).rgb;
  color = vec4(mix(blended, nearest, cut), 1.0);
}`;

  class FlowInterpolator {
    constructor(canvas, options = {}) {
      this.canvas = canvas;
      this.quality = options.quality || "medium";
      this.frames = [];
      this.sourceWidth = 0;
      this.sourceHeight = 0;
      const gl = canvas.getContext("webgl2", {
        alpha: true,
        antialias: false,
        depth: false,
        stencil: false,
        powerPreference: "high-performance",
      });
      if (!gl) throw new Error("WebGL2 不可用");
      if (!gl.getExtension("EXT_color_buffer_float")) throw new Error("显卡不支持浮点渲染目标");
      this.gl = gl;
      this.vao = gl.createVertexArray();
      this.fbo = gl.createFramebuffer();
      this.programs = {
        lumaFromFrame: this.#program(LUMA_FROM_FRAME),
        lumaDown: this.#program(LUMA_DOWN),
        search: this.#program(FLOW_SEARCH),
        median: this.#program(FLOW_MEDIAN),
        smooth: this.#program(FLOW_SMOOTH),
        confidence: this.#program(FLOW_CONFIDENCE),
        cut: this.#program(CUT_MEASURE),
        synthesis: this.#program(SYNTHESIS),
      };
      this.zeroFlow = this.#texture(1, 1, gl.RGBA16F, gl.NEAREST);
    }

    /** Forget all history, e.g. after a seek, so no flow crosses the discontinuity. */
    reset() {
      this.frames.length = 0;
    }

    setQuality(quality) {
      if (quality === this.quality) return;
      this.quality = quality;
      this.#release();
      this.sourceWidth = 0;
      this.reset();
    }

    /** Upload a frame and compute the flow to the previous one. Throws SecurityError if tainted. */
    pushFrame(source, width, height, mediaTime) {
      const gl = this.gl;
      if (width !== this.sourceWidth || height !== this.sourceHeight) this.#allocate(width, height);

      const slot = this.slots[this.nextSlot];
      gl.bindTexture(gl.TEXTURE_2D, slot.frame);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, source);
      this.nextSlot = (this.nextSlot + 1) % FRAME_SLOTS;

      gl.generateMipmap(gl.TEXTURE_2D);
      gl.bindVertexArray(this.vao);
      this.#buildLumaPyramid(slot);

      const previous = this.frames[this.frames.length - 1];
      slot.mediaTime = mediaTime;
      slot.pair = null;
      if (previous) this.#computeFlow(previous, slot);
      this.frames.push(slot);
      if (this.frames.length > HISTORY) this.frames.shift();
      gl.flush();
    }

    /**
     * Draw the picture at mediaTime into the given sub-rectangle of the canvas.
     * Holds the first or last frame when the time falls outside the buffered pair.
     */
    render(mediaTime, viewport, crop) {
      if (this.frames.length < 2) return false;
      let i = 1;
      while (i < this.frames.length - 1 && mediaTime >= this.frames[i].mediaTime) i++;
      const a = this.frames[i - 1];
      const b = this.frames[i];
      const t = Math.min(1, Math.max(0, (mediaTime - a.mediaTime) / (b.mediaTime - a.mediaTime)));
      this.#draw(a, b, t, viewport, crop);
      return true;
    }

    get ready() {
      return this.frames.length >= 2;
    }

    dispose() {
      this.#release();
      this.gl.getExtension("WEBGL_lose_context")?.loseContext();
    }

    #draw(a, b, t, viewport, crop) {
      const gl = this.gl;
      const pair = b.pair;
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.bindVertexArray(this.vao);
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);
      const p = this.programs.synthesis;
      gl.useProgram(p.program);
      this.#bind(p, "uFrameA", a.frame, 0);
      this.#bind(p, "uFrameB", b.frame, 1);
      this.#bind(p, "uFlowA", pair.forward, 2);
      this.#bind(p, "uFlowB", pair.backward, 3);
      this.#bind(p, "uCut", pair.cut, 4);
      gl.uniform1f(p.uniforms.uT, t);
      gl.uniform1f(p.uniforms.uLod, Math.max(0, Math.log2(this.sourceHeight / Math.max(1, viewport.height))));
      gl.uniform4f(p.uniforms.uCrop, crop.x, crop.y, crop.width, crop.height);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    #buildLumaPyramid(slot) {
      const gl = this.gl;
      const top = this.levels[0];
      let p = this.programs.lumaFromFrame;
      this.#target(slot.luma[0], top.width, top.height);
      gl.useProgram(p.program);
      this.#bind(p, "uFrame", slot.frame, 0);
      gl.uniform1f(p.uniforms.uLod, Math.max(0, Math.log2(this.sourceHeight / top.height)));
      gl.drawArrays(gl.TRIANGLES, 0, 3);

      p = this.programs.lumaDown;
      gl.useProgram(p.program);
      for (let i = 1; i < this.levels.length; i++) {
        const level = this.levels[i];
        this.#target(slot.luma[i], level.width, level.height);
        this.#bind(p, "uSrc", slot.luma[i - 1], 0);
        gl.uniform2f(p.uniforms.uTexel, 1 / this.levels[i - 1].width, 1 / this.levels[i - 1].height);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
      }
    }

    #computeFlow(older, newer) {
      const pair = this.pairs[this.nextPair];
      this.nextPair = (this.nextPair + 1) % this.pairs.length;
      const temporal = older.pair?.forward ?? null;

      const forward = this.#flowDirection(older, newer, temporal, 1, this.scratch.forward);
      const backward = this.#flowDirection(newer, older, temporal, -1, this.scratch.backward);
      this.#confidence(forward, backward, older, newer, pair.forward);
      this.#confidence(backward, forward, newer, older, pair.backward);
      this.#measureCut(pair, pair.cut);
      newer.pair = pair;
    }

    #flowDirection(ref, tgt, temporal, temporalSign, finest) {
      const gl = this.gl;
      const search = this.programs.search;
      const median = this.programs.median;
      let prior = this.zeroFlow;
      for (let i = this.levels.length - 1; i >= 0; i--) {
        const level = this.levels[i];
        const texelX = 1 / level.width;
        const texelY = 1 / level.height;

        this.#target(level.raw, level.width, level.height);
        gl.useProgram(search.program);
        this.#bind(search, "uRef", ref.luma[i], 0);
        this.#bind(search, "uTgt", tgt.luma[i], 1);
        this.#bind(search, "uPrior", prior, 2);
        this.#bind(search, "uTemporal", temporal ?? this.zeroFlow, 3);
        gl.uniform2f(search.uniforms.uTexel, texelX, texelY);
        gl.uniform1f(search.uniforms.uTemporalSign, temporalSign);
        gl.uniform1f(search.uniforms.uHasTemporal, temporal ? 1 : 0);
        gl.uniform1i(search.uniforms.uFullRadius, i === this.levels.length - 1 ? 4 : 0);
        gl.uniform1f(search.uniforms.uFinestStep, i === 0 ? 0.5 : 1);
        gl.drawArrays(gl.TRIANGLES, 0, 3);

        this.#target(level.filtered, level.width, level.height);
        gl.useProgram(median.program);
        this.#bind(median, "uFlow", level.raw, 0);
        gl.uniform2f(median.uniforms.uTexel, texelX, texelY);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        prior = level.filtered;
      }

      const top = this.levels[0];
      const smooth = this.programs.smooth;
      this.#target(finest, top.width, top.height);
      gl.useProgram(smooth.program);
      this.#bind(smooth, "uFlow", top.filtered, 0);
      this.#bind(smooth, "uGuide", ref.luma[0], 1);
      gl.uniform2f(smooth.uniforms.uTexel, 1 / top.width, 1 / top.height);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      return finest;
    }

    #confidence(flow, opposite, ref, tgt, output) {
      const gl = this.gl;
      const top = this.levels[0];
      const p = this.programs.confidence;
      this.#target(output, top.width, top.height);
      gl.useProgram(p.program);
      this.#bind(p, "uFlow", flow, 0);
      this.#bind(p, "uOpposite", opposite, 1);
      this.#bind(p, "uRef", ref.luma[0], 2);
      this.#bind(p, "uTgt", tgt.luma[0], 3);
      gl.uniform2f(p.uniforms.uTexel, 1 / top.width, 1 / top.height);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    #measureCut(pair, output) {
      const gl = this.gl;
      const p = this.programs.cut;
      this.#target(output, CUT_WIDTH, CUT_HEIGHT);
      gl.useProgram(p.program);
      this.#bind(p, "uForward", pair.forward, 0);
      this.#bind(p, "uBackward", pair.backward, 1);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.bindTexture(gl.TEXTURE_2D, output);
      gl.generateMipmap(gl.TEXTURE_2D);
    }

    #allocate(width, height) {
      const gl = this.gl;
      this.#release();
      this.sourceWidth = width;
      this.sourceHeight = height;

      const topHeight = Math.min(height, QUALITY_HEIGHT[this.quality] ?? QUALITY_HEIGHT.medium);
      const topWidth = Math.max(16, Math.round((width * topHeight) / height));
      this.levels = [];
      for (let w = topWidth, h = topHeight; Math.min(w, h) >= 16; w = Math.ceil(w / 2), h = Math.ceil(h / 2)) {
        this.levels.push({
          width: w,
          height: h,
          raw: this.#texture(w, h, gl.RGBA16F, gl.LINEAR),
          filtered: this.#texture(w, h, gl.RGBA16F, gl.LINEAR),
        });
      }

      const mipCount = Math.floor(Math.log2(Math.max(width, height))) + 1;
      this.slots = Array.from({ length: FRAME_SLOTS }, () => {
        const frame = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, frame);
        gl.texStorage2D(gl.TEXTURE_2D, mipCount, gl.RGBA8, width, height);
        this.#samplerState(gl.LINEAR_MIPMAP_LINEAR);
        const luma = this.levels.map((l) => this.#texture(l.width, l.height, gl.R8, gl.LINEAR));
        return { frame, luma, mediaTime: 0, pair: null };
      });
      this.nextSlot = 0;

      const top = this.levels[0];
      this.scratch = {
        forward: this.#texture(top.width, top.height, gl.RGBA16F, gl.LINEAR),
        backward: this.#texture(top.width, top.height, gl.RGBA16F, gl.LINEAR),
      };
      this.pairs = Array.from({ length: HISTORY }, () => {
        const cut = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, cut);
        gl.texStorage2D(gl.TEXTURE_2D, Math.floor(Math.log2(CUT_WIDTH)) + 1, gl.R8, CUT_WIDTH, CUT_HEIGHT);
        this.#samplerState(gl.LINEAR_MIPMAP_LINEAR);
        return {
          forward: this.#texture(top.width, top.height, gl.RGBA16F, gl.LINEAR),
          backward: this.#texture(top.width, top.height, gl.RGBA16F, gl.LINEAR),
          cut,
        };
      });
      this.nextPair = 0;
      this.reset();
    }

    #release() {
      const gl = this.gl;
      if (!this.slots) return;
      for (const level of this.levels) {
        gl.deleteTexture(level.raw);
        gl.deleteTexture(level.filtered);
      }
      for (const slot of this.slots) {
        gl.deleteTexture(slot.frame);
        slot.luma.forEach((t) => gl.deleteTexture(t));
      }
      for (const pair of this.pairs) {
        gl.deleteTexture(pair.forward);
        gl.deleteTexture(pair.backward);
        gl.deleteTexture(pair.cut);
      }
      gl.deleteTexture(this.scratch.forward);
      gl.deleteTexture(this.scratch.backward);
      this.slots = null;
    }

    #texture(width, height, format, filter) {
      const gl = this.gl;
      const texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texStorage2D(gl.TEXTURE_2D, 1, format, width, height);
      this.#samplerState(filter);
      return texture;
    }

    #samplerState(minFilter) {
      const gl = this.gl;
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, minFilter);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    }

    #target(texture, width, height) {
      const gl = this.gl;
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
      gl.viewport(0, 0, width, height);
    }

    #bind(program, name, texture, unit) {
      const gl = this.gl;
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.uniform1i(program.uniforms[name], unit);
    }

    #program(fragmentSource) {
      const gl = this.gl;
      const compile = (type, source) => {
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
          throw new Error(`着色器编译失败: ${gl.getShaderInfoLog(shader)}`);
        }
        return shader;
      };
      const program = gl.createProgram();
      gl.attachShader(program, compile(gl.VERTEX_SHADER, VERTEX));
      gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fragmentSource));
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        throw new Error(`着色器链接失败: ${gl.getProgramInfoLog(program)}`);
      }
      const uniforms = {};
      const count = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
      for (let i = 0; i < count; i++) {
        const { name } = gl.getActiveUniform(program, i);
        uniforms[name] = gl.getUniformLocation(program, name);
      }
      return { program, uniforms };
    }
  }

  globalThis.FrameBoostEngine = { FlowInterpolator };
})();
