// FrameBoost flow repair: fixes what a coarse-to-fine network gets wrong on periodic patterns.
// Shrinking stripes turns them grey, so the network's coarse levels see no motion and its fine levels
// settle on whichever repeat is nearby. Neighbouring areas disagree, which tears the picture.
// Here FFT phase correlation on tiles of the source proposes motions, since it sees patterns well.
// Among those that match, the smallest wins: outside a pattern only the true motion matches at all,
// and within one the smallest is the likeliest. A pixel only leaves the network's flow if it fails,
// or if it sits on a pattern strong enough to be periodic and the repair matches as well.
(() => {
  "use strict";

  const TILE = 256;
  const PEAKS = 4;
  const SELECT_CELL = 8;
  const GATE_CELL = 4;
  const PRIOR = 0.0025; // cost per pixel of motion
  const FLOOR_MAX = 0.035; // a match this good is as good as a perfect one
  const FLOOR_CONTRAST = 0.12; // ...scaled down with local contrast, where anything matches
  const STRONG_FROM = 0.12; // local contrast from which a pattern counts as strong
  const STRONG_RANGE = 0.06;
  // A periodic pattern correlates almost as well at its next repeat as at the true motion. A flow the
  // network got wrong for any other reason, such as spokes crossing a shoe, must not be replaced:
  // whole-tile motions cannot describe that, and swapping them in tears the picture.
  const PERIODIC_RATIO = 0.6;
  // Straight stripes get a further rule: if several repeats match, the smallest wins even over a flow
  // that also matches. Curved patterns do not move as whole tiles, so there it would do harm.
  const STRICT_RATIO = 0.85;
  const SAMPLE_COST = 1.5e6 / 3000; // in multiply-adds, so slicing can weigh it against the network
  const FIT_GRID = [96, 54];
  const GLOBAL_CELL = 8; // in 1080p pixels
  const GLOBAL_TOLERANCE = 0.012; // how much worse the fitted motion may match and still be taken
  const GLOBAL_MIN_SHARE = 0.8; // of the picture the fit must explain: a shot that moves as a whole, not a pan over moving people

  const LUMA = "const LUMA = vec3f(0.2126, 0.7152, 0.0722);";

  // Luma of both frames as one complex signal: one FFT then gives both spectra.
  const LOAD_SHADER = /* wgsl */ `
${LUMA}
struct P { offset: vec2f, stride: f32, step: f32, size: vec2f, tiles: vec2i }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var a: texture_2d<f32>;
@group(0) @binding(2) var b: texture_2d<f32>;
@group(0) @binding(3) var linear: sampler;
@group(0) @binding(4) var<storage, read_write> data: array<vec2f>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let t = i32(g.z);
  let origin = p.offset + vec2f(f32(t % p.tiles.x), f32(t / p.tiles.x)) * p.stride * p.step;
  let uv = (origin + (vec2f(g.xy) + 0.5) * p.step) / p.size;
  let w = sin(3.14159265 * (vec2f(g.xy) + 0.5) / ${TILE}.0);
  let window = w.x * w.x * w.y * w.y;
  let la = dot(textureSampleLevel(a, linear, uv, 0.0).rgb, LUMA);
  let lb = dot(textureSampleLevel(b, linear, uv, 0.0).rgb, LUMA);
  data[t * ${TILE * TILE} + i32(g.y) * ${TILE} + i32(g.x)] = vec2f(la, lb) * window;
}`;

  // One thread per line, so the whole picture's tiles spread over the GPU.
  const fftShader = (vertical, inverse) => /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> data: array<vec2f>;
@group(0) @binding(1) var<storage, read> twiddle: array<vec2f>;
var<private> v: array<vec2f, ${TILE}>;
@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let line = i32(g.x);
  if (line >= i32(arrayLength(&data)) / ${TILE}) { return; }
  let t = line / ${TILE};
  let n = line % ${TILE};
  let base = t * ${TILE * TILE} + ${vertical ? "n" : `n * ${TILE}`};
  let at = ${vertical ? TILE : 1};
  for (var i = 0; i < ${TILE}; i++) {
    v[i32(reverseBits(u32(i)) >> ${32 - Math.log2(TILE)}u)] = data[base + i * at];
  }
  for (var m = 2; m <= ${TILE}; m *= 2) {
    let span = m / 2;
    let stepT = ${TILE} / m;
    for (var k = 0; k < ${TILE}; k += m) {
      for (var j = 0; j < span; j++) {
        var w = twiddle[j * stepT];
        ${inverse ? "w.y = -w.y;" : ""}
        let u = v[k + j];
        let c = v[k + j + span];
        let z = vec2f(c.x * w.x - c.y * w.y, c.x * w.y + c.y * w.x);
        v[k + j] = u + z;
        v[k + j + span] = u - z;
      }
    }
  }
  for (var i = 0; i < ${TILE}; i++) { data[base + i * at] = v[i]; }
}`;

  // Normalised cross power of the two spectra, separated from the single complex transform.
  const CROSS_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> z: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> result: array<vec2f>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let base = i32(g.z) * ${TILE * TILE};
  let x = i32(g.x);
  let y = i32(g.y);
  let i = base + y * ${TILE} + x;
  // The window leaves its energy at the lowest frequencies, which say nothing about motion.
  let fx = min(x, ${TILE} - x);
  let fy = min(y, ${TILE} - y);
  if (fx <= 1 && fy <= 1) {
    result[i] = vec2f(0.0);
    return;
  }
  let p = z[i];
  let q = z[base + ((${TILE} - y) % ${TILE}) * ${TILE} + ((${TILE} - x) % ${TILE})];
  let qc = vec2f(q.x, -q.y);
  let fa = (p + qc) * 0.5;
  let d = (p - qc) * 0.5;
  let fb = vec2f(d.y, -d.x);
  let cross = vec2f(fb.x * fa.x + fb.y * fa.y, fb.y * fa.x - fb.x * fa.y);
  result[i] = cross / (length(cross) + 1e-3);
}`;

  // The strongest local maxima of each line, one thread per line: scanning a whole tile in one
  // thread would leave the GPU almost idle. Stripes put several peaks on the same line.
  const ROW_PEAK_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> corr: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> rows: array<vec4f>;
@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let line = i32(g.x);
  if (line >= i32(arrayLength(&rows)) / ${PEAKS}) { return; }
  let base = line * ${TILE};
  var best = array<vec2f, ${PEAKS}>();
  var left = corr[base + ${TILE} - 1].x;
  var here = corr[base].x;
  for (var x = 0; x < ${TILE}; x++) {
    let right = corr[base + (x + 1) % ${TILE}].x;
    if (here > 0.0 && here > left && here >= right) {
      var value = vec2f(f32(x), here);
      for (var k = 0; k < ${PEAKS}; k++) {
        if (value.y > best[k].y) {
          let held = best[k];
          best[k] = value;
          value = held;
        }
      }
    }
    left = here;
    here = right;
  }
  for (var k = 0; k < ${PEAKS}; k++) {
    rows[line * ${PEAKS} + k] = vec4f(best[k], 0.0, 0.0);
  }
}`;

  const PEAK_SHADER = /* wgsl */ `
struct P { offset: vec2f, stride: f32, step: f32, size: vec2f, tiles: vec2i }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> corr: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> peaks: array<vec4f>;
@group(0) @binding(3) var<storage, read> rows: array<vec4f>;
fn at(t: i32, x: i32, y: i32) -> f32 {
  return corr[t * ${TILE * TILE} + ((y + ${TILE}) % ${TILE}) * ${TILE} + ((x + ${TILE}) % ${TILE})].x;
}
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let t = i32(g.x);
  var found: array<vec2i, ${PEAKS}>;
  for (var n = 0; n < ${PEAKS}; n++) {
    var best = 0.0;
    var pos = vec2i(0);
    for (var y = 0; y < ${TILE}; y++) {
      for (var r = 0; r < ${PEAKS}; r++) {
        let row = rows[(t * ${TILE} + y) * ${PEAKS} + r];
        let c = row.y;
        if (c <= best) { break; }
        let x = i32(row.x);
        var taken = false;
        for (var k = 0; k < n; k++) {
          let d = abs(vec2i(x, y) - found[k]);
          let w = min(d, vec2i(${TILE}) - d);
          if (w.x <= 3 && w.y <= 3) { taken = true; }
        }
        if (!taken) {
          best = c;
          pos = vec2i(x, y);
        }
      }
    }
    found[n] = pos;
    if (best <= 0.0) {
      peaks[t * ${PEAKS} + n] = vec4f(0.0);
      continue;
    }
    // Sub-pixel by parabola; the correlation wraps, and past half a tile the motion is negative.
    let c0 = at(t, pos.x, pos.y);
    let lx = at(t, pos.x - 1, pos.y);
    let rx = at(t, pos.x + 1, pos.y);
    let ly = at(t, pos.x, pos.y - 1);
    let ry = at(t, pos.x, pos.y + 1);
    let dx = 0.5 * (lx - rx) / (lx - 2.0 * c0 + rx - 1e-9);
    let dy = 0.5 * (ly - ry) / (ly - 2.0 * c0 + ry - 1e-9);
    let sx = select(pos.x, pos.x - ${TILE}, pos.x >= ${TILE / 2});
    let sy = select(pos.y, pos.y - ${TILE}, pos.y >= ${TILE / 2});
    peaks[t * ${PEAKS} + n] = vec4f((vec2f(f32(sx), f32(sy)) + vec2f(dx, dy)) * p.step, best, 1.0);
  }
}`;

  const SELECT_SHADER = /* wgsl */ `
${LUMA}
struct P { offset: vec2f, stride: f32, step: f32, size: vec2f, tiles: vec2i, cells: vec2i }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var a: texture_2d<f32>;
@group(0) @binding(2) var b: texture_2d<f32>;
@group(0) @binding(3) var linear: sampler;
@group(0) @binding(4) var flow: texture_2d<f32>;
@group(0) @binding(5) var<storage, read> peaks: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> cells: array<vec4f>;

fn la(x: vec2f) -> f32 { return dot(textureSampleLevel(a, linear, x / p.size, 0.0).rgb, LUMA); }
fn lb(x: vec2f) -> f32 { return dot(textureSampleLevel(b, linear, x / p.size, 0.0).rgb, LUMA); }

// Windows may shift, so a cell at an object's edge can decide from the object's side.
fn matchCost(c: vec2f, v: vec2f) -> f32 {
  var q = array<f32, 5>(0.0, 0.0, 0.0, 0.0, 0.0);
  for (var y = -4; y <= 4; y++) {
    for (var x = -4; x <= 4; x++) {
      let s = c + vec2f(f32(x), f32(y)) * 7.0;
      let e = abs(la(s - v * 0.5) - lb(s + v * 0.5));
      if (x <= 0 && y <= 0) { q[0] += e; }
      if (x >= 0 && y <= 0) { q[1] += e; }
      if (x <= 0 && y >= 0) { q[2] += e; }
      if (x >= 0 && y >= 0) { q[3] += e; }
      if (abs(x) <= 2 && abs(y) <= 2) { q[4] += e; }
    }
  }
  return min(min(min(q[0], q[1]), min(q[2], q[3])), q[4]) / 25.0;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let cell = vec2i(g.xy);
  if (cell.x >= p.cells.x || cell.y >= p.cells.y) { return; }
  let c = (vec2f(cell) + 0.5) * ${SELECT_CELL}.0;
  let f = textureSampleLevel(flow, linear, c / p.size, 0.0) * vec4f(p.size, p.size);

  // Contrast, and whether the network already explains this area.
  var mean = 0.0;
  for (var y = -2; y <= 2; y++) {
    for (var x = -2; x <= 2; x++) { mean += la(c + vec2f(f32(x), f32(y)) * 7.0); }
  }
  mean /= 25.0;
  var contrast = 0.0;
  var warp = 0.0;
  for (var y = -2; y <= 2; y++) {
    for (var x = -2; x <= 2; x++) {
      let s = c + vec2f(f32(x), f32(y)) * 7.0;
      contrast += abs(la(s) - mean);
      warp += abs(la(s + f.xy) - lb(s + f.zw));
    }
  }
  contrast /= 25.0;
  warp /= 25.0;
  let index = cell.y * p.cells.x + cell.x;
  let vRife = f.zw - f.xy;
  let tile = (c - p.offset) / (p.stride * p.step) - 0.5;
  let t0 = vec2i(floor(tile));
  // Every tile around the cell must agree: one patterned tile nearby proves nothing about this spot.
  var ratio = 1.0;
  for (var ty = 0; ty <= 1; ty++) {
    for (var tx = 0; tx <= 1; tx++) {
      let t = clamp(t0 + vec2i(tx, ty), vec2i(0), p.tiles - 1);
      let first = peaks[(t.y * p.tiles.x + t.x) * ${PEAKS}];
      let second = peaks[(t.y * p.tiles.x + t.x) * ${PEAKS} + 1];
      ratio = min(ratio, select(0.0, second.z / first.z, first.z > 0.0));
    }
  }
  if (ratio < ${PERIODIC_RATIO} || contrast < ${STRONG_FROM} || (warp < 0.02 && ratio < ${STRICT_RATIO})) {
    cells[index] = vec4f(vRife, contrast, 0.0);
    return;
  }

  let floorCost = min(${FLOOR_MAX}, ${FLOOR_CONTRAST} * contrast);
  var best = vec2f(0.0);
  var bestScore = max(matchCost(c, best), floorCost);
  for (var ty = 0; ty <= 1; ty++) {
    for (var tx = 0; tx <= 1; tx++) {
      let t = clamp(t0 + vec2i(tx, ty), vec2i(0), p.tiles - 1);
      for (var n = 0; n < ${PEAKS}; n++) {
        let peak = peaks[(t.y * p.tiles.x + t.x) * ${PEAKS} + n];
        if (peak.w < 0.5) { continue; }
        let size = ${PRIOR} * length(peak.xy);
        if (size + floorCost >= bestScore) { continue; }
        let score = max(matchCost(c, peak.xy), floorCost) + size;
        if (score < bestScore) {
          bestScore = score;
          best = peak.xy;
        }
      }
    }
  }
  cells[index] = vec4f(best, contrast, ratio);
}`;

  // Cells on a pattern can pick different repeats, each of which matches. Their average matches
  // neither, so the neighbourhood votes first and only motions close to the winner are averaged.
  const SMOOTH_SHADER = /* wgsl */ `
struct P { cells: vec2i }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4f>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let cell = vec2i(g.xy);
  if (cell.x >= p.cells.x || cell.y >= p.cells.y) { return; }
  var votes = array<vec2f, 25>();
  var n = 0;
  for (var y = -2; y <= 2; y++) {
    for (var x = -2; x <= 2; x++) {
      let q = clamp(cell + vec2i(x, y), vec2i(0), p.cells - 1);
      votes[n] = src[q.y * p.cells.x + q.x].xy;
      n++;
    }
  }
  var winner = votes[12];
  var most = 0;
  for (var a = 0; a < 25; a++) {
    var count = 0;
    for (var b = 0; b < 25; b++) {
      if (distance(votes[a], votes[b]) < 1.5) { count++; }
    }
    if (count > most) {
      most = count;
      winner = votes[a];
    }
  }
  var sum = vec2f(0.0);
  var total = 0.0;
  for (var a = 0; a < 25; a++) {
    if (distance(votes[a], winner) < 1.5) {
      sum += votes[a];
      total += 1.0;
    }
  }
  let i = cell.y * p.cells.x + cell.x;
  dst[i] = vec4f(sum / total, src[i].zw);
}`;

  const GATE_SHADER = /* wgsl */ `
${LUMA}
struct P { size: vec2f, cells: vec2i, gate: vec2i }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var a: texture_2d<f32>;
@group(0) @binding(2) var b: texture_2d<f32>;
@group(0) @binding(3) var linear: sampler;
@group(0) @binding(4) var flow: texture_2d<f32>;
@group(0) @binding(5) var<storage, read> cells: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> amounts: array<f32>;

fn la(x: vec2f) -> f32 { return dot(textureSampleLevel(a, linear, x / p.size, 0.0).rgb, LUMA); }
fn lb(x: vec2f) -> f32 { return dot(textureSampleLevel(b, linear, x / p.size, 0.0).rgb, LUMA); }

fn cell(c: vec2f) -> vec4f {
  let g = clamp(c / ${SELECT_CELL}.0 - 0.5, vec2f(0.0), vec2f(p.cells - 1));
  let i = vec2i(g);
  let j = min(i + 1, p.cells - 1);
  let k = g - vec2f(i);
  let top = mix(cells[i.y * p.cells.x + i.x], cells[i.y * p.cells.x + j.x], k.x);
  let bottom = mix(cells[j.y * p.cells.x + i.x], cells[j.y * p.cells.x + j.x], k.x);
  var result = mix(top, bottom, k.y);
  let nearest = cells[select(i.y, j.y, k.y > 0.5) * p.cells.x + select(i.x, j.x, k.x > 0.5)];
  // Between two different repeats a blend matches neither.
  if (distance(result.xy, nearest.xy) > 1.5) { result = vec4f(nearest.xy, result.zw); }
  return result;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let id = vec2i(g.xy);
  if (id.x >= p.gate.x || id.y >= p.gate.y) { return; }
  let i = id.y * p.gate.x + id.x;
  let c = (vec2f(id) + 0.5) * ${GATE_CELL}.0;
  let found = cell(c);
  if (found.w < 0.01) {
    amounts[i] = 0.0;
    return;
  }
  let f = textureSampleLevel(flow, linear, c / p.size, 0.0) * vec4f(p.size, p.size);
  let vRife = f.zw - f.xy;
  let v = found.xy;
  var qFused = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
  var qRife = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
  var qWarp = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
  for (var y = -2; y <= 2; y++) {
    for (var x = -2; x <= 2; x++) {
      let s = c + vec2f(f32(x), f32(y)) * 3.0;
      let eFused = abs(la(s - v * 0.5) - lb(s + v * 0.5));
      let eRife = abs(la(s - vRife * 0.5) - lb(s + vRife * 0.5));
      let eWarp = abs(la(s + f.xy) - lb(s + f.zw));
      for (var q = 0; q < 4; q++) {
        let inX = select(x <= 0, x >= 0, (q & 1) == 1);
        let inY = select(y <= 0, y >= 0, (q & 2) == 2);
        if (inX && inY) {
          qFused[q] += eFused;
          qRife[q] += eRife;
          qWarp[q] += eWarp;
        }
      }
    }
  }
  let fused = min(min(qFused[0], qFused[1]), min(qFused[2], qFused[3])) / 9.0;
  let rife = min(min(qRife[0], qRife[1]), min(qRife[2], qRife[3])) / 9.0;
  let warp = max(rife, min(min(qWarp[0], qWarp[1]), min(qWarp[2], qWarp[3])) / 9.0);
  let contrast = found.z;
  let floorCost = min(${FLOOR_MAX}, ${FLOOR_CONTRAST} * contrast);
  let fusedScore = max(fused, floorCost) + ${PRIOR} * length(v);
  let rifeScore = max(warp, floorCost) + ${PRIOR} * length(vRife);
  let better = clamp((rifeScore - fusedScore - 0.004) / 0.01, 0.0, 1.0) * smoothstep(${STRICT_RATIO} - 0.05, ${STRICT_RATIO}, found.w);
  let fails = clamp((warp - fused - 0.02) / 0.03, 0.0, 1.0);
  let strong = clamp((contrast - ${STRONG_FROM}) / ${STRONG_RANGE}, 0.0, 1.0);
  amounts[i] = max(fails, better) * strong;
}`;

  const FEATHER_SHADER = /* wgsl */ `
struct P { gate: vec2i }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let id = vec2i(g.xy);
  if (id.x >= p.gate.x || id.y >= p.gate.y) { return; }
  var sum = 0.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let q = clamp(id + vec2i(x, y), vec2i(0), p.gate - 1);
      sum += src[q.y * p.gate.x + q.x];
    }
  }
  dst[id.y * p.gate.x + id.x] = sum / 9.0;
}`;

  const APPLY_SHADER = /* wgsl */ `
struct P { size: vec2f, cells: vec2i, gate: vec2i }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> cells: array<vec4f>;
@group(0) @binding(2) var<storage, read> amounts: array<f32>;
@group(0) @binding(3) var flowIn: texture_2d<f32>;
@group(0) @binding(4) var maskIn: texture_2d<f32>;
@group(0) @binding(5) var flowOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var maskOut: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let size = vec2i(textureDimensions(flowOut));
  let id = vec2i(g.xy);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let c = (vec2f(id) + 0.5) / vec2f(size) * p.size;

  let gu = clamp(c / ${GATE_CELL}.0 - 0.5, vec2f(0.0), vec2f(p.gate - 1));
  let ui = vec2i(gu);
  let uj = min(ui + 1, p.gate - 1);
  let uk = gu - vec2f(ui);
  let amount = mix(
    mix(amounts[ui.y * p.gate.x + ui.x], amounts[ui.y * p.gate.x + uj.x], uk.x),
    mix(amounts[uj.y * p.gate.x + ui.x], amounts[uj.y * p.gate.x + uj.x], uk.x),
    uk.y,
  );
  let flow = textureLoad(flowIn, id, 0);
  let mask = textureLoad(maskIn, id, 0);
  if (amount <= 0.001) {
    textureStore(flowOut, id, flow);
    textureStore(maskOut, id, mask);
    return;
  }
  let gc = clamp(c / ${SELECT_CELL}.0 - 0.5, vec2f(0.0), vec2f(p.cells - 1));
  let ci = vec2i(gc);
  let cj = min(ci + 1, p.cells - 1);
  let ck = gc - vec2f(ci);
  var px = mix(
    mix(cells[ci.y * p.cells.x + ci.x].xy, cells[ci.y * p.cells.x + cj.x].xy, ck.x),
    mix(cells[cj.y * p.cells.x + ci.x].xy, cells[cj.y * p.cells.x + cj.x].xy, ck.x),
    ck.y,
  );
  let nearest = cells[select(ci.y, cj.y, ck.y > 0.5) * p.cells.x + select(ci.x, cj.x, ck.x > 0.5)].xy;
  if (distance(px, nearest) > 1.5) { px = nearest; }
  let v = px / p.size;
  // Two different repeats both match, but a blend of them matches neither: choose, do not fade.
  let apart = distance((flow.zw - flow.xy) * p.size, px) > 1.5;
  let share = select(amount, step(0.5, amount), apart);
  textureStore(flowOut, id, mix(flow, vec4f(-v * 0.5, v * 0.5), share));
  textureStore(maskOut, id, vec4f(mix(mask.r, 0.5, share)));
}`;

  // A shot that moves as a whole, such as a drone turning over a field of identical mirrors, has
  // one motion for the entire picture. The network estimates it point by point at a fraction of the
  // source size, and a pixel off at 4K doubles every edge of a dense pattern. One affine motion fitted
  // to the whole field is far more precise, and wherever it matches as well, it is used instead.
  const FIT_SHADER = /* wgsl */ `
struct P { size: vec2f, grid: vec2i, scale: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var flow: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> model: array<vec4f, 2>;
var<workgroup> acc: array<array<f32, 12>, 256>;
var<workgroup> params: array<f32, 6>;
var<workgroup> valid: f32;

fn motion(i: i32, dims: vec2f) -> vec4f {
  let uv = (vec2f(f32(i % p.grid.x), f32(i / p.grid.x)) + 0.5) / vec2f(p.grid);
  let f = textureLoad(flow, vec2i(uv * dims), 0);
  return vec4f(uv - 0.5, (f.zw - f.xy) * p.size);
}

fn predict(x: vec2f) -> vec2f {
  return vec2f(params[0] * x.x + params[1] * x.y + params[2], params[3] * x.x + params[4] * x.y + params[5]);
}

fn reduce(l: u32) {
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (l < stride) {
      for (var k = 0; k < 12; k++) { acc[l][k] += acc[l + stride][k]; }
    }
    workgroupBarrier();
  }
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) l: u32) {
  let n = p.grid.x * p.grid.y;
  let dims = vec2f(textureDimensions(flow));
  // Reweighted least squares: each round trusts the samples the last fit explained.
  var sigmas = array<f32, 5>(1e9, 6.0, 3.0, 1.5, 1.5);
  for (var round = 0; round < 5; round++) {
    var s = array<f32, 12>();
    for (var i = i32(l); i < n; i += 256) {
      let m = motion(i, dims);
      var w = 1.0;
      if (round > 0) {
        let r = distance(m.zw, predict(m.xy)) / (sigmas[round] * p.scale);
        w = 1.0 / (1.0 + r * r);
      }
      let x = m.x;
      let y = m.y;
      s[0] += w * x * x; s[1] += w * x * y; s[2] += w * x; s[3] += w * y * y; s[4] += w * y; s[5] += w;
      s[6] += w * x * m.z; s[7] += w * y * m.z; s[8] += w * m.z;
      s[9] += w * x * m.w; s[10] += w * y * m.w; s[11] += w * m.w;
    }
    acc[l] = s;
    workgroupBarrier();
    reduce(l);
    if (l == 0u) {
      var t = acc[0];
      let det = t[0] * (t[3] * t[5] - t[4] * t[4]) - t[1] * (t[1] * t[5] - t[4] * t[2]) + t[2] * (t[1] * t[4] - t[3] * t[2]);
      valid = select(0.0, 1.0, abs(det) > 1e-9);
      for (var k = 0; k < 2; k++) {
        let r = vec3f(t[6 + k * 3], t[7 + k * 3], t[8 + k * 3]);
        let dx = r.x * (t[3] * t[5] - t[4] * t[4]) - t[1] * (r.y * t[5] - t[4] * r.z) + t[2] * (r.y * t[4] - t[3] * r.z);
        let dy = t[0] * (r.y * t[5] - t[4] * r.z) - r.x * (t[1] * t[5] - t[4] * t[2]) + t[2] * (t[1] * r.z - r.y * t[2]);
        let dz = t[0] * (t[3] * r.z - r.y * t[4]) - t[1] * (t[1] * r.z - r.y * t[2]) + r.x * (t[1] * t[4] - t[3] * t[2]);
        let safe = select(1.0, det, abs(det) > 1e-9);
        params[k * 3] = dx / safe;
        params[k * 3 + 1] = dy / safe;
        params[k * 3 + 2] = dz / safe;
      }
    }
    workgroupBarrier();
  }
  // How much of the picture the fit explains decides whether it is used at all.
  var inside = 0.0;
  for (var i = i32(l); i < n; i += 256) {
    let m = motion(i, dims);
    if (distance(m.zw, predict(m.xy)) < 2.0 * p.scale) { inside += 1.0; }
  }
  acc[l][0] = inside;
  for (var k = 1; k < 12; k++) { acc[l][k] = 0.0; }
  workgroupBarrier();
  reduce(l);
  if (l == 0u) {
    model[0] = vec4f(params[0], params[1], params[2], valid * acc[0][0] / f32(n));
    model[1] = vec4f(params[3], params[4], params[5], 0.0);
  }
}`;

  const GLOBAL_GATE_SHADER = /* wgsl */ `
${LUMA}
struct P { size: vec2f, cells: vec2i, scale: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var a: texture_2d<f32>;
@group(0) @binding(2) var b: texture_2d<f32>;
@group(0) @binding(3) var linear: sampler;
@group(0) @binding(4) var flow: texture_2d<f32>;
@group(0) @binding(5) var<storage, read> model: array<vec4f, 2>;
@group(0) @binding(6) var<storage, read_write> amounts: array<f32>;

fn la(x: vec2f) -> f32 { return dot(textureSampleLevel(a, linear, x / p.size, 0.0).rgb, LUMA); }
fn lb(x: vec2f) -> f32 { return dot(textureSampleLevel(b, linear, x / p.size, 0.0).rgb, LUMA); }

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let id = vec2i(g.xy);
  if (id.x >= p.cells.x || id.y >= p.cells.y) { return; }
  let i = id.y * p.cells.x + id.x;
  if (model[0].w < ${GLOBAL_MIN_SHARE}) {
    amounts[i] = 0.0;
    return;
  }
  let c = (vec2f(id) + 0.5) * ${GLOBAL_CELL}.0 * p.scale;
  let x = c / p.size - 0.5;
  let vGlobal = vec2f(dot(model[0].xyz, vec3f(x, 1.0)), dot(model[1].xyz, vec3f(x, 1.0)));
  let f = textureSampleLevel(flow, linear, c / p.size, 0.0) * vec4f(p.size, p.size);
  let vNow = f.zw - f.xy;
  if (distance(vGlobal, vNow) < 0.5 * p.scale) {
    amounts[i] = 1.0;
    return;
  }
  // Windows may shift, so a cell at an object's edge can decide from the side it belongs to.
  var qGlobal = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
  var qNow = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
  for (var y = -2; y <= 2; y++) {
    for (var x = -2; x <= 2; x++) {
      let s = c + vec2f(f32(x), f32(y)) * 3.0 * p.scale;
      let eGlobal = abs(la(s - vGlobal * 0.5) - lb(s + vGlobal * 0.5));
      let eNow = abs(la(s - vNow * 0.5) - lb(s + vNow * 0.5));
      for (var q = 0; q < 4; q++) {
        let inX = select(x <= 0, x >= 0, (q & 1) == 1);
        let inY = select(y <= 0, y >= 0, (q & 2) == 2);
        if (inX && inY) {
          qGlobal[q] += eGlobal;
          qNow[q] += eNow;
        }
      }
    }
  }
  let global = min(min(qGlobal[0], qGlobal[1]), min(qGlobal[2], qGlobal[3])) / 9.0;
  let now = min(min(qNow[0], qNow[1]), min(qNow[2], qNow[3])) / 9.0;
  // As good is enough: of two motions that both match, the fitted one is the precise one. A moving
  // object does not match the camera's motion, so it keeps the network's.
  amounts[i] = clamp((now - global + ${GLOBAL_TOLERANCE}) / ${GLOBAL_TOLERANCE}, 0.0, 1.0);
}`;

  const GLOBAL_APPLY_SHADER = /* wgsl */ `
struct P { size: vec2f, cells: vec2i, scale: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> model: array<vec4f, 2>;
@group(0) @binding(2) var<storage, read> amounts: array<f32>;
@group(0) @binding(3) var flowIn: texture_2d<f32>;
@group(0) @binding(4) var maskIn: texture_2d<f32>;
@group(0) @binding(5) var flowOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var maskOut: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let size = vec2i(textureDimensions(flowOut));
  let id = vec2i(g.xy);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let uv = (vec2f(id) + 0.5) / vec2f(size);
  let flow = textureLoad(flowIn, id, 0);
  let mask = textureLoad(maskIn, id, 0);
  let gc = clamp(uv * p.size / (${GLOBAL_CELL}.0 * p.scale) - 0.5, vec2f(0.0), vec2f(p.cells - 1));
  let ci = vec2i(gc);
  let cj = min(ci + 1, p.cells - 1);
  let ck = gc - vec2f(ci);
  let amount = mix(
    mix(amounts[ci.y * p.cells.x + ci.x], amounts[ci.y * p.cells.x + cj.x], ck.x),
    mix(amounts[cj.y * p.cells.x + ci.x], amounts[cj.y * p.cells.x + cj.x], ck.x),
    ck.y,
  );
  if (amount <= 0.001) {
    textureStore(flowOut, id, flow);
    textureStore(maskOut, id, mask);
    return;
  }
  let x = uv - 0.5;
  let v = vec2f(dot(model[0].xyz, vec3f(x, 1.0)), dot(model[1].xyz, vec3f(x, 1.0)));
  // Two motions far apart cannot be blended into one that matches: choose, do not fade.
  let apart = distance((flow.zw - flow.xy) * p.size, v) > 1.5 * p.scale;
  let share = select(amount, step(0.5, amount), apart);
  let u = v / p.size;
  textureStore(flowOut, id, mix(flow, vec4f(-u * 0.5, u * 0.5), share));
  textureStore(maskOut, id, vec4f(mix(mask.r, 0.5, share)));
}`;

  // What the repaired flow still cannot explain, on a coarse grid: only a hard cut stays high.
  const RESIDUAL_SHADER = /* wgsl */ `
${LUMA}
struct P { gridW: i32, gridH: i32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var a: texture_2d<f32>;
@group(0) @binding(2) var b: texture_2d<f32>;
@group(0) @binding(3) var linear: sampler;
@group(0) @binding(4) var flow: texture_2d<f32>;
@group(0) @binding(5) var<storage, read_write> residual: array<f32>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let x = i32(g.x);
  let y = i32(g.y);
  if (x >= p.gridW || y >= p.gridH) { return; }
  let uv = (vec2f(g.xy) + 0.5) / vec2f(f32(p.gridW), f32(p.gridH));
  let f = textureSampleLevel(flow, linear, uv, 0.0);
  let la = dot(textureSampleLevel(a, linear, uv + f.xy, 0.0).rgb, LUMA);
  let lb = dot(textureSampleLevel(b, linear, uv + f.zw, 0.0).rgb, LUMA);
  residual[y * p.gridW + x] = abs(la - lb);
}`;

  class FlowRepair {
    constructor(device, sampler) {
      this.device = device;
      this.sampler = sampler;
      const compute = (code) =>
        device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
      this.pipelines = {
        load: compute(LOAD_SHADER),
        fftRows: compute(fftShader(false, false)),
        fftCols: compute(fftShader(true, false)),
        ifftRows: compute(fftShader(false, true)),
        ifftCols: compute(fftShader(true, true)),
        cross: compute(CROSS_SHADER),
        rowPeak: compute(ROW_PEAK_SHADER),
        peak: compute(PEAK_SHADER),
        select: compute(SELECT_SHADER),
        smooth: compute(SMOOTH_SHADER),
        gate: compute(GATE_SHADER),
        feather: compute(FEATHER_SHADER),
        apply: compute(APPLY_SHADER),
        residual: compute(RESIDUAL_SHADER),
        fit: compute(FIT_SHADER),
        globalGate: compute(GLOBAL_GATE_SHADER),
        globalApply: compute(GLOBAL_APPLY_SHADER),
      };
      this.twiddle = device.createBuffer({ size: (TILE / 2) * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const angles = new Float32Array(TILE);
      for (let i = 0; i < TILE / 2; i++) {
        angles[i * 2] = Math.cos((-2 * Math.PI * i) / TILE);
        angles[i * 2 + 1] = Math.sin((-2 * Math.PI * i) / TILE);
      }
      device.queue.writeBuffer(this.twiddle, 0, angles);
      this.owned = [];
    }

    configure(width, height) {
      this.release();
      const d = this.device;
      // Tiles cover the picture edge to edge, at about 1080p detail on larger sources.
      const step = Math.max(1, Math.round(height / 1080));
      const stride = TILE;
      const nx = Math.max(1, Math.floor((width / step - TILE) / stride) + 1);
      const ny = Math.max(1, Math.floor((height / step - TILE) / stride) + 1);
      const tiles = nx * ny;
      const offset = [(width - ((nx - 1) * stride + TILE) * step) / 2, (height - ((ny - 1) * stride + TILE) * step) / 2];
      this.width = width;
      this.height = height;
      this.tiles = tiles;
      this.nx = nx;
      this.cells = [Math.ceil(width / SELECT_CELL), Math.ceil(height / SELECT_CELL)];
      this.gate = [Math.ceil(width / GATE_CELL), Math.ceil(height / GATE_CELL)];
      const storage = (size) => {
        const buffer = d.createBuffer({ size, usage: GPUBufferUsage.STORAGE });
        this.owned.push(buffer);
        return buffer;
      };
      const uniform = (write) => {
        const data = new ArrayBuffer(48);
        write(new Float32Array(data), new Int32Array(data));
        const buffer = d.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        d.queue.writeBuffer(buffer, 0, data);
        this.owned.push(buffer);
        return buffer;
      };
      this.spectrum = storage(tiles * TILE * TILE * 8);
      this.correlation = storage(tiles * TILE * TILE * 8);
      this.rowPeaks = storage(tiles * TILE * PEAKS * 16);
      this.peaks = storage(tiles * PEAKS * 16);
      this.cellsRaw = storage(this.cells[0] * this.cells[1] * 16);
      this.cellsSmooth = storage(this.cells[0] * this.cells[1] * 16);
      this.useRaw = storage(this.gate[0] * this.gate[1] * 4);
      this.useSoft = storage(this.gate[0] * this.gate[1] * 4);
      this.tileParams = uniform((f, i) => {
        f.set([offset[0], offset[1], stride, step, width, height]);
        i.set([nx, ny, this.cells[0], this.cells[1]], 6);
      });
      this.cellParams = uniform((f, i) => i.set([this.cells[0], this.cells[1]]));
      const scale = height / 1080;
      this.globalCells = [Math.ceil(width / (GLOBAL_CELL * scale)), Math.ceil(height / (GLOBAL_CELL * scale))];
      this.model = d.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }); // readable, so tests can see the fit
      this.owned.push(this.model);
      this.globalRaw = storage(this.globalCells[0] * this.globalCells[1] * 4);
      this.globalSoft = storage(this.globalCells[0] * this.globalCells[1] * 4);
      this.fitParams = uniform((f, i) => {
        f.set([width, height]);
        i.set(FIT_GRID, 2);
        f[4] = scale;
      });
      this.globalParams = uniform((f, i) => {
        f.set([width, height]);
        i.set(this.globalCells, 2);
        f[4] = scale;
      });
      this.globalFeatherParams = uniform((f, i) => i.set(this.globalCells));
      this.gateParams = uniform((f, i) => i.set([this.gate[0], this.gate[1]]));
      this.sizeParams = uniform((f, i) => {
        f.set([width, height]);
        i.set([this.cells[0], this.cells[1], this.gate[0], this.gate[1]], 2);
      });
      const group = (pipeline, resources) =>
        d.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: resources.map((r, binding) => ({ binding, resource: r instanceof GPUTexture ? r.createView() : r instanceof GPUSampler ? r : { buffer: r } })) });
      this.group = group;
      const p = this.pipelines;
      const lines = Math.ceil((tiles * TILE) / 32);
      const fft = (pipeline, buffer) => ({ pipeline, group: group(pipeline, [buffer, this.twiddle]), x: lines, y: 1, z: 1, cost: (tiles * TILE * TILE * 8 * 8) / 2 });
      this.fixedFft = [
        fft(p.fftRows, this.spectrum),
        fft(p.fftCols, this.spectrum),
        { pipeline: p.cross, group: group(p.cross, [this.spectrum, this.correlation]), x: TILE / 16, y: TILE / 16, z: tiles, cost: tiles * TILE * TILE * 16 },
        fft(p.ifftRows, this.correlation),
        fft(p.ifftCols, this.correlation),
        { pipeline: p.rowPeak, group: group(p.rowPeak, [this.correlation, this.rowPeaks]), x: lines, y: 1, z: 1, cost: tiles * TILE * TILE * 4 },
        { pipeline: p.peak, group: group(p.peak, [this.tileParams, this.correlation, this.peaks, this.rowPeaks]), x: tiles, y: 1, z: 1, cost: tiles * TILE * PEAKS * PEAKS * 4 },
      ];
      this.fixedSmooth = [
        { pipeline: p.smooth, group: group(p.smooth, [this.cellParams, this.cellsRaw, this.cellsSmooth]), x: Math.ceil(this.cells[0] / 8), y: Math.ceil(this.cells[1] / 8), z: 1, cost: this.cells[0] * this.cells[1] * 16 },
      ];
      const [gcw, gch] = this.globalCells;
      this.fixedGlobalFeather = [
        { pipeline: p.feather, group: group(p.feather, [this.globalFeatherParams, this.globalRaw, this.globalSoft]), x: Math.ceil(gcw / 8), y: Math.ceil(gch / 8), z: 1, cost: gcw * gch * 16 },
      ];
      this.fixedFeather = [
        { pipeline: p.feather, group: group(p.feather, [this.gateParams, this.useRaw, this.useSoft]), x: Math.ceil(this.gate[0] / 8), y: Math.ceil(this.gate[1] / 8), z: 1, cost: this.gate[0] * this.gate[1] * 16 },
      ];
    }

    /** Steps writing the repaired flow and mask for a pair, from the network's flow in uv units. */
    plan(frameA, frameB, flowIn, maskIn, flowOut, maskOut, residual, grid) {
      const p = this.pipelines;
      const { group, sampler, tiles } = this;
      if (this.midFlow?.width !== flowOut.width || this.midFlow?.height !== flowOut.height) {
        this.midFlow?.destroy();
        this.midMask?.destroy();
        const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING;
        this.midFlow = this.device.createTexture({ size: [flowOut.width, flowOut.height], format: "rgba16float", usage });
        this.midMask = this.device.createTexture({ size: [flowOut.width, flowOut.height], format: "rgba8unorm", usage });
      }
      const { midFlow, midMask } = this;
      const [gcw, gch] = this.globalCells;
      const sampled = sampler;
      const [cw, ch] = this.cells;
      const [gw, gh] = this.gate;
      const steps = [
        {
          pipeline: p.load,
          group: group(p.load, [this.tileParams, frameA, frameB, sampled, this.spectrum]),
          x: TILE / 16,
          y: TILE / 16,
          z: tiles,
          cost: tiles * TILE * TILE * 2 * SAMPLE_COST,
        },
        ...this.fixedFft,
        {
          pipeline: p.select,
          group: group(p.select, [this.tileParams, frameA, frameB, sampled, flowIn, this.peaks, this.cellsRaw]),
          x: Math.ceil(cw / 8),
          y: Math.ceil(ch / 8),
          z: 1,
          cost: cw * ch * 200 * SAMPLE_COST,
        },
        ...this.fixedSmooth,
        {
          pipeline: p.gate,
          group: group(p.gate, [this.sizeParams, frameA, frameB, sampled, flowIn, this.cellsSmooth, this.useRaw]),
          x: Math.ceil(gw / 8),
          y: Math.ceil(gh / 8),
          z: 1,
          cost: gw * gh * 40 * SAMPLE_COST,
        },
        ...this.fixedFeather,
        {
          pipeline: p.apply,
          group: group(p.apply, [this.sizeParams, this.cellsSmooth, this.useSoft, flowIn, maskIn, midFlow, midMask]),
          x: Math.ceil(flowOut.width / 8),
          y: Math.ceil(flowOut.height / 8),
          z: 1,
          cost: flowOut.width * flowOut.height * 16,
        },
        {
          pipeline: p.fit,
          group: group(p.fit, [this.fitParams, midFlow, this.model]),
          x: 1,
          y: 1,
          z: 1,
          cost: FIT_GRID[0] * FIT_GRID[1] * 6 * 40,
        },
        {
          pipeline: p.globalGate,
          group: group(p.globalGate, [this.globalParams, frameA, frameB, sampled, midFlow, this.model, this.globalRaw]),
          x: Math.ceil(gcw / 8),
          y: Math.ceil(gch / 8),
          z: 1,
          cost: gcw * gch * 50 * SAMPLE_COST,
        },
        ...this.fixedGlobalFeather,
        {
          pipeline: p.globalApply,
          group: group(p.globalApply, [this.globalParams, this.model, this.globalSoft, midFlow, midMask, flowOut, maskOut]),
          x: Math.ceil(flowOut.width / 8),
          y: Math.ceil(flowOut.height / 8),
          z: 1,
          cost: flowOut.width * flowOut.height * 16,
        },
        {
          pipeline: p.residual,
          group: group(p.residual, [this.residualParams(grid), frameA, frameB, sampled, flowOut, residual]),
          x: Math.ceil(grid[0] / 8),
          y: Math.ceil(grid[1] / 8),
          z: 1,
          cost: grid[0] * grid[1] * 3 * SAMPLE_COST,
        },
      ];
      return steps;
    }

    residualParams(grid) {
      if (!this.gridParams) {
        this.gridParams = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.device.queue.writeBuffer(this.gridParams, 0, new Int32Array(grid));
      }
      return this.gridParams;
    }

    release() {
      for (const buffer of this.owned) buffer.destroy();
      this.owned = [];
      this.midFlow?.destroy();
      this.midMask?.destroy();
      this.midFlow = this.midMask = null;
    }
  }

  globalThis.FrameBoostRepair = { FlowRepair };
})();
