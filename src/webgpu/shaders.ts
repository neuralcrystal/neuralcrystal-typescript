// The fast light path's kernels (fp32): a hop is a row kernel and a column kernel, each FFT done in workgroup
// memory, so the field crosses GPU memory twice a hop. Generated per grid size N (a power of two, 64..2048).
//
// row kernel, flags: 1 inverse row FFT first · 2 multiply by e^{iφ} · 4 forward row FFT after · 32 absorbing rim (zero outside the
//   window box [u.off, u.off + u.D) on both axes, applied to the spatial field right after the inverse)
// col kernel: forward column FFT · × H(z) · inverse column FFT (scale 1/N, the rows' two 1/√N make the rest)

export const ROW_INVERSE = 1;
export const ROW_PHASE = 2;
export const ROW_FORWARD = 4;
export const ROW_ABSORB = 32;

function head(N: number): string {
  const T = Math.min(N / 2, 256);
  const L2 = Math.round(Math.log2(N));
  return `
const N : u32 = ${N}u; const LOG2 : u32 = ${L2}u; const T : u32 = ${T}u; const E : u32 = ${N / T}u; const BF : u32 = ${N / 2 / T}u;
const ISQ : f32 = ${(1 / Math.sqrt(N)).toPrecision(9)};
struct U { flags : u32, off : u32, D : u32, s : u32, scale : f32, p0 : f32, p1 : f32, p2 : f32 };
@group(0) @binding(0) var<uniform> u : U;
@group(0) @binding(2) var<storage, read> tw : array<vec2<f32>>;
var<workgroup> sr : array<f32, ${N}>;
var<workgroup> si : array<f32, ${N}>;
fn rev(k : u32) -> u32 { return reverseBits(k) >> (32u - LOG2); }
fn dif(t : u32, sg : f32) {                 // natural order in, bit-reversed out (Gentleman–Sande); sg = 1 forward, -1 inverse
  var L = N;
  for (var s = 0u; s < LOG2; s = s + 1u) {
    let h = L >> 1u; let st = N / L;
    for (var b = 0u; b < BF; b = b + 1u) {
      let q = t + b * T; let g = q / h; let p = q - g * h; let i = g * L + p; let j = i + h;
      let w = tw[p * st]; let wi = w.y * sg;
      let ar = sr[i]; let ai = si[i]; let br = sr[j]; let bi = si[j];
      sr[i] = ar + br; si[i] = ai + bi;
      let dr = ar - br; let di = ai - bi;
      sr[j] = dr * w.x - di * wi; si[j] = dr * wi + di * w.x;
    }
    workgroupBarrier();
    L = h;
  }
}
fn dit(t : u32, sg : f32) {                 // bit-reversed in, natural order out (Cooley–Tukey)
  var L = 2u;
  for (var s = 0u; s < LOG2; s = s + 1u) {
    let h = L >> 1u; let st = N / L;
    for (var b = 0u; b < BF; b = b + 1u) {
      let q = t + b * T; let g = q / h; let p = q - g * h; let i = g * L + p; let j = i + h;
      let w = tw[p * st]; let wi = w.y * sg;
      let br = sr[j]; let bi = si[j];
      let tr = br * w.x - bi * wi; let ti = br * wi + bi * w.x;
      let ar = sr[i]; let ai = si[i];
      sr[i] = ar + tr; si[i] = ai + ti; sr[j] = ar - tr; si[j] = ai - ti;
    }
    workgroupBarrier();
    L = L << 1u;
  }
}
`;
}

/**
 * q8: the phase map held as 8-bit steps, four to a u32 (for phones): an 8-bit crystal's every value is k·2π/256, so the
 * step k is the whole map at a quarter of the memory, and the phase is k·2π/256 exactly as the CPU unpacks it
 */
export function rowKernel(N: number, q8 = false): string {
  const T = Math.min(N / 2, 256);
  const phase = q8 ? "f32((ph[(base + c) >> 2u] >> (((base + c) & 3u) * 8u)) & 255u) * 0.02454369260617026" : "ph[base + c]";
  return `${head(N)}
@group(0) @binding(1) var<storage, read_write> x : array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> ph : array<${q8 ? "u32" : "f32"}>;
@compute @workgroup_size(${T})
fn main(@builtin(workgroup_id) wg : vec3<u32>, @builtin(local_invocation_id) li : vec3<u32>) {
  let t = li.x; let y = wg.x; let base = y * N; let f = u.flags; let inv = (f & 1u) != 0u;
  let rowOut = (f & 32u) != 0u && (y < u.off || y >= u.off + u.D);
  for (var q = 0u; q < E; q = q + 1u) { let k = t + q * T; let v = x[base + k]; var j = k; if (!inv) { j = rev(k); } sr[j] = v.x; si[j] = v.y; }
  workgroupBarrier();
  var sc = 1.0; if (inv) { dif(t, -1.0); sc = ISQ; }
  for (var q = 0u; q < E; q = q + 1u) {
    let j = t + q * T; let c = rev(j); var a = sr[j] * sc; var b = si[j] * sc;
    if ((f & 32u) != 0u && (rowOut || c < u.off || c >= u.off + u.D)) { a = 0.0; b = 0.0; }
    if ((f & 2u) != 0u) { let p = ${phase}; let cs = cos(p); let sn = sin(p); let na = a * cs - b * sn; b = a * sn + b * cs; a = na; }
    sr[j] = a; si[j] = b;
  }
  workgroupBarrier();
  if ((f & 4u) != 0u) {
    dit(t, 1.0);
    for (var q = 0u; q < E; q = q + 1u) { let k = t + q * T; x[base + k] = vec2<f32>(sr[k] * ISQ, si[k] * ISQ); }
  } else {
    for (var q = 0u; q < E; q = q + 1u) { let j = t + q * T; x[base + rev(j)] = vec2<f32>(sr[j], si[j]); }
  }
}`;
}

export function colKernel(N: number): string {
  const T = Math.min(N / 2, 256);
  return `${head(N)}
@group(0) @binding(1) var<storage, read_write> x : array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> H : array<vec2<f32>>;
@compute @workgroup_size(${T})
fn main(@builtin(workgroup_id) wg : vec3<u32>, @builtin(local_invocation_id) li : vec3<u32>) {
  let t = li.x; let l = wg.x;
  for (var q = 0u; q < E; q = q + 1u) { let k = t + q * T; let v = x[k * N + l]; sr[k] = v.x; si[k] = v.y; }
  workgroupBarrier();
  dif(t, 1.0);
  for (var q = 0u; q < E; q = q + 1u) { let j = t + q * T; let h = H[rev(j) * N + l] * u.scale; let a = sr[j]; let b = si[j]; sr[j] = a * h.x - b * h.y; si[j] = a * h.y + b * h.x; }
  workgroupBarrier();
  dit(t, -1.0);
  for (var q = 0u; q < E; q = q + 1u) { let k = t + q * T; x[k * N + l] = vec2<f32>(sr[k], si[k]); }
}`;
}

/**
 * The half-precision readback (a phone's light trace): the field's window, each complex sample packed into one u32 (re in the low half,
 * im in the high), clamped to f16's range. p = (x0, y0, M, N).
 */
export function packKernel(): string {
  return `
struct P { x0 : u32, y0 : u32, M : u32, N : u32 };
@group(0) @binding(0) var<uniform> p : P;
@group(0) @binding(1) var<storage, read> x : array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> o : array<u32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  let k = id.x;
  if (k >= p.M * p.M) { return; }
  let r = k / p.M; let c = k - r * p.M;
  o[k] = pack2x16float(clamp(x[(p.y0 + r) * p.N + p.x0 + c], vec2<f32>(-65504.0), vec2<f32>(65504.0)));
}`;
}
