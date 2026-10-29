// What the examples share: loading a crystal from a path or a URL, an engine on the GPU when Node has one (the `webgpu` package, a dev
// dependency of this repo) or else on the CPU, and reading and writing PNGs with no image library (8-bit grey or RGB, as the assets are).
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { inflateSync, deflateSync } from "node:zlib";
import { CrystalEngine, WebGpuBackend, loadCrystal, requestGpu } from "../src/index.js";
import type { CrystalData, Field } from "../src/index.js";


/** a crystal from a local file or a URL */
export async function load(src: string): Promise<CrystalData> {
  return loadCrystal(existsSync(src) ? new Uint8Array(readFileSync(src)) : src);
}

let held: unknown = null; // Dawn's instance must outlive every engine made from it
/** an engine on WebGPU through Dawn when the `webgpu` package is installed (npm install in this repo brings it), else the CPU */
export async function engine(data: CrystalData): Promise<CrystalEngine> {
  try {
    const webgpu = (await import("webgpu")) as { create: (o: string[]) => unknown; globals: object };
    Object.assign(globalThis, webgpu.globals);
    held ??= webgpu.create([]);
    const gpu = await requestGpu(held as GPU);
    if (gpu) return new CrystalEngine(data, (g) => new WebGpuBackend(gpu.device, g));
  } catch {
    // no webgpu package or no adapter: the CPU it is
  }
  return new CrystalEngine(data);
}

/** a PNG → its pixels, `channels` a pixel (1 grey, 3 RGB) */
export function readPng(path: string): { w: number; h: number; channels: number; px: Uint8Array } {
  const b = readFileSync(path);
  let w = 0, h = 0, type = 0;
  const idat: Buffer[] = [];
  for (let at = 8; at < b.length; ) {
    const n = b.readUInt32BE(at), kind = b.toString("latin1", at + 4, at + 8), data = b.subarray(at + 8, at + 8 + n);
    if (kind === "IHDR") {
      w = data.readUInt32BE(0); h = data.readUInt32BE(4); type = data[9]!;
      if (data[8] !== 8 || data[12] !== 0 || (type !== 0 && type !== 2)) throw new Error(`${path}: only 8-bit grey or RGB PNGs without interlacing`);
    } else if (kind === "IDAT") idat.push(data);
    at += 12 + n;
  }
  const c = type === 2 ? 3 : 1, stride = w * c, raw = inflateSync(Buffer.concat(idat)), px = new Uint8Array(w * h * c);
  for (let y = 0; y < h; y++) {
    const f = raw[y * (stride + 1)]!, row = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    for (let x = 0; x < stride; x++) {
      const a = x >= c ? px[y * stride + x - c]! : 0, up = y ? px[(y - 1) * stride + x]! : 0, ul = y && x >= c ? px[(y - 1) * stride + x - c]! : 0;
      const p = a + up - ul, pa = Math.abs(p - a), pb = Math.abs(p - up), pc = Math.abs(p - ul);
      const pred = f === 1 ? a : f === 2 ? up : f === 3 ? (a + up) >> 1 : f === 4 ? (pa <= pb && pa <= pc ? a : pb <= pc ? up : ul) : 0;
      px[y * stride + x] = (row[x]! + pred) & 255;
    }
  }
  return { w, h, channels: c, px };
}

/** an RGB picture, w × h */
export interface Tile { w: number; h: number; rgb: Uint8Array }

/** a 0..1 picture (a DMD frame, a hidden picture) in the laser's red, `scale` times larger */
export function frameTile(v: ArrayLike<number>, F: number, scale = 1): Tile {
  return tile(F, scale, (i) => { const g = Math.round(255 * Math.min(1, Math.max(0, v[i]!))); return [g, Math.round(g * 0.4), Math.round(g * 0.28)]; });
}
/** light: square-root brightness scaled to its brightest point, so faint light shows */
export function lightTile(v: ArrayLike<number>, F: number, scale = 1): Tile {
  let m = 0;
  for (let i = 0; i < F * F; i++) m = Math.max(m, v[i]!);
  return frameTile(Array.from({ length: F * F }, (_, i) => Math.sqrt(v[i]! / (m || 1))), F, scale);
}
/** the intensity of a field (or of its window) as light */
export function fieldTile(E: Field, scale = 1): Tile {
  return lightTile(Array.from({ length: E.N * E.N }, (_, i) => E.re[i]! ** 2 + E.im[i]! ** 2), E.N, scale);
}
/** an F×F photo, RGB bytes */
export function photoTile(rgb: Uint8Array, F: number, scale = 1): Tile {
  return tile(F, scale, (i) => [rgb[3 * i]!, rgb[3 * i + 1]!, rgb[3 * i + 2]!]);
}
function tile(F: number, scale: number, at: (i: number) => [number, number, number]): Tile {
  const W = F * scale, rgb = new Uint8Array(W * W * 3);
  for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) rgb.set(at(Math.floor(y / scale) * F + Math.floor(x / scale)), 3 * (y * W + x));
  return { w: W, h: W, rgb };
}
/** outline a rectangle on a tile (in the tile's pixels) */
export function outline(t: Tile, x0: number, y0: number, w: number, h: number, rgb: [number, number, number]): void {
  const put = (x: number, y: number): void => { if (x >= 0 && y >= 0 && x < t.w && y < t.h) t.rgb.set(rgb, 3 * (y * t.w + x)); };
  for (let x = x0; x < x0 + w; x++) { put(x, y0); put(x, y0 + h - 1); }
  for (let y = y0; y < y0 + h; y++) { put(x0, y); put(x0 + w - 1, y); }
}

/** tiles side by side on black (rows of them, top to bottom), written as a PNG */
export function savePng(path: string, rows: Tile[][], gap = 8): void {
  const W = Math.max(...rows.map((r) => r.reduce((s, t) => s + t.w + gap, gap))), H = rows.reduce((s, r) => s + Math.max(...r.map((t) => t.h)) + gap, gap);
  const px = new Uint8Array(W * H * 3);
  let y0 = gap;
  for (const r of rows) {
    let x0 = gap;
    for (const t of r) {
      for (let y = 0; y < t.h; y++) px.set(t.rgb.subarray(3 * y * t.w, 3 * (y + 1) * t.w), 3 * ((y0 + y) * W + x0));
      x0 += t.w + gap;
    }
    y0 += Math.max(...r.map((t) => t.h)) + gap;
  }
  const raw = new Uint8Array(H * (3 * W + 1));
  for (let y = 0; y < H; y++) raw.set(px.subarray(3 * y * W, 3 * (y + 1) * W), y * (3 * W + 1) + 1);
  const crcT = Array.from({ length: 256 }, (_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c >>> 0; });
  const chunk = (kind: string, data: Uint8Array): Buffer => {
    const body = Buffer.concat([Buffer.from(kind, "latin1"), data]);
    let c = 0xffffffff;
    for (const x of body) c = crcT[(c ^ x) & 255]! ^ (c >>> 8);
    const out = Buffer.alloc(body.length + 8);
    out.writeUInt32BE(data.length, 0); body.copy(out, 4); out.writeUInt32BE((c ^ 0xffffffff) >>> 0, body.length + 4);
    return out;
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(W, 0); ihdr.writeUInt32BE(H, 4); ihdr[8] = 8; ihdr[9] = 2;
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", ihdr), chunk("IDAT", deflateSync(raw)), chunk("IEND", new Uint8Array())]));
}

/** the light on the exit face, the whole N grid summed down to size × size (rectangles on the grid scale by size / N) */
export function exitTile(E: Field, size: number): Tile {
  const N = E.N, k = N / size, I = new Float64Array(size * size);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) I[Math.floor(y / k) * size + Math.floor(x / k)]! += E.re[y * N + x]! ** 2 + E.im[y * N + x]! ** 2;
  return lightTile(I, size);
}
