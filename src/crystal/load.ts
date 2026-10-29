// Loading a crystal: a URL, a Response, a Blob or bytes; gzip-compressed or not (decided by the magic bytes, not the name).
// Gzip is unpacked here with the platform's own DecompressionStream, so the site serves `.safetensors.gz` as plain
// application/octet-stream (no Content-Encoding) and nothing unpacks it before the engine does.
import type { CrystalData } from "./types.js";
import { readCrystal } from "./read.js";

export interface LoadProgress {
  /** bytes received so far (compressed bytes when the file is gzip) */
  loaded: number;
  /** total bytes expected, when the server said (Content-Length), else null */
  total: number | null;
  stage: "download" | "unpack" | "read";
}

export interface LoadOptions {
  onProgress?: (p: LoadProgress) => void;
  signal?: AbortSignal;
  fetchFn?: typeof fetch;
  /**
   * an 8-bit crystal's maps kept as their 8-bit steps (Uint8Array, phase k·2π/256) — for phones: a quarter of 32-bit, half
   * of f16, and exactly what the file holds and the phone's GPU uses (phase8). Readers turn steps into radians (spreadMap, mapWindow);
   * CrystalData.steps says so. A 16-bit crystal's maps stay 32-bit.
   */
  stepMaps?: boolean;
}

export type CrystalSource = string | URL | Response | Blob | ArrayBuffer | Uint8Array;

export async function loadCrystal(src: CrystalSource, opts: LoadOptions = {}): Promise<CrystalData> {
  const bytes = await loadBytes(src, opts);
  opts.onProgress?.({ loaded: bytes.length, total: bytes.length, stage: "read" });
  return readCrystal(bytes, { stepMaps: !!opts.stepMaps });
}

/** The file's bytes, unpacked if it was gzip. */
export async function loadBytes(src: CrystalSource, opts: LoadOptions = {}): Promise<Uint8Array> {
  const raw = await rawBytes(src, opts);
  if (!isGzip(raw)) return raw;
  opts.onProgress?.({ loaded: raw.length, total: raw.length, stage: "unpack" });
  return gunzip(raw, opts.signal);
}

export function isGzip(b: Uint8Array): boolean {
  return b.length > 2 && b[0] === 0x1f && b[1] === 0x8b;
}

export async function gunzip(b: Uint8Array, signal?: AbortSignal): Promise<Uint8Array> {
  const stream = new Blob([b as BlobPart]).stream().pipeThrough(new DecompressionStream("gzip"));
  // the unpacked size is in gzip's last four bytes (mod 2³²): the bytes go straight into one buffer of that size, not pieces then a copy
  const n = b.length >= 4 ? (b[b.length - 4]! | (b[b.length - 3]! << 8) | (b[b.length - 2]! << 16) | (b[b.length - 1]! << 24)) >>> 0 : 0;
  return collect(stream, null, signal, n || null);
}

async function rawBytes(src: CrystalSource, opts: LoadOptions): Promise<Uint8Array> {
  if (src instanceof Uint8Array) return src;
  if (src instanceof ArrayBuffer) return new Uint8Array(src);
  if (typeof Blob !== "undefined" && src instanceof Blob) return new Uint8Array(await src.arrayBuffer());
  const res = src instanceof Response ? src : await (opts.fetchFn ?? fetch)(String(src), opts.signal ? { signal: opts.signal } : {});
  if (!res.ok) throw new Error(`crystal download failed: ${res.status} ${res.statusText}`);
  const len = Number(res.headers.get("content-length"));
  const total = len > 0 ? len : null;
  if (!res.body) return new Uint8Array(await res.arrayBuffer());
  return collect(res.body, (loaded) => opts.onProgress?.({ loaded, total, stage: "download" }), opts.signal, total);
}

/**
 * a stream's bytes in one buffer. size: what it is expected to come to — each chunk is written straight into a buffer of that size (no
 * second copy of the whole file at the end); if it turns out larger, the chunks are gathered and joined as before
 */
async function collect(stream: ReadableStream<Uint8Array>, tick: ((n: number) => void) | null, signal?: AbortSignal, size: number | null = null): Promise<Uint8Array> {
  const reader = stream.getReader();
  let parts: Uint8Array[] | null = size ? null : [];
  let buf: Uint8Array | null = size ? new Uint8Array(size) : null;
  let n = 0;
  for (;;) {
    if (signal?.aborted) {
      await reader.cancel();
      throw new DOMException("crystal load aborted", "AbortError");
    }
    const { done, value } = await reader.read();
    if (done) break;
    if (buf && n + value.length > buf.length) {
      parts = [buf.subarray(0, n)]; // larger than said: gather from here on
      buf = null;
    }
    if (buf) buf.set(value, n);
    else parts!.push(value);
    n += value.length;
    tick?.(n);
  }
  if (buf) return n === buf.length ? buf : buf.subarray(0, n);
  parts ??= [];
  const out = new Uint8Array(n);
  let p = 0;
  for (const part of parts) {
    out.set(part, p);
    p += part.length;
  }
  return out;
}
