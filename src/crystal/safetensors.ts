// The safetensors container: [u64 LE header length][JSON header][tensor bytes].

export interface TensorInfo {
  dtype: string;
  shape: number[];
  /** byte range [begin, end) relative to the start of the data section */
  data_offsets: [number, number];
}

export interface SafetensorsFile {
  metadata: Record<string, string>;
  tensors: Map<string, TensorInfo>;
  /** the whole file */
  bytes: Uint8Array;
  /** where the data section starts in `bytes` */
  base: number;
}

/** True when these bytes look like a safetensors file: a sane header length followed by `{`. */
export function isSafetensors(bytes: Uint8Array): boolean {
  if (bytes.length < 16) return false;
  const n = Number(new DataView(bytes.buffer, bytes.byteOffset, 8).getBigUint64(0, true));
  return n > 1 && n <= bytes.length - 8 && bytes[8] === 0x7b;
}

export function parseSafetensors(bytes: Uint8Array): SafetensorsFile {
  if (!isSafetensors(bytes)) throw new Error("not a safetensors file");
  const n = Number(new DataView(bytes.buffer, bytes.byteOffset, 8).getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + n))) as Record<string, unknown>;
  const metadata = (header["__metadata__"] ?? {}) as Record<string, string>;
  const tensors = new Map<string, TensorInfo>();
  for (const [k, v] of Object.entries(header)) if (k !== "__metadata__") tensors.set(k, v as TensorInfo);
  return { metadata, tensors, bytes, base: 8 + n };
}

/** A tensor as a Float32Array (copied, so alignment never matters). F32 only — the crystal format has no other dtype. */
export function tensorF32(file: SafetensorsFile, name: string): Float32Array | null {
  const t = file.tensors.get(name);
  if (!t) return null;
  if (t.dtype !== "F32") throw new Error(`${name}: dtype ${t.dtype}, expected F32`);
  const [a, b] = t.data_offsets;
  const out = new Float32Array((b - a) / 4);
  new Uint8Array(out.buffer).set(file.bytes.subarray(file.base + a, file.base + b));
  return out;
}

/** A tensor's raw bytes and dtype (a copy), for the packed web form's integer maps */
export function tensorBytes(file: SafetensorsFile, name: string): { dtype: string; shape: number[]; bytes: Uint8Array } | null {
  const t = file.tensors.get(name);
  if (!t) return null;
  const [a, b] = t.data_offsets;
  return { dtype: t.dtype, shape: t.shape, bytes: file.bytes.slice(file.base + a, file.base + b) };
}

/** Write a safetensors file (tests, and the web-copy tool): tensors in the order given, header padded with spaces to 8. */
export function writeSafetensors(metadata: Record<string, string>, tensors: [string, Float32Array | Uint16Array | Uint8Array, number[]][]): Uint8Array {
  const header: Record<string, unknown> = { __metadata__: metadata };
  let off = 0;
  for (const [name, a, shape] of tensors) {
    const dtype = a instanceof Float32Array ? "F32" : a instanceof Uint16Array ? "U16" : "U8";
    header[name] = { dtype, shape, data_offsets: [off, off + a.byteLength] };
    off += a.byteLength;
  }
  let hs = new TextEncoder().encode(JSON.stringify(header));
  const pad = (8 - (hs.length % 8)) % 8;
  if (pad) {
    const h2 = new Uint8Array(hs.length + pad).fill(0x20);
    h2.set(hs);
    hs = h2;
  }
  const out = new Uint8Array(8 + hs.length + off);
  new DataView(out.buffer).setBigUint64(0, BigInt(hs.length), true);
  out.set(hs, 8);
  let p = 8 + hs.length;
  for (const [, a] of tensors) {
    out.set(new Uint8Array(a.buffer, a.byteOffset, a.byteLength), p);
    p += a.byteLength;
  }
  return out;
}
