// MNIST-Hand digits (QMNIST shards, whole writers per shard) and the frame the crystal was trained on:
// the 28 × 28 digit at +2,+2 in a 32 × 32 frame, amplitude = pixel / 255, as in training.

export const MNIST_FRAME = 32;
export const MNIST_PAD = 2;
export const MNIST_DIGIT = 28;

export interface DigitShard {
  n: number;
  w: number;
  h: number;
  /** n × h × w bytes, row-major */
  pixels: Uint8Array;
  labels: number[];
  writers?: number[];
}

interface ShardJson {
  n: number;
  w: number;
  h: number;
  data: string;
  labels: number[];
  writers?: number[];
}

export function parseShard(json: string | ShardJson): DigitShard {
  const d: ShardJson = typeof json === "string" ? (JSON.parse(json) as ShardJson) : json;
  const bin = atob(d.data);
  const pixels = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) pixels[i] = bin.charCodeAt(i);
  if (pixels.length !== d.n * d.w * d.h) throw new Error(`shard holds ${pixels.length} bytes, expected ${d.n}×${d.h}×${d.w}`);
  return { n: d.n, w: d.w, h: d.h, pixels, labels: d.labels, ...(d.writers ? { writers: d.writers } : {}) };
}

/** Digit i of a shard, 28 × 28 bytes. */
export function digitOf(s: DigitShard, i: number): Uint8Array {
  const sz = s.w * s.h;
  return s.pixels.subarray(i * sz, (i + 1) * sz);
}

/** A 28 × 28 digit (bytes 0..255) → the 32 × 32 frame, amplitudes 0..1. */
export function digitFrame(digit: Uint8Array): Float32Array {
  const F = MNIST_FRAME;
  const out = new Float32Array(F * F);
  for (let y = 0; y < MNIST_DIGIT; y++) for (let x = 0; x < MNIST_DIGIT; x++) out[(y + MNIST_PAD) * F + x + MNIST_PAD] = digit[y * MNIST_DIGIT + x]! / 255;
  return out;
}

/** The first digit of `label` in a shard at or after `from`, wrapping — how a dial finds a fresh hand for each figure. */
export function findDigit(s: DigitShard, label: number, from = 0): number {
  for (let k = 0; k < s.n; k++) {
    const i = (from + k) % s.n;
    if (s.labels[i] === label) return i;
  }
  return -1;
}
