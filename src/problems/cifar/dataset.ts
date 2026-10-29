// The CIFAR-10 shards (from Hugging Face uoft-cs/cifar10): 1,000 pictures a shard, 32 × 32 RGB bytes (HWC) as base64, and
// their labels. The demo plays the TEST shards — pictures the crystal never trained on.

export const CIFAR_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"] as const;

export interface CifarShard {
  n: number;
  labels: number[];
  /** n × 32 × 32 × 3 bytes, HWC */
  data: Uint8Array;
}

/** one picture: 32 × 32 × 3 bytes, HWC */
export type CifarPicture = Uint8Array;

export function parseCifarShard(text: string): CifarShard {
  const j = JSON.parse(text) as { n: number; w: number; h: number; c?: number; labels: number[]; data: string };
  if (j.w !== 32 || j.h !== 32 || (j.c ?? 3) !== 3) throw new Error("cifar: not a 32 × 32 RGB shard");
  const bin = atob(j.data), data = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) data[i] = bin.charCodeAt(i);
  if (data.length !== j.n * 3072) throw new Error("cifar: shard data is the wrong size");
  return { n: j.n, labels: j.labels, data };
}

export function cifarPicture(s: CifarShard, i: number): CifarPicture {
  return s.data.subarray(i * 3072, (i + 1) * 3072);
}

/** the picture mirrored left to right (the flip view) */
export function flipPicture(p: CifarPicture): CifarPicture {
  const o = new Uint8Array(3072);
  for (let y = 0; y < 32; y++) for (let x = 0; x < 32; x++) for (let c = 0; c < 3; c++) o[(y * 32 + x) * 3 + c] = p[(y * 32 + 31 - x) * 3 + c]!;
  return o;
}
