import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { parseSafetensors, readCrystal, tensorF32 } from "../src/index.js";
import type { CrystalData, Field } from "../src/index.js";

export interface Fixture {
  data: CrystalData;
  F: number;
  Fr: number;
  t: (name: string) => Float32Array;
  has: (name: string) => boolean;
  /** the fixture's own notes (__metadata__.fixture) */
  meta: Record<string, unknown>;
}

export function fixture(name: string): Fixture {
  const bytes = new Uint8Array(readFileSync(fileURLToPath(new URL(`./fixtures/${name}.safetensors`, import.meta.url))));
  const file = parseSafetensors(bytes);
  const meta = JSON.parse(file.metadata["fixture"]!) as Record<string, unknown>;
  const { F, Fr } = meta as { F: number; Fr: number };
  const t = (n: string): Float32Array => {
    const a = tensorF32(file, n);
    if (!a) throw new Error(`fixture ${name} has no ${n}`);
    return a;
  };
  const data = file.metadata["neuralcrystal"] ? readCrystal(bytes) : (null as unknown as CrystalData); // mnist-real carries results only
  return { data, F, Fr, t, has: (n) => file.tensors.has(n), meta };
}

/** max |a − b| over max |b| (complex fields compare re and im together) */
export function relErr(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let d = 0;
  let m = 0;
  for (let i = 0; i < b.length; i++) {
    d = Math.max(d, Math.abs(a[i]! - b[i]!));
    m = Math.max(m, Math.abs(b[i]!));
  }
  return d / Math.max(m, 1e-30);
}

export function fieldErr(E: Field, re: Float32Array, im: Float32Array): number {
  let d = 0;
  let m = 0;
  for (let i = 0; i < re.length; i++) {
    d = Math.max(d, Math.hypot(E.re[i]! - re[i]!, E.im[i]! - im[i]!));
    m = Math.max(m, Math.hypot(re[i]!, im[i]!));
  }
  return d / Math.max(m, 1e-30);
}

/** the share of entries that differ by more than tol */
export function mismatch(a: ArrayLike<number>, b: ArrayLike<number>, tol = 1e-6): number {
  let n = 0;
  for (let i = 0; i < b.length; i++) if (Math.abs(a[i]! - b[i]!) > tol) n++;
  return n / b.length;
}

/** a real trained crystal from NC_CRYSTALS (a folder of crystal files; they are large and not shipped), or "" when it isn't set */
export function realCrystal(name: string): string {
  const dir = process.env["NC_CRYSTALS"];
  return dir ? `${dir.replace(/\/$/, "")}/${name}` : "";
}
