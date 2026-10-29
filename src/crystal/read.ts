// Bytes → CrystalData: a v2 safetensors crystal, or a v1 JSON crystal (maps base64-encoded float32).
import type { CrystalData, ModelSpec, PhaseMap } from "./types.js";
import { isSafetensors, parseSafetensors, tensorBytes, tensorF32 } from "./safetensors.js";
import type { SafetensorsFile } from "./safetensors.js";
import type { PackMeta } from "./pack.js";
import { unpackMap, unpackSteps } from "./pack.js";

/**
 * A crystal's layout alone, from its model JSON (the few kB a page can fetch ahead of the crystal file): every surface where it goes, none
 * of them etched yet — enough to draw the whole hardware with a blank crystal while the surfaces download.
 */
export function crystalLayout(modelJson: unknown): CrystalData {
  const model = checkModel(modelJson);
  return { model, maps: model.planes.map(() => [null]), mapN: model.planes.map(() => 0) };
}

/** stepMaps: an 8-bit packed crystal's maps kept as their 8-bit steps (LoadOptions.stepMaps) */
export function readCrystal(bytes: Uint8Array, opts: { stepMaps?: boolean } = {}): CrystalData {
  return isSafetensors(bytes) ? readV2(bytes, !!opts.stepMaps) : readV1(JSON.parse(new TextDecoder().decode(bytes)) as V1Model);
}

function readV2(bytes: Uint8Array, stepMaps = false): CrystalData {
  const file = parseSafetensors(bytes);
  const raw = file.metadata["neuralcrystal"];
  if (!raw) throw new Error("no NeuralCrystal model in the safetensors header");
  const model = checkModel(JSON.parse(raw));
  if (file.metadata["nc.pack"]) return readPacked(file, model, JSON.parse(file.metadata["nc.pack"]) as PackMeta, stepMaps);
  const maps: (PhaseMap | null)[][] = [];
  const mapN: number[] = [];
  model.planes.forEach((_, i) => {
    const secs: (PhaseMap | null)[] = [tensorF32(file, `plane.${i}.phi`)];
    let n = sideOf(secs[0] ?? null);
    for (const name of file.tensors.keys()) {
      const m = name.match(/^plane\.(\d+)\.phi\.(\d+)$/);
      if (!m || Number(m[1]) !== i) continue;
      const k = Number(m[2]);
      while (secs.length <= k) secs.push(null);
      if (k === 0 && secs[0]) continue; // plane.<i>.phi is section 0 (the trainer writes .phi.<k> for k ≥ 1 only)
      secs[k] = tensorF32(file, name);
      n = n || sideOf(secs[k] ?? null);
    }
    maps.push(secs);
    mapN.push(n);
  });
  return { model, maps, mapN };
}

/** the web form (packCrystal): cropped, wrapped, 16- or 8-bit maps, expanded back to full maps */
function readPacked(file: SafetensorsFile, model: ModelSpec, pack: PackMeta, stepMaps = false): CrystalData {
  const steps = stepMaps && pack.bits === 8;
  if (pack.format !== "nc.pack.v1") throw new Error(`unknown crystal packing ${pack.format}`);
  const maps: (PhaseMap | null)[][] = [];
  const mapN: number[] = [];
  model.planes.forEach((_, i) => {
    const secs: (PhaseMap | null)[] = [];
    let n = 0;
    for (const [name, p] of Object.entries(pack.maps)) {
      const m = name.match(/^plane\.(\d+)\.phi(?:\.(\d+))?$/);
      if (!m || Number(m[1]) !== i) continue;
      const k = m[2] ? Number(m[2]) : 0, t = tensorBytes(file, name);
      if (!t) continue;
      while (secs.length <= k) secs.push(null);
      secs[k] = steps ? unpackSteps(t.bytes, t.dtype, p) : unpackMap(t.bytes, t.dtype, p, pack.bits);
      n = p.n;
    }
    if (!secs.length) secs.push(null);
    maps.push(secs);
    mapN.push(n);
  });
  return { model, maps, mapN, bits: pack.bits, ...(steps && { steps: true }) };
}

interface V1Plane {
  phi?: string | null;
  phis?: (string | null)[] | null;
  n?: number;
}
type V1Model = Omit<ModelSpec, "planes"> & { planes: (ModelSpec["planes"][number] & V1Plane)[] };

function readV1(m: V1Model): CrystalData {
  const model = checkModel(m);
  const maps: (PhaseMap | null)[][] = [];
  const mapN: number[] = [];
  m.planes.forEach((p) => {
    const secs: (PhaseMap | null)[] = [p.phi ? b64f32(p.phi) : null];
    (p.phis ?? []).forEach((s, k) => {
      if (k === 0 && secs[0]) return; // section 0 is the plane's own map
      while (secs.length <= k) secs.push(null);
      secs[k] = s ? b64f32(s) : null;
    });
    maps.push(secs);
    mapN.push(p.n ?? sideOf(secs.find((a) => a) ?? null));
    delete p.phi;
    delete p.phis;
  });
  return { model, maps, mapN };
}

function checkModel(m: unknown): ModelSpec {
  const o = m as ModelSpec;
  if (!o || !Array.isArray(o.planes)) throw new Error("not a NeuralCrystal crystal file");
  if (!(o.N > 0) || !(o.dx_um > 0)) throw new Error("crystal file has no grid (N, dx_um)");
  return o;
}

function sideOf(a: Float32Array | null): number {
  return a ? Math.round(Math.sqrt(a.length)) : 0;
}

function b64f32(s: string): Float32Array {
  const bin = atob(s);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(u8.buffer, 0, u8.length >> 2);
}
