// A section of the crystal as a list of operations — hops and phase planes — compiled once from the recipe, the way
// the Python package's Crystal.forward walks the planes. Light backends (CPU, WebGPU) execute plans; they never read the recipe.
import type { CrystalData, LensArraySpec, PlaneSpec } from "../crystal/types.js";
import type { Geometry } from "./geometry.js";
import { hopPlan } from "./geometry.js";
import { NON_FIXED, fixedPhase, isSteps, spreadMap, spreadSteps } from "./phases.js";

export type Op =
  /** propagate n sub-hops of zSub µm each */
  /** pupil: the first sub-hop also applies the DMD relay's pupil (dmdNA, the Python package's dmd()): folded into its H, the same FFT */
  | { op: "hop"; zSub: number; n: number; pupil?: boolean }
  /** multiply by e^{iφ}; `key` identifies the map so backends can keep it on the GPU across solves */
  /** levels: every value is k·2π/levels (an 8-bit crystal's learned maps, 256) — a backend may hold k alone; steps: those k, N×N, when the
   *  crystal keeps its maps as steps (no radians made for the GPU at all) */
  | { op: "phase"; key: string; phi: () => Float32Array; free?: () => void; levels?: number; steps?: () => Uint8Array };

export interface SectionPlan {
  section: number;
  ops: Op[];
}

interface Role {
  kind: string;
  param: number;
  arr: LensArraySpec | null;
}

/** The kind, parameter and lens array of plane p in section sec (per-section roles when the plane has `kinds`). */
export function roleOf(p: PlaneSpec, sec: number): Role {
  if (!p.kinds) return { kind: p.kind, param: p.param ?? 0, arr: p.arr ?? null };
  const kind = sec < p.kinds.length ? p.kinds[sec]! : "glass";
  const param = (p.params && sec < p.params.length ? p.params[sec] : p.param) ?? 0;
  const arr = (p.arrs && sec < p.arrs.length ? p.arrs[sec] : p.arr) ?? null;
  return { kind, param, arr };
}

export function planSection(c: CrystalData, g: Geometry, sec: number): SectionPlan {
  const ops: Op[] = [];
  const hop = (zUm: number): void => {
    if (zUm > 0) ops.push({ op: "hop", ...hopPlan(g, zUm) });
  };
  const multi = c.model.planes.some((p) => !!p.kinds);
  hop(g.inGapUm);
  pupilFirst(ops, g);
  let pend = 0; // µm of unetched glass waiting to be propagated (per-section roles merge them into the next hop)
  c.model.planes.forEach((p, i) => {
    const { kind, param, arr } = roleOf(p, sec);
    if (!p.off && kind !== "glass") {
      if (pend > 0) {
        hop(pend);
        pend = 0;
      }
      if (kind === "nl") throw new Error("a surface of kind \"nl\" is not supported by the engine");
      if (kind === "learned") ops.push(learnedOp(c, g, i, sec));
      else if (!NON_FIXED.has(kind)) ops.push(phaseOp(`fixed:${i}:${kind}:${param}:${JSON.stringify(arr)}`, () => fixedPhase(g, kind, param, arr)));
    }
    const dUm = (p.dist_mm ?? 0) * 1000;
    if (multi) pend += dUm;
    else hop(dUm);
  });
  hop(pend);
  return { section: sec, ops };
}

/**
 * The section surface by surface, for drawing the light: the input gap, then for each plane its phase
 * (empty when the plane is off, glass or a relay-only kind) and the wafer behind it. Run in turn, these give the field arriving at and
 * leaving every surface. Glass is never merged into the next hop here, so a multi-section crystal's numbers differ from planSection's
 * by rounding only.
 */
export interface SectionStages {
  gap: SectionPlan;
  planes: { phase: SectionPlan; hop: SectionPlan }[];
}

export function planStages(c: CrystalData, g: Geometry, sec: number): SectionStages {
  const hopOps = (zUm: number): Op[] => (zUm > 0 ? [{ op: "hop", ...hopPlan(g, zUm) }] : []);
  const planes = c.model.planes.map((p, i) => {
    const { kind, param, arr } = roleOf(p, sec);
    const ops: Op[] = [];
    if (!p.off && kind !== "glass") {
      if (kind === "nl") throw new Error("a surface of kind \"nl\" is not supported by the engine");
      if (kind === "learned") ops.push(learnedOp(c, g, i, sec));
      else if (!NON_FIXED.has(kind)) ops.push(phaseOp(`fixed:${i}:${kind}:${param}:${JSON.stringify(arr)}`, () => fixedPhase(g, kind, param, arr)));
    }
    return { phase: { section: sec, ops }, hop: { section: sec, ops: hopOps((p.dist_mm ?? 0) * 1000) } };
  });
  const gap = hopOps(g.inGapUm);
  pupilFirst(gap, g);
  return { gap: { section: sec, ops: gap }, planes };
}

/** dmdNA > 0: every picture the DMD launches (the frame, each hidden picture) passes the relay's pupil before the input gap. The
 * pupil is a mask in the same frequency space as the gap's H, so it rides on the first sub-hop (no absorbing rim comes between them in
 * the Python package either); with no input gap, a hop of 0 µm (H = 1 inside the pupil) carries it. */
function pupilFirst(ops: Op[], g: Geometry): void {
  if (!(g.dmdNA > 0)) return;
  const first = ops[0];
  if (first?.op === "hop") ops[0] = { ...first, pupil: true };
  else ops.unshift({ op: "hop", zSub: 0, n: 1, pupil: true });
}

function learnedOp(c: CrystalData, g: Geometry, i: number, sec: number): Op {
  const map = c.maps[i]?.[sec] ?? null;
  const n = map ? Math.round(Math.sqrt(map.length)) : 0;
  // a section with no map of its own is flat glass (the Python package's learned_phase returns zeros)
  const op = phaseOp(`learned:${i}:${sec}`, () => (map ? spreadMap(map, n, g.N) : new Float32Array(g.N * g.N)), c.bits === 8 ? 256 : undefined);
  if (map && isSteps(map)) (op as { steps?: () => Uint8Array }).steps = () => spreadSteps(map as unknown as Uint8Array, n, g.N);
  return op;
}

/** a phase op whose map is made once and kept (the CPU reads it every run); free() lets it go (the GPU, once it holds its own copy) */
function phaseOp(key: string, f: () => Float32Array, levels?: number): Op {
  let v: Float32Array | undefined;
  return { op: "phase", key, phi: () => (v ??= f()), free: () => (v = undefined), ...(levels && { levels }) };
}
