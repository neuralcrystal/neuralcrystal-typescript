// A loaded crystal, ready to run: its geometry, its section plans, the hand-off between stages, and a light backend.
// run(frame) is the Python package's Crystal.run for one sample: every pass, the hidden pictures between them, the last exit field.
import type { CrystalData } from "../crystal/types.js";
import type { Crop, LightBackend } from "../light/backend.js";
import type { Field } from "../light/field.js";
import { makeField } from "../light/field.js";
import type { Geometry } from "../physics/geometry.js";
import { geometryOf } from "../physics/geometry.js";
import type { SectionPlan, SectionStages } from "../physics/plan.js";
import { planSection, planStages } from "../physics/plan.js";
import { dmdField } from "../hardware/dmd.js";
import type { HandoffConfig } from "../hardware/handoff.js";
import { handoffPicture, handoffConfig } from "../hardware/handoff.js";
import { binThrough, imagerResponse, readoutMap } from "../hardware/sensor.js";
import type { ReadoutMap } from "../hardware/sensor.js";
import { CpuBackend } from "../cpu/cpu-backend.js";
import { blockMean, combineWeights, cropField, cropShape, uncrop } from "./fields.js";
import { runSplit, traceSplit } from "./split.js";
import type { LightTrace, PassHook, StageTrace, RunResult, SplitResult } from "./types.js";
import { EngineDisposed } from "./types.js";

/** fields the engine keeps and writes over, so a game loop allocates nothing (see reuseReads) */
interface Scratch {
  readIn: Field | null;
  readOut: Field | null;
  /** a traced step's DMD field for each pass, and the whole-grid exits of every pass but the last */
  traceIn: Field | null;
  traceExit: Field | null;
  splitOuts: Field[];
  branchIns: Field[];
}

export class CrystalEngine {
  readonly geometry: Geometry;
  readonly quad: HandoffConfig;
  readonly backend: LightBackend;
  private readonly plans = new Map<number, SectionPlan>();
  private readonly stagePlans = new Map<number, SectionStages>();
  private readonly maps = new Map<string, ReadoutMap>();
  /** @internal */
  readonly scratch: Scratch = { readIn: null, readOut: null, traceIn: null, traceExit: null, splitOuts: [], branchIns: [] };

  constructor(readonly data: CrystalData, backend?: (g: Geometry) => LightBackend) {
    const m = data.model;
    if (m.overlay?.on) throw new Error("the overlay path is not supported by the engine yet");
    if (m.planes.some((p) => p.kind === "nl" || p.kinds?.includes("nl"))) throw new Error("this crystal has a surface of kind \"nl\", which the engine does not support");
    this.geometry = geometryOf(m);
    this.quad = handoffConfig(m);
    this.backend = backend ? backend(this.geometry) : new CpuBackend(this.geometry);
  }

  /** The plan of physical section `sec`, compiled on first use. */
  plan(sec: number): SectionPlan {
    let p = this.plans.get(sec);
    if (!p) {
      p = planSection(this.data, this.geometry, sec);
      this.plans.set(sec, p);
    }
    return p;
  }

  /**
   * The light surface by surface for one frame, on a single-section crystal: what rays and lit reliefs are drawn from. One small run per
   * surface, so it costs a readback each — call it once a solve, not in a loop. (A multi-stage crystal is traced with traceStages.)
   */
  async trace(frame: Float32Array, F: number): Promise<LightTrace> {
    if (this.quad.hops > 1) throw new Error("trace: a multi-stage crystal is traced with traceStages");
    const reuse = this.backend.runStages ? this.freeTraces.pop() : undefined;
    return this.traceSection(this.dmd(frame, F, false, reuse?.input), this.quad.passSection[0] ?? 0, undefined, reuse);
  }

  /** the window a whole-glass crystal's light lives in (the maps are zero outside it): the square a multi-stage trace is cut to */
  get window(): Crop {
    const g = this.geometry, n = Math.max(1, Math.round(g.N * g.winFrac));
    return { x0: Math.round((g.N - n) / 2), y0: Math.round((g.N - n) / 2), n };
  }

  /**
   * A MULTI-STAGE CRYSTAL'S STEP, TRACED: each pass surface by surface — over its window only (crop; at 1024² a pass's 21 fields are 176 MB
   * whole, 44 MB over a 512 window) — and between passes the hidden picture, made exactly as run() makes it (the exit field is whole again for
   * it: zeros outside the window, where the maps are zero too). onPass: told after each pass with the passes so far and their hidden
   * pictures, and awaited, so a caller can draw them while the rest solve.
   */
  async traceStages(frame: Float32Array, F: number, crop: Crop | null = this.window, onPass?: (passes: LightTrace[], hiddens: Float32Array[]) => void | Promise<void>): Promise<StageTrace> {
    const q = this.quad, g = this.geometry, k = this.scratch;
    const Fh = q.hidden || F;
    if (Fh !== F && (q.ghost || q.band)) throw new Error("quad.ghost / quad.band cannot be used with quad.hidden");
    let stageInput = Fh === F ? frame : blockMean(frame, F, Fh);
    // with reuseReads and a window, each pass's DMD field and the exits between passes go in kept scratch: only the last exit leaves here
    const scratch = this.reuseReads && !!crop;
    const into = scratch ? (k.traceIn ??= makeField(g.N)) : undefined;
    const set = scratch ? this.ring[(this.ringAt ^= 1)]! : null;
    let entry = this.dmd(frame, F, false, into);
    const passes: LightTrace[] = [], hiddens: Float32Array[] = [];
    let exit: Field = entry;
    for (let p = 0; p < q.hops; p++) {
      if (p > 0) {
        const pic = handoffPicture(exit, this.data.model, g, q, Fh, p - 1, { board: frame, stageInput });
        hiddens.push(pic);
        stageInput = pic;
        entry = this.dmd(pic, Fh, true, into);
      }
      this.alive();
      const keep = set && crop ? { input: (set.input[p] ??= cropShape(crop.n)), exit: (set.exit[p] ??= cropShape(crop.n)) } : undefined;
      const tr = await this.traceSection(entry, q.passSection[p] ?? p, crop ?? undefined, undefined, keep);
      this.alive();
      passes.push(tr);
      await onPass?.([...passes], [...hiddens]);
      // the passes between share one scratch field; the step's own exit (what the readout and the caller keep) is its set's
      const last = p === q.hops - 1;
      exit = uncrop(tr.arriving[tr.arriving.length - 1]!, tr.crop, g.N, scratch ? (last ? (set!.run ??= makeField(g.N)) : (k.traceExit ??= makeField(g.N))) : undefined);
      if (this.passDone) {
        await this.passDone(tr, p, last);
        this.alive();
      }
    }
    return { passes, hiddens, run: { exit, entry, hiddens } };
  }

  /** @deprecated the old name of traceStages */
  traceQuad(frame: Float32Array, F: number, crop: Crop | null = this.window, onPass?: (passes: LightTrace[], hiddens: Float32Array[]) => void | Promise<void>): Promise<StageTrace> {
    return this.traceStages(frame, F, crop, onPass);
  }

  /**
   * one section's light surface by surface from an entry field (cropped to `crop` when given); keep: where a reused traced step puts the
   * pass's window-cut input and its exit instead of new arrays
   * @internal
   */
  async traceSection(input: Field, section: number, crop?: Crop, reuse?: LightTrace, keep?: { input: Field; exit: Field }): Promise<LightTrace> {
    let st = this.stagePlans.get(section);
    if (!st) {
      st = planStages(this.data, this.geometry, section);
      this.stagePlans.set(section, st);
    }
    const b = this.backend, N = this.geometry.N;
    const cut = (f: Field): Field => (crop ? cropField(f, crop, keep?.input) : f);
    const tag = crop ? { crop: { ...crop, N } } : {};
    if (b.runStages) {
      // every surface in one go: [gap, phase₀, hop₀, phase₁, hop₁, …], one submission and one readback (the GPU); into a recycled trace's
      // fields when there is one
      const plans = [st.gap, ...st.planes.flatMap((p) => [p.phase, p.hop])];
      const into = reuse && reuse.leaving.length === st.planes.length ? [reuse.arrival, ...reuse.leaving.flatMap((l, i) => [l, reuse.arriving[i + 1]!])] : undefined;
      const f = await b.runStages(input, plans, crop || this.halfTraces ? undefined : into, crop, this.halfTraces, keep?.exit);
      const arrival = f[0]!, leaving: Field[] = [], arriving: Field[] = [arrival];
      st.planes.forEach((_, i) => {
        leaving.push(f[1 + 2 * i]!);
        arriving.push(f[2 + 2 * i]!);
      });
      return { input: cut(input), arrival, leaving, arriving, ...tag };
    }
    const arrival = st.gap.ops.length ? await b.runSection(input, st.gap) : input;
    const leaving: Field[] = [], arriving: Field[] = [arrival];
    let E = arrival;
    for (const p of st.planes) {
      this.alive(); // a disposed engine stops between surfaces too
      E = p.phase.ops.length ? await b.runSection(E, p.phase) : E;
      leaving.push(E);
      E = p.hop.ops.length ? await b.runSection(E, p.hop) : E;
      arriving.push(E);
    }
    if (!crop) return { input, arrival, leaving, arriving };
    return { input: cut(input), arrival: cut(arrival), leaving: leaving.map(cut), arriving: arriving.map(cut), ...tag };
  }

  /** An F×F picture on the DMD → the field it launches. hidden: a picture handed between stages (the polarity switch does not apply). */
  dmd(frame: Float32Array, F: number, hidden = false, out?: Field): Field {
    const im = this.data.model.imager ?? {};
    return dmdField(this.geometry, frame, F, { inv: !!im.inv, flip: im.flip !== false }, hidden, out);
  }

  /**
   * Traces a caller is done with, to be written over by the next ones (a trace at 512² is 11 fields, ~25 MB). Only GPU traces (single
   * precision) are kept; hand back only a trace nothing reads any more.
   */
  private readonly freeTraces: LightTrace[] = [];
  /** how many spare traces are kept for reuse (each is ~25 MB at 512²; a phone keeps 1) */
  keepTraces = 3;
  /**
   * HALF-PRECISION TRACES (for phones): on the GPU, every surface's field but the exit is read back as f16 (Float16Array), half the memory;
   * the exit, which the next pass and the readout use, stays f32. Where the runtime has no Float16Array it does nothing. Half traces are
   * never recycled (recycle keeps f32 ones only).
   */
  halfTraces = false;
  /**
   * READS WITHOUT GARBAGE: run() writes the DMD's field and each pass's exit into the same two kept fields instead of new ones (a look-ahead
   * can run 120 reads faster than a phone's garbage collector frees 1024² fields). So run()'s `exit` and `entry` are only good until the next
   * run(): read the answer from them before running again. traceStages keeps two sets of its fields and alternates, so at most two traced steps
   * may be held at once; run() with an onPass hook makes its own fields as before.
   */
  reuseReads = false;
  private readonly ring: { input: Field[]; exit: Field[]; run: Field | null }[] = [{ input: [], exit: [], run: null }, { input: [], exit: [], run: null }];
  private ringAt = 0;

  /** a pass's full fields the caller is done with: the next pass is read back into them instead of new ones (WebGpuBackend.recycleFields) */
  recycleFields(fields: Field[]): void {
    this.backend.recycleFields?.(fields);
  }
  /**
   * told after each pass of a traced step, before the next is solved, and awaited: a caller can build the pass's rays and free its fields
   * here, so a step never holds more than one pass's light. last: the step's last pass.
   */
  passDone: ((tr: LightTrace, k: number, last: boolean) => void | Promise<void>) | null = null;
  recycle(tr: LightTrace): void {
    const f = tr.leaving[0];
    if (!f || !(f.re instanceof Float32Array) || f.N !== this.geometry.N || this.freeTraces.includes(tr) || this.freeTraces.length >= this.keepTraces) return;
    this.freeTraces.push(tr);
  }

  async run(frame: Float32Array, F: number, onPass?: PassHook): Promise<RunResult> {
    const q = this.quad, k = this.scratch;
    const Fh = q.hidden || F;
    if (Fh !== F && (q.ghost || q.band)) throw new Error("quad.ghost / quad.band cannot be used with quad.hidden");
    let stageInput = Fh === F ? frame : blockMean(frame, F, Fh);
    const N = this.geometry.N, reuse = this.reuseReads && !onPass;
    const into = reuse ? (k.readIn ??= makeField(N)) : undefined, out = reuse ? (k.readOut ??= makeField(N)) : undefined;
    let entry = this.dmd(frame, F, false, into);
    let exit = await this.backend.runSection(entry, this.plan(q.passSection[0] ?? 0), out);
    onPass?.(0, exit);
    const hiddens: Float32Array[] = [];
    for (let p = 1; p < q.hops; p++) {
      const pic = handoffPicture(exit, this.data.model, this.geometry, q, Fh, p - 1, { board: frame, stageInput });
      hiddens.push(pic);
      stageInput = pic;
      entry = this.dmd(pic, Fh, true, into);
      this.alive();
      exit = await this.backend.runSection(entry, this.plan(q.passSection[p] ?? p), out);
      onPass?.(p, exit);
    }
    return { exit, entry, hiddens };
  }

  /** A split crystal's step (two input frames): see split.ts. */
  runSplit(fa: Float32Array, fb: Float32Array, F: number): Promise<SplitResult> {
    return runSplit(this, fa, fb, F);
  }

  /** A split crystal's step, traced surface by surface: see split.ts. */
  traceSplit(fa: Float32Array, fb: Float32Array, F: number, crop: Crop | null = this.window): Promise<StageTrace & { exits: Field[]; pictures: Float32Array[] }> {
    return traceSplit(this, fa, fb, F, crop);
  }

  private comb: { F: number; abc: [Float32Array, Float32Array, Float32Array] } | null = null;
  /** the split crystal's combine weights for an F×F picture, built once @internal */
  combine(F: number): [Float32Array, Float32Array, Float32Array] {
    if (this.comb?.F !== F) this.comb = { F, abc: combineWeights(this.data, F) };
    return this.comb.abc;
  }

  /** The readout box binned to F×F through the sensor's pixel map (sums of intensity). */
  readout(E: Field, F: number): Float64Array {
    const g = this.geometry;
    const key = `${g.so}/${F}`;
    let map = this.maps.get(key);
    if (!map) {
      map = readoutMap(this.data.model, g, g.so, F);
      this.maps.set(key, map);
    }
    return binThrough(E, g.xo, g.so, F, map);
  }

  /** The readout through the imager's response (what the camera reports, 0..1). */
  picture(E: Field, F: number): Float64Array {
    return imagerResponse(this.data.model, this.geometry, this.readout(E, F), F);
  }

  private disposed = false;
  /** a run still going when the engine is disposed stops at its next pass (EngineDisposed) instead of finishing for nobody @internal */
  alive(): void {
    if (this.disposed) throw new EngineDisposed();
  }

  dispose(): void {
    this.disposed = true;
    this.backend.dispose();
  }
}
