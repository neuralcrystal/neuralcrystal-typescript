// THE SPLIT CRYSTAL (the Python package's run_split, quad.split in the file): input head A on frame fa and input head B on frame fb (pass 1,
// side by side), each picture with its own frame as the residual; the controller's per-pixel combine clip(a·x + b·y + c) to the DMD's
// levels; the middle sections in a row; then either every output section on the same last picture, or (quad.split.branch) each head its own
// row of sections ending in its output. Craftax's crystals are split crystals.
import type { Crop } from "../light/backend.js";
import type { Field } from "../light/field.js";
import { makeField } from "../light/field.js";
import { handoffPicture } from "../hardware/handoff.js";
import type { CrystalEngine } from "./crystal-engine.js";
import { combinePictures, uncrop } from "./fields.js";
import type { LightTrace, StageTrace, SplitResult } from "./types.js";

/** runSplit: exits in quad.split.out order; hiddens are A's picture, B's, then each middle section's. With reuseReads the exits are kept fields. */
export async function runSplit(e: CrystalEngine, fa: Float32Array, fb: Float32Array, F: number): Promise<SplitResult> {
  const sp = e.data.model.quad?.split;
  if (!sp) throw new Error("runSplit: not a split crystal (no quad.split)");
  const q = e.quad, m = e.data.model, g = e.geometry, N = g.N, reuse = e.reuseReads, k = e.scratch;
  const into = reuse ? (k.readIn ??= makeField(N)) : undefined, out = reuse ? (k.readOut ??= makeField(N)) : undefined;
  const hiddens: Float32Array[] = [];
  for (const [f, sec] of [[fa, sp.in[0]!], [fb, sp.in[1]!]] as const) {
    const exit = await e.backend.runSection(e.dmd(f, F, false, into), e.plan(sec), out);
    hiddens.push(handoffPicture(exit, m, g, q, F, 0, { board: fa, stageInput: f }));
    e.alive();
  }
  let x = combinePictures(e.combine(F), hiddens[0]!, hiddens[1]!, q.levels);
  for (let j = 0; j < sp.mid.length; j++) {
    const exit = await e.backend.runSection(e.dmd(x, F, true, into), e.plan(sp.mid[j]!), out);
    x = handoffPicture(exit, m, g, q, F, 1 + j, { board: fa, stageInput: x });
    hiddens.push(x);
    e.alive();
  }
  if (sp.branch?.length) {
    // each head its own path of sections after the shared middle ([..., its out section]); the branches' sections at one depth fly side by
    // side in one pass. entry: the first branch's last input.
    const k0 = 1 + sp.mid.length, exits: Field[] = [];
    let entry: Field | null = null;
    for (let b = 0; b < sp.branch.length; b++) {
      const br = sp.branch[b]!;
      let y = x;
      for (let j = 0; j < br.length - 1; j++) {
        const exit = await e.backend.runSection(e.dmd(y, F, true, into), e.plan(br[j]!), out);
        y = handoffPicture(exit, m, g, q, F, k0 + j, { board: fa, stageInput: y });
        hiddens.push(y);
        e.alive();
      }
      const en = e.dmd(y, F, true, reuse ? (k.branchIns[b] ??= makeField(N)) : undefined), o = reuse ? (k.splitOuts[b] ??= makeField(N)) : undefined;
      entry ??= en;
      exits.push(await e.backend.runSection(en, e.plan(br[br.length - 1]!), o));
    }
    return { exits, entry: entry!, hiddens };
  }
  const entry = e.dmd(x, F, true, into), exits: Field[] = [];
  for (let s = 0; s < sp.out.length; s++) {
    const o = reuse ? (k.splitOuts[s] ??= makeField(N)) : undefined;
    exits.push(await e.backend.runSection(entry, e.plan(sp.out[s]!), o));
  }
  return { exits, entry, hiddens };
}

/**
 * A SPLIT CRYSTAL'S STEP, TRACED (rays for all its sections): runSplit surface by surface, over the window, as a StageTrace in section order —
 * input heads A and B (one stage), the middle sections, output heads 1 and 2 (one stage). hiddens are what each pass's DMD shows after the
 * first (pass k's is hiddens[k − 1]): B's frame, the combined picture, each middle section's, then the last one for both outputs. run.exit is
 * output head 1's exit; exits holds both, as runSplit's.
 */
export async function traceSplit(e: CrystalEngine, fa: Float32Array, fb: Float32Array, F: number, crop: Crop | null): Promise<StageTrace & { exits: Field[]; pictures: Float32Array[] }> {
  const sp = e.data.model.quad?.split;
  if (!sp) throw new Error("traceSplit: not a split crystal (no quad.split)");
  const q = e.quad, m = e.data.model, g = e.geometry, N = g.N;
  const passes: LightTrace[] = [], pictures: Float32Array[] = [];
  const exitOf = (tr: LightTrace): Field => uncrop(tr.arriving[tr.arriving.length - 1]!, tr.crop, N);
  for (const [f, sec] of [[fa, sp.in[0]!], [fb, sp.in[1]!]] as const) {
    const tr = await e.traceSection(e.dmd(f, F, false), sec, crop ?? undefined);
    tr.turn = 0;
    passes.push(tr);
    pictures.push(handoffPicture(exitOf(tr), m, g, q, F, 0, { board: fa, stageInput: f }));
    e.alive();
  }
  let x = combinePictures(e.combine(F), pictures[0]!, pictures[1]!, q.levels);
  const shown: Float32Array[] = [fb, x];
  for (let j = 0; j < sp.mid.length; j++) {
    const tr = await e.traceSection(e.dmd(x, F, true), sp.mid[j]!, crop ?? undefined);
    tr.turn = 1 + j;
    passes.push(tr);
    x = handoffPicture(exitOf(tr), m, g, q, F, 1 + j, { board: fa, stageInput: x });
    pictures.push(x);
    if (j < sp.mid.length - 1) shown.push(x);
    e.alive();
  }
  if (sp.branch?.length) {
    // each branch's sections in turn (their stages side by side: depth j flies at stage 1 + mid + j), passes in section order
    const k0 = 1 + sp.mid.length, depth = Math.max(...sp.branch.map((br) => br.length));
    const ys: Float32Array[] = sp.branch.map(() => x), byStage: { sec: number; tr: LightTrace; pic: Float32Array }[] = [];
    const exits: Field[] = new Array(sp.branch.length);
    let entry: Field | null = null;
    for (let j = 0; j < depth; j++) for (let b = 0; b < sp.branch.length; b++) {
      const br = sp.branch[b]!;
      if (j >= br.length) continue;
      const shownIn = ys[b]!, en = e.dmd(shownIn, F, true), tr = await e.traceSection(en, br[j]!, crop ?? undefined);
      tr.turn = k0 + j;
      if (j === br.length - 1) {
        if (b === 0) { tr.aim = true; entry = en; }
        exits[b] = exitOf(tr);
      } else {
        ys[b] = handoffPicture(exitOf(tr), m, g, q, F, k0 + j, { board: fa, stageInput: shownIn });
        pictures.push(ys[b]!);
      }
      byStage.push({ sec: br[j]!, tr, pic: shownIn });
      e.alive();
    }
    byStage.sort((u, v) => u.sec - v.sec);
    for (const p of byStage) { passes.push(p.tr); shown.push(p.pic); }
    return { passes, hiddens: shown, run: { exit: exits[0]!, entry: entry!, hiddens: pictures }, exits, pictures };
  }
  const entry = e.dmd(x, F, true), exits: Field[] = [];
  for (let s = 0; s < sp.out.length; s++) {
    const tr = await e.traceSection(entry, sp.out[s]!, crop ?? undefined);
    tr.turn = 1 + sp.mid.length;
    if (s === 0) tr.aim = true;
    passes.push(tr);
    shown.push(x);
    exits.push(exitOf(tr));
    e.alive();
  }
  return { passes, hiddens: shown, run: { exit: exits[0]!, entry, hiddens: pictures }, exits, pictures };
}
