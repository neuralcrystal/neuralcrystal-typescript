// CIFAR-10 (problem "cifarwin"): a photograph on the DMD in the "mega" code (frame.ts), four passes through
// the quad crystal, and ten differential detector pairs on the last section's exit face — class d reads (p⁺ − p⁻)/(p⁺ + p⁻), the largest
// wins, as in training. Optionally the mirrored picture is read too and the pair scores added — the flip
// view the crystal's 68 % was measured with; the simulation reads one view.
import type { CrystalData } from "../../crystal/types.js";
import type { CrystalEngine } from "../../engine/crystal-engine.js";
import type { RunResult } from "../../engine/types.js";
import type { ClassReading, DetectorLayout, Rect } from "../../hardware/detectors.js";
import { detectorLayout, detectorLight, detectorRects, readClasses } from "../../hardware/detectors.js";
import type { Panel } from "../../hardware/screen/panel.js";
import type { ProblemKit } from "../../kits/types.js";
import type { CifarPicture } from "./dataset.js";
import { CIFAR_NAMES } from "./dataset.js";
import type { CifarFrameSpec } from "./frame.js";
import { cifarFrame } from "./frame.js";
import type { CifarScreenView } from "./screen.js";
import { paintCifarScreen } from "./screen.js";

export interface CifarAnswer extends ClassReading {
  /** the class read, by name */
  name: string;
}

export interface CifarKit extends ProblemKit<CifarPicture, CifarAnswer, CifarScreenView> {
  readonly layout: DetectorLayout;
  readonly spec: CifarFrameSpec;
  /** the 20 squares on the N grid (0..9 the + squares, 10..19 the − squares) */
  rects(engine: CrystalEngine): Rect[];
  /** the squares' light on an exit field, read as classes */
  readExit(engine: CrystalEngine, exit: RunResult["exit"]): CifarAnswer;
  /** two readings' pair scores added (the flip view) */
  combine(a: CifarAnswer, b: CifarAnswer): CifarAnswer;
}

export function cifarKit(data: CrystalData): CifarKit {
  const m = data.model as unknown as { tr?: { det?: string; cifar?: { gcn?: number; p4?: string; frame?: number; code?: string } }; quad?: { mir?: number; levels?: number } };
  const c = m.tr?.cifar ?? {}, mir = m.quad?.mir ?? 2;
  if ((c.code ?? "mega") !== "mega") throw new Error(`cifar: the "${c.code}" frame code is not ported (only "mega")`);
  const spec: CifarFrameSpec = { frame: c.frame ?? 256, levels: m.quad?.levels || mir * mir + 1, gcn: c.gcn ?? 0, p4: (c.p4 ?? "dark") as CifarFrameSpec["p4"] };
  const layout = detectorLayout(m.tr?.det ?? "diff");
  let cache: { key: string; rects: Rect[] } | null = null;
  const rects = (engine: CrystalEngine): Rect[] => {
    const key = `${engine.geometry.N}/${engine.geometry.winFrac}`;
    if (cache?.key !== key) cache = { key, rects: detectorRects(layout, engine.geometry) };
    return cache.rects;
  };
  const readExit = (engine: CrystalEngine, exit: RunResult["exit"]): CifarAnswer => {
    const r = readClasses(detectorLight(exit, rects(engine)), layout);
    return { ...r, name: CIFAR_NAMES[r.pick] ?? String(r.pick) };
  };
  return {
    id: "cifar",
    label: "CIFAR-10",
    frameSize: spec.frame,
    layout,
    spec,
    rects,
    readExit,
    combine(a: CifarAnswer, b: CifarAnswer): CifarAnswer {
      const scores = a.scores.map((v, d) => v + b.scores[d]!);
      let pick = 0;
      for (let d = 1; d < scores.length; d++) if (scores[d]! > scores[pick]!) pick = d;
      return { ...a, scores, pick, name: CIFAR_NAMES[pick] ?? String(pick) };
    },
    frame(p: CifarPicture): Float32Array {
      return cifarFrame(p, spec);
    },
    read(engine: CrystalEngine, run: RunResult): CifarAnswer {
      return readExit(engine, run.exit);
    },
    paint(panel: Panel, view: CifarScreenView): void {
      paintCifarScreen(panel, view);
    },
  };
}
