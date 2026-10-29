// Detector squares on the exit face: the light each square collects is a class's score (the trainer's detector tables and squares).
// Centres and half-widths are in window widths, rows downward.
import type { Field } from "../light/field.js";
import type { Geometry } from "../physics/geometry.js";
import { rnd } from "../physics/geometry.js";

type Centre = [number, number];

const TEN: Centre[] = [[-0.3, -0.3], [0, -0.3], [0.3, -0.3], [-0.45, 0], [-0.15, 0], [0.15, 0], [0.45, 0], [-0.3, 0.3], [0, 0.3], [0.3, 0.3]];
const DIFF: Centre[] = [-0.38, -0.14, 0.14, 0.38].flatMap((y) => [-0.4, -0.2, 0, 0.2, 0.4].map((x): Centre => [x, y]));
/** DIFF-WIN: the same twenty in the same order, fitted inside the window crystals' lit square
 *  columns ±0.28 on a 0.14 pitch, rows ±0.07 / ±0.21, squares 0.11 wide */
const DIFF_WIN: Centre[] = [-0.21, -0.07, 0.07, 0.21].flatMap((y) => [-0.28, -0.14, 0, 0.14, 0.28].map((x): Centre => [x, y]));
/** the phone pad: 1-2-3 / 4-5-6 / 7-8-9 / 0 at the bottom middle (entry d is digit d) */
const PAD: Centre[] = [[0, 0.375], [-0.25, -0.375], [0, -0.375], [0.25, -0.375], [-0.25, -0.125], [0, -0.125], [0.25, -0.125], [-0.25, 0.125], [0, 0.125], [0.25, 0.125]];

export interface DetectorLayout {
  name: string;
  centres: Centre[];
  /** half-width, window widths */
  half: number;
  /** differential pairs: entries 0..9 are the + squares, 10..19 the − squares */
  diff: boolean;
}

export const DETECTOR_LAYOUTS: Record<string, DetectorLayout> = {
  ten: { name: "ten", centres: TEN, half: 0.1, diff: false },
  diff: { name: "diff", centres: DIFF, half: 0.08, diff: true },
  diffwin: { name: "diffwin", centres: DIFF_WIN, half: 0.055, diff: true },
  pad: { name: "pad", centres: PAD, half: 0.1, diff: false },
  /** the pad turned 180° — 0 on top, 3-2-1 along the bottom, as an imaging relay shows it */
  padrot: { name: "padrot", centres: PAD.map(([x, y]): Centre => [-x, -y]), half: 0.1, diff: false },
};

export function detectorLayout(name: unknown): DetectorLayout {
  return DETECTOR_LAYOUTS[String(name)] ?? DETECTOR_LAYOUTS["ten"]!;
}

export interface Rect {
  x0: number;
  y0: number;
  w: number;
  h: number;
}

/** Each detector's rectangle in grid samples (det_rect on the N grid). */
export function detectorRects(lay: DetectorLayout, g: Geometry): Rect[] {
  const n = g.N;
  const w = n * g.winFrac;
  const c = (n - 1) / 2;
  const s = rnd(2 * lay.half * w);
  return lay.centres.map(([cx, cy]) => ({ x0: rnd(c + (cx - lay.half) * w), y0: rnd(c + (cy - lay.half) * w), w: s, h: s }));
}

/** The light in each rectangle, Σ|E|². */
export function detectorLight(E: Field, rects: Rect[]): Float64Array {
  const N = E.N;
  const out = new Float64Array(rects.length);
  rects.forEach((r, d) => {
    let t = 0;
    for (let y = Math.max(0, r.y0); y < Math.min(N, r.y0 + r.h); y++)
      for (let x = Math.max(0, r.x0); x < Math.min(N, r.x0 + r.w); x++) {
        const k = y * N + x;
        t += E.re[k]! * E.re[k]! + E.im[k]! * E.im[k]!;
      }
    out[d] = t;
  });
  return out;
}

export interface ClassReading {
  /** the class read: the brightest square, or the largest (p⁺ − p⁻)/(p⁺ + p⁻) for pairs */
  pick: number;
  /** a score per class (light, or the pair contrast) */
  scores: Float64Array;
  /** each class's share of the squares' light (softmax-free, what a page's bars show) */
  shares: Float64Array;
  /** Σ light in the squares */
  total: number;
}

export function readClasses(light: Float64Array, lay: DetectorLayout): ClassReading {
  const p = light.map((v) => v + 1e-12);
  const total = p.reduce((a, b) => a + b, 0);
  let scores: Float64Array;
  let shares: Float64Array;
  if (lay.diff) {
    scores = new Float64Array(10).map((_, d) => (p[d]! - p[d + 10]!) / (p[d]! + p[d + 10]!));
    shares = new Float64Array(10).map((_, d) => p[d]! / total);
  } else {
    scores = p;
    shares = p.map((v) => v / total);
  }
  let pick = 0;
  for (let d = 1; d < scores.length; d++) if (scores[d]! > scores[pick]!) pick = d;
  return { pick, scores, shares, total };
}
