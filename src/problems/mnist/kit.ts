// MNIST: a handwritten digit on the DMD, ten detector squares on the exit face, the brightest one is the answer.
// Frames and reading rules are the trainer's, so a crystal reads here as it was trained.
import type { CrystalData } from "../../crystal/types.js";
import { mirrorFrame } from "../../crystal/mirror.js";
import type { CrystalEngine } from "../../engine/crystal-engine.js";
import type { RunResult } from "../../engine/types.js";
import type { ClassReading, DetectorLayout, Rect } from "../../hardware/detectors.js";
import { detectorLayout, detectorLight, detectorRects, readClasses } from "../../hardware/detectors.js";
import type { Panel } from "../../hardware/screen/panel.js";
import type { ProblemKit } from "../../kits/types.js";
import { MNIST_DIGIT, MNIST_FRAME, digitFrame } from "./dataset.js";
import type { DigitScreenView } from "./digit-screen.js";
import { paintDigitScreen } from "./digit-screen.js";

/** a digit: 28 × 28 bytes (the data set) or 0..1 floats (a normalised drawing) */
export type MnistInput = Uint8Array | Float32Array;

export interface MnistAnswer extends ClassReading {
  /** the squares' light over the light the DMD launched (how much of the input the crystal delivered to an answer) */
  efficiency: number;
}

export interface MnistKit extends ProblemKit<MnistInput, MnistAnswer, DigitScreenView> {
  readonly layout: DetectorLayout;
  /** the detector squares on the N grid, for the scene and the sensor tile */
  rects(engine: CrystalEngine): Rect[];
}

/** The pen, in digit-field pixels: 2.1 when it touches, fattening to 2.8 over 0.9 s of dwell (calibrated on the set). */
export const MNIST_PEN = { w0: 2.1, w1: 2.8, ms: 900 };

export function mnistKit(data: CrystalData): MnistKit {
  const tr = (data.model["tr"] ?? {}) as { det?: string };
  const mn = (data.model["mnist"] ?? {}) as { det?: string };
  const layout = detectorLayout(tr.det ?? mn.det);
  let cache: { key: string; rects: Rect[] } | null = null;
  const rects = (engine: CrystalEngine): Rect[] => {
    const key = `${engine.geometry.N}/${engine.geometry.winFrac}`;
    if (cache?.key !== key) {
      const N = engine.geometry.N, r = detectorRects(layout, engine.geometry);
      cache = { key, rects: data.mirrored ? r.map((q) => ({ ...q, x0: N - q.x0 - q.w })) : r }; // a mirrored crystal's squares, mirrored
    }
    return cache.rects;
  };
  return {
    id: "mnist",
    label: "MNIST",
    frameSize: MNIST_FRAME,
    layout,
    rects,
    frame(input: MnistInput): Float32Array {
      if (input.length !== MNIST_DIGIT * MNIST_DIGIT) throw new Error(`an MNIST digit is 28 × 28, got ${input.length} values`);
      const f = digitFrame(input instanceof Uint8Array ? input : Uint8Array.from(input, (v) => Math.round(255 * Math.min(1, Math.max(0, v)))));
      return data.mirrored ? mirrorFrame(f, MNIST_FRAME) : f; // a mirrored crystal sees the picture mirrored
    },
    read(engine: CrystalEngine, run: RunResult): MnistAnswer {
      const reading = readClasses(detectorLight(run.exit, rects(engine)), layout);
      let pin = 0;
      for (let k = 0; k < run.entry.re.length; k++) pin += run.entry.re[k]! ** 2 + run.entry.im[k]! ** 2;
      return { ...reading, efficiency: pin > 0 ? reading.total / pin : 0 };
    },
    paint(panel: Panel, view: DigitScreenView): void {
      paintDigitScreen(panel, view);
    },
  };
}
