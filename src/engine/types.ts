// What the engine hands back: a run's result, a split crystal's run, and the light traced surface by surface for drawing.
import type { Crop } from "../light/backend.js";
import type { Field } from "../light/field.js";

export interface RunResult {
  /** the last pass's exit field */
  exit: Field;
  /** the last pass's input field (what the DMD launched into it) */
  entry: Field;
  /** each hidden picture handed between stages, F×F (Fh×Fh) amplitudes, pass 1 onward */
  hiddens: Float32Array[];
}

/** a split crystal's run (runSplit) */
export interface SplitResult {
  /** each output section's exit field, in quad.split.out order */
  exits: Field[];
  /** the last pass's input field */
  entry: Field;
  /** A's picture, B's, then each middle section's (as the Python package's run_split) */
  hiddens: Float32Array[];
}

/** The light at every surface of one pass, for drawing rays and lit glass. */
export interface LightTrace {
  /** what the DMD launched */
  input: Field;
  /** the input after the input gap: the field arriving at surface 0 */
  arrival: Field;
  /** leaving[i]: the field just after surface i */
  leaving: Field[];
  /** arriving[i]: the field arriving at surface i (arriving[0] = arrival); arriving[planes] = the exit face */
  arriving: Field[];
  /** the fields cover only this square of the grid (a pass's window): each field is crop.n², its [0] at (crop.x0, crop.y0) */
  crop?: Crop & { N: number };
  /**
   * set by a caller that is done with the pass's detail: arrival and arriving[] (all but the exit, which stays whole) kept at 1/slim of
   * the resolution, leaving[] emptied. Nothing traces rays from it again.
   */
  slim?: number;
  /** a split crystal's traced pass (traceSplit): the stage its light flies in (the two input heads share one, the two outputs one) */
  turn?: number;
  /** the pass whose exit face carries the answer (traceSplit: output head 1); absent: the last pass */
  aim?: boolean;
}

/** a multi-stage crystal's step, traced: every pass surface by surface (over its window), the pictures handed between them, and the result */
export interface StageTrace {
  passes: LightTrace[];
  hiddens: Float32Array[];
  /** as run() returns it: the last pass's exit on the whole grid (zeros outside the window) and its entry */
  run: RunResult;
}

export interface PassHook {
  /** called after each pass with its exit field (for light sheets, packets, progress) */
  (pass: number, exit: Field): void;
}

/** Thrown by a run whose engine was disposed under it (the page moved on, so the answer has no reader). Callers can ignore it. */
export class EngineDisposed extends Error {
  constructor() {
    super("the engine was disposed");
    this.name = "EngineDisposed";
  }
}

/** @deprecated the old name of StageTrace */
export type QuadTrace = StageTrace;
