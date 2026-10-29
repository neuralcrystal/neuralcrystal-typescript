// A problem kit: what the hardware's slots do for one problem. The light engine, a 3D scene
// and a page call the kit; they never ask which problem it is.
import type { CrystalEngine } from "../engine/crystal-engine.js";
import type { LightTrace, RunResult } from "../engine/types.js";
import type { Panel } from "../hardware/screen/panel.js";

export interface ProblemKit<In, Answer, View> {
  readonly id: string;
  readonly label: string;
  /** the DMD frame's size, F × F */
  readonly frameSize: number;
  /** DMD handler: an input → the frame the mirrors show (amplitudes 0..1) */
  frame(input: In): Float32Array;
  /** FPGA (controller) handler: the run's result → the answer, with the problem's own reading rules */
  read(engine: CrystalEngine, run: RunResult, input: In): Answer;
  /** screen handler: draw the player's screen for this state */
  paint(panel: Panel, view: View): void;
}

export interface Solved<Answer> {
  answer: Answer;
  run: RunResult;
  ms: number;
  /** the light at every surface, when solved with solveTraced (a caller draws its rays and glass from it) */
  trace?: LightTrace;
}

/** One input through the whole hardware: frame → light → answer. */
export async function solve<In, Answer, View>(kit: ProblemKit<In, Answer, View>, engine: CrystalEngine, input: In): Promise<Solved<Answer>> {
  const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();
  const run = await engine.run(kit.frame(input), kit.frameSize);
  const answer = kit.read(engine, run, input);
  return { answer, run, ms: (typeof performance !== "undefined" ? performance.now() : Date.now()) - t0 };
}

/**
 * One input through the hardware with the light kept at every surface: ONE pass that is both the answer and the rays to draw (the trace
 * ends at the exit face, which the kit reads). Single-section crystals; a quad crystal falls back to solve().
 */
export async function solveTraced<In, Answer, View>(kit: ProblemKit<In, Answer, View>, engine: CrystalEngine, input: In): Promise<Solved<Answer>> {
  if (engine.quad.hops > 1) return solve(kit, engine, input);
  const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();
  const trace = await engine.trace(kit.frame(input), kit.frameSize);
  const run: RunResult = { exit: trace.arriving[trace.arriving.length - 1]!, entry: trace.input, hiddens: [] };
  const answer = kit.read(engine, run, input);
  return { answer, run, trace, ms: (typeof performance !== "undefined" ? performance.now() : Date.now()) - t0 };
}
