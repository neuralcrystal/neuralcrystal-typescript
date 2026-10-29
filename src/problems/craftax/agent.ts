// The crystal playing Craftax. It is recurrent: it carries a memory (one 4-bit level per unit) from one step to the next, drawn into
// its input picture, and writes the next memory as light. Each step: the board picture and the memory → the light → the move scores
// and the memory update h ← round(15 · clamp(h + z (hc − h), 0, 1)) / 15. Asleep or resting, the game takes no move and the memory is
// held, as the crystals were trained.
import type { CrystalEngine } from "../../engine/crystal-engine.js";
import type { CraftaxLayout } from "./layout.js";
import { craftaxLayout, pictures } from "./layout.js";
import type { CraftaxReading } from "./read.js";
import { cellPowers, readCraftax } from "./read.js";
import type { FrameOptions } from "./frame.js";
import { frame } from "./frame.js";
import type { CraftaxState } from "./state.js";
import { CRAFTAX_N_ACTIONS, actionMask } from "./state.js";

/** the parts of a CrystalEngine the agent uses */
export type CraftaxEngine = Pick<CrystalEngine, "data" | "geometry" | "run" | "runSplit">;

/** One step of the light: the input pictures (two on a split crystal, one otherwise) → the scores and the memory update. */
export async function craftaxStep(engine: CraftaxEngine, L: CraftaxLayout, pics: Float32Array[]): Promise<CraftaxReading> {
  if (L.split) {
    const run = await engine.runSplit(pics[0]!, pics[1]!, L.F);
    return readCraftax(L, cellPowers(engine.geometry, run.exits[0]!), cellPowers(engine.geometry, run.exits[1]!));
  }
  const run = await engine.run(pics[0]!, L.F);
  return readCraftax(L, cellPowers(engine.geometry, run.exit));
}

export interface CraftaxMove extends CraftaxReading {
  /** the move, 0 … 42; on a crystal with a TARGET region, 43 means TARGET (the caller's walker turns it into a press) */
  action: number;
  /** each region's chance: softmax(scores / temp) over the allowed ones */
  probs: Float64Array;
  /** the memory before this step */
  memoryIn: Float32Array;
}

export interface AgentOptions {
  /** the sampling temperature; 0 or less plays the most likely move (default 1) */
  temperature?: number;
  /** a source of uniform numbers in [0, 1) for sampling (default Math.random) */
  random?: () => number;
  /** how the board picture is drawn (the status mode pad, the player tile mark) */
  frame?: FrameOptions;
}

/** The most likely allowed region, or one drawn from softmax(scores / temp) over the allowed ones. */
export function chooseMove(scores: ArrayLike<number>, allowed: ArrayLike<boolean>, temp: number, random: () => number): { best: number; probs: Float64Array } {
  const n = scores.length;
  let best = 0, max = -Infinity;
  for (let i = 0; i < n; i++) if (allowed[i] && scores[i]! > max) { max = scores[i]!; best = i; }
  const probs = new Float64Array(n), k = temp > 0 ? 1 / temp : 1;
  let sum = 0;
  for (let i = 0; i < n; i++) if (allowed[i]) sum += probs[i] = Math.exp((scores[i]! - max) * k);
  for (let i = 0; i < n; i++) probs[i]! /= sum;
  if (temp > 0) {
    let r = random();
    for (let i = 0; i < n; i++) if (allowed[i] && (r -= probs[i]!) <= 0) { best = i; break; }
  }
  return { best, probs };
}

export class CraftaxAgent {
  readonly layout: CraftaxLayout;
  /** the memory, one level 0 … 1 per unit (steps of 1/15); save it with a game, restore it with set */
  memory: Float32Array;
  temperature: number;
  private readonly random: () => number;
  private readonly frameOptions: FrameOptions;

  constructor(readonly engine: CraftaxEngine, o: AgentOptions = {}) {
    this.layout = craftaxLayout(engine.data.model);
    this.memory = new Float32Array(this.layout.units);
    this.temperature = o.temperature ?? 1;
    this.random = o.random ?? Math.random;
    this.frameOptions = o.frame ?? {};
  }

  /** a new game: every memory level back to 0 */
  reset(): void {
    this.memory.fill(0);
  }

  /**
   * The move for a game state. mask: the 43 moves allowed (the game supplies it; default actionMask(state); false allows every move);
   * regions past the 43 (TARGET) are always allowed. temp: the sampling temperature (default the agent's).
   */
  async act(state: CraftaxState, mask?: ArrayLike<boolean | number> | false, temp = this.temperature): Promise<CraftaxMove> {
    const m = mask === false ? new Array<boolean>(CRAFTAX_N_ACTIONS).fill(true) : (mask ?? actionMask(state));
    return this.actOn(frame(state, this.frameOptions), m, temp, state.sleeping || state.resting);
  }

  /** The move for a board picture already drawn (frame()). hold: asleep or resting, so the move is noop and the memory is kept. */
  async actOn(board: ArrayLike<number>, mask: ArrayLike<boolean | number>, temp = this.temperature, hold = false): Promise<CraftaxMove> {
    const memoryIn = this.memory.slice();
    const r = await craftaxStep(this.engine, this.layout, pictures(this.layout, board, memoryIn));
    const allowed = Array.from({ length: r.scores.length }, (_, i) => (i < CRAFTAX_N_ACTIONS ? !!mask[i] : true));
    const c = chooseMove(r.scores, allowed, temp, this.random);
    if (hold) return { ...r, action: 0, probs: c.probs, memoryIn };
    for (let u = 0; u < this.layout.units; u++) {
      const h = memoryIn[u]!, hn = Math.min(1, Math.max(0, h + r.z[u]! * (r.hc[u]! - h)));
      this.memory[u] = Math.round(hn * 15) / 15;
    }
    return { ...r, action: c.best, probs: c.probs, memoryIn };
  }
}
