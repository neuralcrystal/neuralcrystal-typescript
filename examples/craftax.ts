// A Craftax game state in, a move out. The package doesn't run the game: give it a state in the browser engine's JSON form (craftax-wasm,
// the C engine compiled to WebAssembly, plays whole games in a page or in Node), and it draws the two pictures the crystal sees, runs the
// light, and returns the move and the next memory.
//   npx vite-node examples/craftax.ts [crystal] [state.json]
import { readFileSync } from "node:fs";
import { CRAFTAX_ACTIONS, CraftaxAgent, craftaxActionMask, craftaxFrame, craftaxLayout, craftaxPictures, huggingFaceUrl } from "../src/index.js";
import type { CraftaxState } from "../src/index.js";
import { engine, frameTile, lightTile, load, savePng } from "./lib.js";

const [src = huggingFaceUrl("craftax"), file = "examples/assets/craftax-state.json"] = process.argv.slice(2);
const crystal = await load(src), eng = await engine(crystal);
const state = JSON.parse(readFileSync(file, "utf8")) as CraftaxState;

// holds the crystal's memory from step to step; agent.reset() before a new game
const agent = new CraftaxAgent(eng);
const memoryIn = agent.memory.slice();
// the board picture and the memory in, the move out (0 … 42); temperature 0 takes the most likely move, the default 1 samples as the game does
const mv = await agent.act(state, craftaxActionMask(state), 0);
const top = Array.from(mv.probs, (p, i) => ({ p, i })).sort((a, b) => b.p - a.p).slice(0, 5);
console.log(`the crystal plays ${CRAFTAX_ACTIONS[mv.action] ?? mv.action}`);
for (const t of top) console.log(`  ${String(CRAFTAX_ACTIONS[t.i] ?? t.i).padEnd(22)} ${"#".repeat(Math.round(40 * t.p))} ${(100 * t.p).toFixed(1)} %`);

// the two pictures it saw: the board, and memory and additional item status (as it was before this step)
const L = craftaxLayout(crystal.model), board = craftaxFrame(state), [first, second] = craftaxPictures(L, board, memoryIn);
// the action head's 43 regions on the 64 × 64 grid, each as bright as its chance
const grid = new Float64Array(64 * 64);
L.cells.act.forEach((c, i) => { for (const k of typeof c === "number" ? [c] : c) grid[k] = mv.probs[i] ?? 0; });
savePng("examples/out/craftax.png", [[frameTile(first!, L.F, 2), ...(second ? [frameTile(second, L.F, 2)] : []), lightTile(grid, 64, 6)]]);
console.log("wrote examples/out/craftax.png");
eng.dispose();
