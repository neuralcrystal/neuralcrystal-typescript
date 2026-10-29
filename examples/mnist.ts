// A digit in, the digit out: one pass through the MNIST crystal; ten squares on the exit face collect the light, the brightest wins.
//   npx vite-node examples/mnist.ts [crystal] [digit.png] [out.png]
import { mnistKit, solve, huggingFaceUrl } from "../src/index.js";
import { engine, exitTile, frameTile, load, outline, readPng, savePng } from "./lib.js";

const [src = huggingFaceUrl("mnist"), png = "examples/assets/digit-4.png", out = "examples/out/mnist.png"] = process.argv.slice(2);
const crystal = await load(src), eng = await engine(crystal), kit = mnistKit(crystal);

const digit = readPng(png).px; // uint8 28 × 28, white ink on black
const { answer, run } = await solve(kit, eng, digit);
console.log(`the crystal reads ${answer.pick}`);
answer.shares.forEach((s, d) => console.log(`  ${d}  ${"#".repeat(Math.round(60 * s))} ${(100 * s).toFixed(1)} %`));

// the DMD frame going in, and the light on the exit face with the ten squares (the answer's in white)
const size = 256, exit = exitTile(run.exit, size), k = size / eng.geometry.N;
kit.rects(eng).forEach((r, d) => outline(exit, Math.round(r.x0 * k), Math.round(r.y0 * k), Math.round(r.w * k), Math.round(r.h * k), d === answer.pick ? [255, 255, 255] : [60, 170, 160]));
savePng(out, [[frameTile(kit.frame(digit), kit.frameSize, 8), exit]]);
console.log(`wrote ${out}`);
eng.dispose();
