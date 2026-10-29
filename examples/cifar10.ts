// A photo in, a class out: four passes through the CIFAR-10 crystal, twenty squares in ten differential pairs read the answer.
//   npx vite-node examples/cifar10.ts [crystal] [photo.png] [out.png]
import { CIFAR_NAMES, cifarKit, flipPicture, solve, huggingFaceUrl } from "../src/index.js";
import { engine, exitTile, frameTile, load, outline, photoTile, readPng, savePng } from "./lib.js";

const [src = huggingFaceUrl("cifar10"), png = "examples/assets/frog.png", out = "examples/out/cifar10.png"] = process.argv.slice(2);
const crystal = await load(src), eng = await engine(crystal), kit = cifarKit(crystal);

const photo = readPng(png).px; // uint8 32 × 32 × 3
const { answer, run } = await solve(kit, eng, photo); // one run
const flipped = (await solve(kit, eng, flipPicture(photo))).answer; // + the mirrored photo: the "with flip" accuracy
const both = answer.scores.map((s, k) => s + flipped.scores[k]!);
const withFlip = both.indexOf(Math.max(...both));
console.log(`the crystal reads ${answer.name}; with the mirrored photo too: ${CIFAR_NAMES[withFlip]}`);

// every picture the light makes: the DMD frame, the camera's picture after passes 1–3, the exit face with the twenty squares
const size = 256, exit = exitTile(run.exit, size), k = size / eng.geometry.N;
kit.rects(eng).forEach((r, i) => outline(exit, Math.round(r.x0 * k), Math.round(r.y0 * k), Math.round(r.w * k), Math.round(r.h * k), i === answer.pick ? [255, 255, 255] : [60, 170, 160]));
const F = kit.frameSize;
savePng(out, [[photoTile(photo, 32, 8), frameTile(kit.frame(photo), F), ...run.hiddens.map((h) => frameTile(h, F)), exit]]);
console.log(`wrote ${out}`);
eng.dispose();
