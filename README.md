# neuralcrystal

Run a trained **NeuralCrystal** in TypeScript: in the browser on **WebGPU**, or in Node.

A NeuralCrystal is a passive optical computer: a block of glass with stacked phase surfaces that answers a question with light. This package simulates that light exactly as the crystals were trained, so a web page can play chess against a crystal, read handwriting, or test against RL benchmarks like Craftax.

[NeuralCrystal](https://neuralcrystal.com) was built by Pete DeLaurentis with [Shellcaster](https://shellcaster.com), an IDE for building complex projects with agents.

## Languages

The crystals run in two packages, with the same crystal files, the same physics and the same answers:

- **Python:** [`neuralcrystal`](https://github.com/neuralcrystal/neuralcrystal) (`pip install neuralcrystal`), on PyTorch. Use it for research and experiments, and for running many inputs at once on a GPU.
- **TypeScript:** this package (`npm install neuralcrystal`), on WebGPU or the CPU. Use it to put a crystal in a web page or a Node app, and to show its light in the browser.

## Install

```
npm install neuralcrystal
```

No runtime dependencies. ES modules, Node 20.9 or later. In a browser it runs on WebGPU where there is one, and on the CPU everywhere else. In Node it runs on the CPU, or on the GPU through the [`webgpu`](https://www.npmjs.com/package/webgpu) package (Dawn, Chrome's WebGPU).

```ts
import { CrystalEngine, WebGpuBackend, requestGpu } from "neuralcrystal";

// null where WebGPU isn't available: then the engine runs on the CPU
const gpu = await requestGpu();
const engine = new CrystalEngine(crystal, gpu ? (g) => new WebGpuBackend(gpu.device, g) : undefined);
```

## The crystals

The crystals are on Hugging Face under [neuralcrystal](https://huggingface.co/neuralcrystal). Each comes as three files with the same contents at different precision; the browser loads the 16-bit one.

| File | What it holds |
|---|---|
| `<problem>-crystal.safetensors` | the phase surfaces in fp32 |
| `<problem>-crystal-16bit.safetensors` | the phase wrapped to one turn, 16 bits a sample, cropped to the lit window |
| `<problem>-crystal-8bit.safetensors` | the same at 8 bits |

The Craftax crystal comes as `craftax-crystal.safetensors` (fp32), plus the 16-bit `craftax-crystal-16bit.nc16.gz` the browser game loads.

```ts
import { huggingFaceUrl, loadCrystal } from "neuralcrystal";

// a URL, a Response, a Blob or the bytes; gzip is unpacked on the way in.
// huggingFaceUrl("cifar10") is the 16-bit file on Hugging Face; huggingFaceUrl("cifar10", 32) the fp32 one, huggingFaceUrl("cifar10", 8) the 8-bit
const crystal = await loadCrystal(huggingFaceUrl("cifar10"));
```

## Examples

Every example below is a file in [`examples/`](examples) that runs from a clone of this repo: `npm install`, then `npx vite-node examples/<name>.ts`. Each takes a crystal path or URL (the 16-bit file on Hugging Face by default), prints what the crystal said, and writes its pictures to `examples/out/`. With `npm install`, the `webgpu` package comes along and the examples run on your GPU.

| Example | What it does |
|---|---|
| `mnist.ts` | a digit in, the digit out |
| `cifar10.ts` | a photo in, a class out, with every picture the light makes |
| `checkers.ts` | a position in, a move out |
| `chess.ts` | a position in, a move out, win/draw/loss, and `--think` to look two moves ahead |
| `chess-input-output.ts` | the chess crystal's board picture and move map, drawn large |
| `input-output.ts` | the same for MNIST, CIFAR-10 and checkers |
| `craftax.ts` | a Craftax game state in, a move out |
| `inside.ts` | the light at every surface of every pass |

## MNIST: a digit in, the digit out

```ts
import { huggingFaceUrl, mnistKit, solve } from "neuralcrystal";

const crystal = await loadCrystal(huggingFaceUrl("mnist"));
const kit = mnistKit(crystal);

// uint8 28 × 28, white ink on black (readPng: the examples' small PNG reader; any decoder works)
const digit = readPng("examples/assets/digit-4.png").px;

const { answer } = await solve(kit, engine, digit);

// 4, and each digit's share of the light
console.log(answer.pick, answer.shares);
```

The digit is drawn on a 32 × 32 DMD frame and makes one pass through the crystal. Ten squares on the exit face collect the light, and the brightest square is the answer. To read handwriting from a page, `StrokeInput` turns pen strokes into a digit sized and centered the way MNIST's were.

`npx vite-node examples/mnist.ts` prints the light on each square and draws the DMD frame going in and the light on the exit face, with the ten squares outlined and the answer in white:

![MNIST: the digit 4 going in, the light on the exit face](https://raw.githubusercontent.com/neuralcrystal/neuralcrystal-typescript/main/docs/img/mnist-digit-4.png)

## CIFAR-10: a photo in, a class out

```ts
import { CIFAR_NAMES, huggingFaceUrl, cifarKit, flipPicture, solve } from "neuralcrystal";

const crystal = await loadCrystal(huggingFaceUrl("cifar10"));
const kit = cifarKit(crystal);

// uint8 32 × 32 × 3
const photo = readPng("examples/assets/frog.png").px;

// one run
const { answer } = await solve(kit, engine, photo);

// + the mirrored photo: the "with flip" accuracy
const flipped = (await solve(kit, engine, flipPicture(photo))).answer;
const scores = answer.scores.map((s, k) => s + flipped.scores[k]!);

// frog
console.log(answer.name, CIFAR_NAMES[scores.indexOf(Math.max(...scores))]);
```

The photo is contrast-normalized and drawn on a 256 × 256 DMD frame. Every photo pixel becomes a 2 × 2 of color regions (red, green, blue, and 255 − luma), each one area-dithered. The light makes four passes. Between passes, the camera's picture is exposed from its own mean and spread, then dithered back onto the DMD. Twenty squares in ten differential pairs read the answer: class *k* scores (p⁺ − p⁻) / (p⁺ + p⁻), and the largest wins.

`npx vite-node examples/cifar10.ts` draws every picture the light makes: the DMD frame, the camera's picture after passes 1, 2 and 3 (what the DMD shows the next pass, in `run.hiddens`), and the exit face with the twenty squares, the answer's plus square in white:

![CIFAR-10: the frog's DMD frame, the three hidden pictures, and the exit face](https://raw.githubusercontent.com/neuralcrystal/neuralcrystal-typescript/main/docs/img/cifar10-frog.png)

## Checkers: a position in, a move out

```ts
import { huggingFaceUrl, checkers as ck, checkersKit } from "neuralcrystal";

const crystal = await loadCrystal(huggingFaceUrl("checkers"));
const kit = checkersKit(crystal);

// 32 squares in draughts order, side to move, jumping piece, idle moves
const pos = ck.parse("bb.bb...b.wb.b.b....w.ww.w.ww.ww w - 0");

// the brightest legal move, and every cell's light (answer.powers)
const run = await engine.run(kit.frame(pos), kit.frameSize);
const answer = kit.readWith(engine, run, pos, { temp: 0 });

// e.g. 22-18 (a step) or 22x15 (a jump)
console.log(answer.uci);

// if next.jump >= 0 the same side must jump again: ask again
const next = ck.make(pos, answer.move!);
```

The position is drawn as a 96 × 96 picture with the crystal as the side to move: its pieces are solid discs, its kings are discs with a crown, the opponent's pieces are rings, and empty dark squares have a corner bracket. The light makes four passes. The exit face holds a 32 × 32 move map, where the row is the from-square and the column is the to-square. The kit reads the light in every legal move's cell, and the brightest wins. `ck.moves`, `ck.make` and `ck.status` implement the full rules of American checkers: captures are compulsory, jump chains are played one jump at a time, and a man crowned on the far row ends its turn. Pass `temp` and `topK` to `readWith` to sample among the brightest moves instead, or use `kit.read`, which plays by the crystal file's own rules.

`npx vite-node examples/checkers.ts` prints the move and draws the DMD frame, the camera's picture after passes 1, 2 and 3, and the move map with the crystal's move in white:

![Checkers: the board frame, the three hidden pictures, and the move map](https://raw.githubusercontent.com/neuralcrystal/neuralcrystal-typescript/main/docs/img/checkers-midgame.png)

## Chess: a position in, a move out

```ts
import { huggingFaceUrl, chess as ch, chessKit, chessThink, chessValueOf, solve } from "neuralcrystal";

const crystal = await loadCrystal(huggingFaceUrl("chess"));
const kit = chessKit(crystal);

// any FEN, either side to move
const pos = ch.parseFEN("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");

// the brightest legal move
const { answer } = await solve(kit, engine, pos);

// e6, and (win, draw, loss) for the side to move
console.log(answer.san, chessValueOf(answer.powers, kit.valueCells!));

// think two moves ahead: its 3 brightest moves against the opponent's 5 brightest replies to each
const thought = await chessThink(pos, answer.movePowers, {
  mode: "always", share: 1, widths: [3, 5], side: kit.art.playAs, keys: [], valueCells: kit.valueCells!, moveP: (p) => kit.moveP(p),
  look: async (s) => kit.powers(engine, (await engine.run(kit.frame(s), kit.frameSize)).exit),
});
const move = thought!.cand[thought!.best!]!.move;
```

The crystal's input is a picture of the board, which you can make yourself, look at, and run by hand:

```ts
// the 192 × 192 board picture the crystal sees (0 = dark mirror, 1 = lit)
const board = kit.frame(pos);

// the simulation: the light through all four passes of the crystal
const run = await engine.run(board, kit.frameSize);

// 64 × 64 light: row = from-square, column = to-square
const moveMap = kit.powers(engine, run.exit);

// the brightest legal move (the position says which moves are legal), and (win, draw, loss)
const reading = kit.read(engine, run, pos);
console.log(reading.san, chessValueOf(moveMap, kit.valueCells!));
```

The picture has to be in the crystal's own drawing style. It won't read a photo or a screenshot of an ordinary chessboard.

The crystal always sees the board from the side to move. White's positions are mirrored before they go in (`chessMirror`: the ranks flipped, the colors swapped) and the move is turned back afterwards, so you can ask about either side; `answer.mirrored` says when that happened. The move map stays the crystal's own, so `kit.cells(pos)` gives each legal move's cell on it, with the move on the real board:

```ts
// the five brightest legal moves
const top = [...kit.cells(pos)].map(([cell, move]) => ({ move, light: answer.movePowers[cell]! })).sort((a, b) => b.light - a.light).slice(0, 5);
```

The position is drawn as a 192 × 192 picture of the board. The light makes four passes and lands on a 64 × 64 move map (row = from-square, column = to-square) plus three stretches of the diagonal that read win, draw and loss. The brightest legal cell is the move. `ch.moves`, `ch.make`, `ch.status` and `ch.san` implement the full rules: castling, en passant, promotion, check, mate, stalemate and the draw rules. `ch.fen`, `ch.uci` and `ch.fromUci` read and write positions and moves.

With `widths: [3, 5]` the crystal looks ahead. It takes its three brightest moves, reads the board after each, takes the opponent's five brightest replies to each, and reads those boards' win/draw/loss. It then plays the move with the best expected score after those replies. That's 19 runs of the crystal instead of one, and it's how the published chess numbers were measured. With `mode: "unsure"` and `share: 0.5` it looks ahead only when the brightest move holds under half the light of its top three, as the website does.

`npx vite-node examples/chess.ts` prints the move, the five brightest moves and the win/draw/loss reading. It also draws the DMD frame, the camera's picture after passes 1, 2 and 3, and the move map, with the legal moves outlined in teal and the crystal's move in white. After 1. e4 the crystal answers e6, the French Defense:

![Chess: the board after 1. e4, the three hidden pictures, and the move map](https://raw.githubusercontent.com/neuralcrystal/neuralcrystal-typescript/main/docs/img/chess-e4.png)

Looking ahead matters. In this endgame (`7k/1p5p/7P/2P5/p1B5/P3q3/K2R2P1/7R b - - 0 40`) the brightest move is Qe2, which walks the queen into the rook and bishop. Thinking two moves ahead sees that and plays Qf2 instead (`npx vite-node examples/chess.ts "" "<FEN>" out.png --think`):

```
black to move: the crystal plays Qe2 (e3e2)
  Qe2     ########################################
  Qf2     ###############################
  thought about e3e2, e3f2, e3f4, expected scores 0.03, 0.23, 0.13
  after looking ahead: Qf2
```

`npx vite-node examples/chess-input-output.ts` draws the input and the output large: the board picture, and the move map with the legal moves in blue, the crystal's move in orange and the win/draw/loss cells in grey.

![Chess: the board picture the crystal sees](https://raw.githubusercontent.com/neuralcrystal/neuralcrystal-typescript/main/docs/img/chess-board.png)

![Chess: the move map, legal moves outlined](https://raw.githubusercontent.com/neuralcrystal/neuralcrystal-typescript/main/docs/img/chess-move-map.png)

## Craftax: a crystal RNN with memory

[Craftax](https://github.com/MichaelTMatthews/Craftax) is a 2D survival RL research benchmark where the agent gathers resources, crafts tools, fights creatures and goes down through the dungeon floors. The package doesn't run the game itself: give it a Craftax game state, and it draws the two pictures the crystal sees, runs the light, and returns the move and the next memory. To play whole games in a page or in Node, [craftax-wasm](https://github.com/delaurentis/craftax-wasm) runs Craftax compiled to WebAssembly, bit-exact with the JAX original, and hands you its states in the form this package reads.

```ts
import { CRAFTAX_ACTIONS, CraftaxAgent, huggingFaceUrl } from "neuralcrystal";

const crystal = await loadCrystal(huggingFaceUrl("craftax"));

// holds the crystal's memory from step to step; agent.reset() before a new game
const agent = new CraftaxAgent(engine);

// a state in the browser engine's JSON form
const state = JSON.parse(readFileSync("examples/assets/craftax-state.json", "utf8"));

// the board picture and the memory in, the move out (0 … 42)
const move = await agent.act(state);

// e.g. "left", and the moves' chances
console.log(CRAFTAX_ACTIONS[move.action], move.probs);
```

`agent.act` keeps the crystal to the moves the game allows in that state (`craftaxActionMask(state)`: Craftax's own rules, as the crystal was trained and scored) and samples the move from their chances, as the game page plays. Pass a temperature of 0 to take the most likely move, a mask of your own, or `false` to allow every move. While the player sleeps or rests it plays no-op and keeps its memory, as in training.

The crystal sees two 192 × 192 pictures side by side each step. The first is the board: the 9 × 11 tiles around the player as symbols, with its health, food, drink and energy, and the player's own tile marked with its needs and its pickaxe. The second is memory and additional item status: its memory of 1,024 numbers, 4 bits each, as it wrote them the step before, with more of its items around them.

Here is one moment of a game, 127 steps in, as Craftax draws it: the player on grass by a lake, with a cow and another creature to its left.

![Craftax: the game 127 steps in, as Craftax draws it](https://raw.githubusercontent.com/neuralcrystal/neuralcrystal-typescript/main/docs/img/craftax-game.png)

And the two pictures the crystal sees at that moment:

![Craftax: the board picture, and the memory and additional item status picture](https://raw.githubusercontent.com/neuralcrystal/neuralcrystal-typescript/main/docs/img/craftax-inputs.png)

The light makes five passes. The two pictures go through their own input sections side by side, their camera pictures are combined pixel by pixel, two shared sections follow, and then the light splits into two heads of two sections each. The action head lights 43 regions on a 64 × 64 grid, one for each move. The memory head lights two cells for each memory number: how much to change it, and what to change it to.

![Craftax: the 43 action regions, each as bright as its chance, and the memory head's two regions](https://raw.githubusercontent.com/neuralcrystal/neuralcrystal-typescript/main/docs/img/craftax-outputs.png)

You can make the pictures yourself and run one step by hand:

```ts
import { craftaxFrame, craftaxLayout, craftaxPictures, craftaxStep } from "neuralcrystal";

// the 192 × 192 board picture (0 = dark mirror, 1 = lit)
const board = craftaxFrame(state);

// the two input pictures: the board, and memory and item status
const layout = craftaxLayout(crystal.model);
const [first, second] = craftaxPictures(layout, board, agent.memory);

// both heads' light leaving the glass, read out: the moves' scores, and each memory number's update gate z and new value hc
const { exits } = await engine.runSplit(first, second, layout.F);
const { scores, z, hc } = await craftaxStep(engine, layout, [first, second]);
```

A move's chance is the softmax of its score over the allowed moves. The next memory is h + z·(hc − h), rounded to 4 bits. `npx vite-node examples/craftax.ts` plays one step from a saved state, prints the five likeliest moves, and draws both input pictures and the action regions.

## Seeing inside: pictures at every stage

Every number above comes from light, and you can look at that light anywhere along the way. `engine.trace(frame, F)` (one-section crystals) and `engine.traceStages(frame, F)` (the multi-stage ones) return the field arriving at every surface of every pass, and the pictures handed between passes.

```ts
const t = await engine.traceStages(kit.frame(photo), kit.frameSize);

// per pass: arriving[0 … planes − 1] is the light at each surface, arriving[planes] the exit face
for (const [k, pass] of t.passes.entries()) console.log(`pass ${k + 1}: ${pass.arriving.length - 1} surfaces`);

// what the DMD showed passes 2, 3 and 4
t.hiddens;
```

`npx vite-node examples/inside.ts cifar10` (or `checkers`, `chess`, `mnist`) draws one row per pass: what the DMD launched, the light arriving at each of the section's surfaces, and the light at its exit face. Every tile is scaled to its own brightest point, with a square-root brightness so faint light shows, so compare shapes rather than brightness between tiles.

The CIFAR-10 frog: the photo goes in at top left, and the twenty readout squares light up at bottom right:

![Inside the CIFAR-10 crystal: four passes, ten surfaces each](https://raw.githubusercontent.com/neuralcrystal/neuralcrystal-typescript/main/docs/img/inside-cifar10.png)

The checkers position from the example above. In the first two passes you can see the board itself being re-imaged at several surfaces, and squeezed to a small spot at surfaces 3 and 8. Passes 3 and 4 no longer look like the board: by then the light is computing the move.

![Inside the checkers crystal: four passes and the move map](https://raw.githubusercontent.com/neuralcrystal/neuralcrystal-typescript/main/docs/img/inside-checkers.png)

And the chess crystal after 1. e4: four passes of twenty surfaces each.

![Inside the chess crystal: four passes and the move map](https://raw.githubusercontent.com/neuralcrystal/neuralcrystal-typescript/main/docs/img/inside-chess.png)

## What's in it

| Folder | What |
|---|---|
| `crystal/` | crystal files: safetensors with a JSON header, loading with progress and gzip, the compact 16-bit web format, mirroring |
| `physics/` | the grid, the transfer functions between surfaces, fixed and learned phase surfaces, a section compiled to a plan |
| `light/`, `cpu/`, `webgpu/` | the light field, the backend interface, and its two implementations: a float64 CPU reference and WebGPU |
| `engine/` | `CrystalEngine`: run a frame, trace it surface by surface, multi-stage and split crystals |
| `hardware/` | the DMD, the camera and its imager response, detector squares, the hand-off between stages, energy and battery, a 96 × 96 screen |
| `kits/`, `problems/` | a problem kit turns a game or dataset into the crystal's picture and its light back into an answer: MNIST, CIFAR-10, checkers, chess, Craftax |

Write your own `LightBackend` to run the light another way, or your own `ProblemKit` to give a crystal a new game.

## What the simulation is

Scalar, coherent light at one wavelength in fused silica (n = 1.4607). The light travels between surfaces by the band-limited angular spectrum method, and each surface multiplies the field by exp(iφ). The block's sides absorb light that reaches them. Optical cross talk is included in the simulation. This is the same forward pass the crystals were trained with, and the same as the Python package's: the test fixtures are small crystals run through the Python code with every intermediate field saved, and both backends here match them to 1e-6.

```
npm run check      # typecheck, lint, tests on the CPU, 500-line file limit
npm run test:gpu   # the WebGPU path through Dawn in Node
```

## License

The code is MIT. The crystals are CC BY-NC 4.0. © 2026 TextJam, Inc.
