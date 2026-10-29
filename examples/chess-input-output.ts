// The chess crystal's input and output, drawn large: board.png is the 192 × 192 picture the micromirror array shows the crystal;
// move-map.png is the 64 × 64 move map the light lands on (row = from-square, column = to-square, as the crystal sees the board: white's positions mirrored), legal
// moves outlined in blue and the crystal's move in orange. A piece can't move to its own square, so the diagonal is free: the crystal
// uses three stretches of it to read win, draw and loss for the side to move (outlined in grey).
//   npx vite-node examples/chess-input-output.ts [crystal] ["<FEN>"] [board.png] [move-map.png]
import { chess as ch, chessKit, solve, huggingFaceUrl } from "../src/index.js";
import { engine, frameTile, lightTile, load, outline, savePng } from "./lib.js";

const [src = huggingFaceUrl("chess"), fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", boardOut = "examples/out/board.png", mapOut = "examples/out/move-map.png"] = process.argv.slice(2);
const crystal = await load(src), eng = await engine(crystal), kit = chessKit(crystal);
const pos = ch.parseFEN(fen), { answer: a } = await solve(kit, eng, pos);

savePng(boardOut, [[frameTile(kit.frame(pos), kit.frameSize, 4)]]);
const S = 10, map = lightTile(a.powers, 64, S);
for (const cs of kit.valueCells ?? []) for (const c of cs) outline(map, (c % 64) * S, Math.floor(c / 64) * S, S, S, [120, 120, 120]);
for (const cell of kit.cells(pos).keys()) outline(map, (cell % 64) * S, Math.floor(cell / 64) * S, S, S, [70, 120, 230]);
outline(map, (a.cell % 64) * S, Math.floor(a.cell / 64) * S, S, S, [255, 150, 40]);
savePng(mapOut, [[map]]);
console.log(`the crystal plays ${a.san}; wrote ${boardOut} and ${mapOut}`);
eng.dispose();
