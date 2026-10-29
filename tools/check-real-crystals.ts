// Runs the published crystals (the 16-bit files from Hugging Face) on a few inputs each — CIFAR-10, checkers, chess, Craftax — on
// WebGPU through Dawn, writes what went in and what came out as PNGs with an index.html, and, when ORIGINAL_ENGINE names the original
// engine's build (a path to its dist/index.js), runs every input through it too and checks the answers are the same.
// It runs on the BUILT package (dist/), as a user gets it, compiled to plain JS so Node can also load the original engine's build:
//   npm run build
//   npx esbuild test/f16-polyfill.ts --format=esm --outfile=tools/.f16.mjs
//   npx esbuild tools/check-real-crystals.ts --format=esm --outfile=tools/.check.mjs
//   sed -i '' -e 's#"../src/index.js"#"../dist/index.js"#' -e 's#"../test/f16-polyfill.js"#"./.f16.mjs"#' tools/.check.mjs
//   CRYSTALS=<dir with the four files> OUT=<dir> CIFAR=<a CIFAR test shard .json> [ORIGINAL_ENGINE=<dist/index.js>] node tools/.check.mjs
import "../test/f16-polyfill.js";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { deflateSync } from "node:zlib";
import { create, globals } from "webgpu";
import * as nc from "../src/index.js";

Object.assign(globalThis, globals);
const instance = create([]) as unknown as GPU; // held for the whole run: Dawn crashes if its instance is collected
const gpu = await nc.requestGpu(instance);
if (!gpu) throw new Error("no WebGPU adapter");

const CRYSTALS = process.env["CRYSTALS"]!, OUT = process.env["OUT"]!, CIFAR = process.env["CIFAR"];
mkdirSync(join(OUT, "png"), { recursive: true });
// the original engine's build, when ORIGINAL_ENGINE names it, loaded by Node itself rather than through Vite
const OLD_PATH = process.env["ORIGINAL_ENGINE"] ? pathToFileURL(resolve(process.env["ORIGINAL_ENGINE"])) : null;
const nativeImport = (u: string): Promise<unknown> => import(/* @vite-ignore */ u);
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const old: any = OLD_PATH && existsSync(OLD_PATH) ? await nativeImport(OLD_PATH.href) : null;

// ---- PNG
const crcTable = Array.from({ length: 256 }, (_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c >>> 0; });
const crc = (b: Uint8Array): number => { let c = 0xffffffff; for (const x of b) c = crcTable[(c ^ x) & 255]! ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; };
function png(file: string, w: number, h: number, rgb: Uint8Array, scale = 1): string {
  const W = w * scale, H = h * scale, raw = new Uint8Array(H * (W * 3 + 1));
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const s = (Math.floor(y / scale) * w + Math.floor(x / scale)) * 3, d = y * (W * 3 + 1) + 1 + x * 3;
    raw[d] = rgb[s]!; raw[d + 1] = rgb[s + 1]!; raw[d + 2] = rgb[s + 2]!;
  }
  const chunk = (type: string, data: Uint8Array): Buffer => {
    const t = Buffer.concat([Buffer.from(type), Buffer.from(data)]), len = Buffer.alloc(4), c = Buffer.alloc(4);
    len.writeUInt32BE(data.length); c.writeUInt32BE(crc(t));
    return Buffer.concat([len, t, c]);
  };
  const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(W, 0); ihdr.writeUInt32BE(H, 4); ihdr[8] = 8; ihdr[9] = 2;
  writeFileSync(join(OUT, "png", file), Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", ihdr), chunk("IDAT", deflateSync(raw)), chunk("IEND", new Uint8Array())]));
  return `png/${file}`;
}
/** grey 0..1 → RGB, optionally in the laser's red */
const grey = (v: ArrayLike<number>, red = false): Uint8Array => {
  const o = new Uint8Array(v.length * 3);
  for (let i = 0; i < v.length; i++) { const g = Math.round(255 * Math.min(1, Math.max(0, v[i]!))); o[3 * i] = g; o[3 * i + 1] = red ? Math.round(g * 0.4) : g; o[3 * i + 2] = red ? Math.round(g * 0.28) : g; }
  return o;
};
/** light → a picture: square-root brightness, scaled to its brightest point, so faint light shows */
const light = (v: ArrayLike<number>): Uint8Array => { let m = 0; for (let i = 0; i < v.length; i++) m = Math.max(m, v[i]!); return grey(Array.from(v, (x) => Math.sqrt(x / (m || 1))), true); };
const panelPng = (file: string, p: nc.Panel): string => { const a = p.rgba(), rgb = new Uint8Array(p.w * p.h * 3); for (let i = 0; i < p.w * p.h; i++) { rgb[3 * i] = a[4 * i]!; rgb[3 * i + 1] = a[4 * i + 1]!; rgb[3 * i + 2] = a[4 * i + 2]!; } return png(file, p.w, p.h, rgb, 3); };

// ---- engines
const bytes = (f: string): Uint8Array => new Uint8Array(readFileSync(join(CRYSTALS, f)));
// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function engines(file: string): Promise<{ data: nc.CrystalData; eng: nc.CrystalEngine; od: any; oe: any }> {
  const data = await nc.loadCrystal(bytes(file));
  const eng = new nc.CrystalEngine(data, (g) => new nc.WebGpuBackend(gpu!.device, g));
  if (!old) return { data, eng, od: null, oe: null };
  const od = await old.loadCrystal(bytes(file));
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const oe = new old.CrystalEngine(od, (g: any) => new old.WebGpuBackend(gpu!.device, g));
  return { data, eng, od, oe };
}
const same = (a: ArrayLike<number>, b: ArrayLike<number>): boolean => a.length === b.length && Array.from(a).every((x, i) => x === b[i] || (Number.isNaN(x) && Number.isNaN(b[i]!)));

interface Row { title: string; images: { label: string; src: string }[]; lines: string[]; match: boolean | null }
const sections: { name: string; about: string; rows: Row[]; ms: number[] }[] = [];

// ---- CIFAR-10
if (CIFAR && existsSync(CIFAR)) {
  const { data, eng, od, oe } = await engines("cifar10-crystal-16bit.safetensors");
  const kit = nc.cifarKit(data), okit = od ? old.cifarKit(od) : null, shard = nc.parseCifarShard(readFileSync(CIFAR, "utf8"));
  const sec = { name: "CIFAR-10", about: `${data.model.N}² grid, ${eng.quad.hops} stages. A photo from the CIFAR-10 test set in, one of ten classes out.`, rows: [] as Row[], ms: [] as number[] };
  for (const i of [0, 1, 2, 3, 4, 5, 6, 7]) {
    const pic = nc.cifarPicture(shard, i), truth = nc.CIFAR_NAMES[shard.labels[i]!]!;
    const t0 = performance.now(), { answer, run } = await nc.solve(kit, eng, pic);
    sec.ms.push(performance.now() - t0);
    const camera = eng.readout(run.exit, 128);
    const match = okit ? same((await old.solve(okit, oe, pic)).answer.scores, answer.scores) : null;
    const panel = new nc.Panel();
    kit.paint(panel, { picture: pic, answer: answer.name, correct: answer.name === truth, truth });
    sec.rows.push({
      title: `test picture ${i}: a ${truth}`, match,
      images: [
        { label: "the photo", src: png(`cifar-${i}-photo.png`, 32, 32, pic, 4) },
        { label: "the DMD picture (what the mirrors show)", src: png(`cifar-${i}-dmd.png`, kit.frameSize, kit.frameSize, grey(kit.frame(pic))) },
        { label: "the camera after the last stage", src: png(`cifar-${i}-camera.png`, 128, 128, light(camera), 2) },
        { label: "the player's screen", src: panelPng(`cifar-${i}-screen.png`, panel) },
      ],
      lines: [`crystal: <b>${answer.name}</b> ${answer.name === truth ? "✓" : "✗ (truth: " + truth + ")"}`, `share of the light: ${(100 * answer.shares[answer.pick]!).toFixed(1)} %`],
    });
  }
  eng.dispose(); oe?.dispose();
  sections.push(sec);
}

// ---- checkers
{
  const { data, eng, od, oe } = await engines("checkers-crystal-16bit.safetensors");
  const kit = nc.checkersKit(data), okit = od ? old.checkersKit(od) : null, C = nc.checkers;
  const sec = { name: "Checkers", about: `${data.model.N}² grid, ${eng.quad.hops} stages. A position in, a move out (the brightest legal cell of the 32 × 32 move map).`, rows: [] as Row[], ms: [] as number[] };
  let st = C.start();
  const positions: { label: string; st: nc.CheckersState }[] = [];
  for (let ply = 0; positions.length < 5 && ply < 40; ply++) {
    if (ply % 4 === 0) positions.push({ label: `after ${ply} plies, ${st.turn > 0 ? "black" : "red"} to move`, st });
    const ms = C.moves(st);
    if (!ms.length) break;
    st = C.make(st, ms[(ply * 7) % ms.length]!);
  }
  for (const [k, p] of positions.entries()) {
    const t0 = performance.now(), run = await eng.run(kit.frame(p.st), kit.frameSize), a = kit.readWith(eng, run, p.st, { temp: 0 });
    sec.ms.push(performance.now() - t0);
    let match: boolean | null = null;
    if (okit) { const orun = await oe.run(okit.frame(p.st), okit.frameSize), oa = okit.readWith(oe, orun, p.st, { temp: 0 }); match = oa.uci === a.uci && same(oa.powers, a.powers); }
    const before = new nc.Panel(), after = new nc.Panel();
    kit.paint(before, { state: p.st, seat: 0 });
    if (a.move) kit.paint(after, { state: C.make(p.st, a.move), seat: 0, last: [a.move.from, a.move.to] });
    sec.rows.push({
      title: `position ${k + 1}: ${p.label}`, match,
      images: [
        { label: "the position", src: panelPng(`checkers-${k}-before.png`, before) },
        { label: "the DMD picture", src: png(`checkers-${k}-dmd.png`, kit.frameSize, kit.frameSize, grey(kit.frame(p.st)), 2) },
        { label: "the move map (from × to)", src: png(`checkers-${k}-map.png`, 32, 32, light(a.powers), 6) },
        { label: "after the crystal's move", src: panelPng(`checkers-${k}-after.png`, after) },
      ],
      lines: [`crystal plays <b>${a.uci ?? "nothing"}</b>`, `${a.cells} legal moves; this one holds ${(100 * a.share).toFixed(1)} % of the light, ${(100 * a.legal).toFixed(0)} % fell on legal cells`],
    });
  }
  eng.dispose(); oe?.dispose();
  sections.push(sec);
}

// ---- chess
{
  const { data, eng, od, oe } = await engines("chess-crystal-16bit.safetensors");
  const kit = nc.chessKit(data), okit = od ? old.chessKit(od) : null, C = nc.chess;
  const sec = { name: "Chess", about: `${data.model.N}² grid, ${eng.quad.hops} stages, plays ${kit.art.playAs === "b" ? "black" : kit.art.playAs === "w" ? "white" : "either side"}. A position in, a move out (the brightest legal cell of the 64 × 64 move map).`, rows: [] as Row[], ms: [] as number[] };
  const lines: [string, string[]][] = [["the opening", []], ["1. e4", ["e2e4"]], ["1. e4 e5 2. Nf3", ["e2e4", "e7e5", "g1f3"]], ["1. d4 d5 2. c4", ["d2d4", "d7d5", "c2c4"]], ["1. e4 c5 2. Nf3 d6 3. d4", ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4"]], ["Italian, 3. Bc4", ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"]]];
  const positions = lines.map(([label, ucis]) => { let s = C.start(); for (const u of ucis) s = C.make(s, C.fromUci(s, u)!); return { label, st: s }; })
    .filter((p) => !kit.art.playAs || p.st.turn === kit.art.playAs).slice(0, 4);
  for (const [k, p] of positions.entries()) {
    const t0 = performance.now(), { answer: a } = await nc.solve(kit, eng, p.st);
    sec.ms.push(performance.now() - t0);
    const match = okit ? (await old.solve(okit, oe, p.st)).answer.uci === a.uci : null;
    const before = new nc.Panel(), after = new nc.Panel();
    kit.paint(before, { state: p.st });
    if (a.state) kit.paint(after, { state: a.state, last: [a.from, a.to] });
    sec.rows.push({
      title: `${p.label}, ${p.st.turn === "w" ? "white" : "black"} to move`, match,
      images: [
        { label: "the position", src: panelPng(`chess-${k}-before.png`, before) },
        { label: "the DMD picture", src: png(`chess-${k}-dmd.png`, kit.frameSize, kit.frameSize, grey(kit.frame(p.st)), 2) },
        { label: "the move map (from × to)", src: png(`chess-${k}-map.png`, 64, 64, light(a.movePowers), 4) },
        { label: "after the crystal's move", src: panelPng(`chess-${k}-after.png`, after) },
      ],
      lines: [`crystal plays <b>${a.san ?? "nothing"}</b> (${a.uci})`],
    });
  }
  eng.dispose(); oe?.dispose();
  sections.push(sec);
}

// ---- Craftax
{
  const { data, eng, oe } = await engines("craftax-crystal-16bit.nc16.gz");
  let match = true, compared = 0;
  // the agent's engine, with every step also run through the original engine and compared byte for byte
  const probe: nc.CraftaxEngine = {
    data: eng.data, geometry: eng.geometry, run: eng.run.bind(eng),
    runSplit: async (fa, fb, F) => {
      const r = await eng.runSplit(fa, fb, F);
      if (oe) { const o = await oe.runSplit(fa, fb, F); compared++; r.exits.forEach((x, i) => { if (!same(x.re, o.exits[i].re) || !same(x.im, o.exits[i].im)) match = false; }); }
      return r;
    },
  };
  const agent = new nc.CraftaxAgent(probe, { temperature: 0 });
  const L = nc.craftaxLayout(data.model);
  const cases = (JSON.parse(readFileSync(new URL("../test/fixtures/craftax-frames.json", import.meta.url), "utf8")) as { cases: { label: string; state: nc.CraftaxState }[] }).cases;
  const sec = { name: "Craftax", about: `${data.model.N}² grid, a split crystal (${L.split ? "two input heads" : "one head"}), ${L.units} memory units. A game state in, a move and the next memory out; the memory carries from step to step.`, rows: [] as Row[], ms: [] as number[] };
  for (const [k, c] of cases.filter((c) => !/edited|edge/i.test(c.label)).slice(0, 5).entries()) {
    const before = compared, was = match;
    match = true;
    const t0 = performance.now(), mv = await agent.act(c.state, undefined, 0);
    sec.ms.push(performance.now() - t0);
    const board = nc.craftaxFrame(c.state), F = Math.round(Math.sqrt(board.length));
    const top = Array.from(mv.probs, (p, i) => ({ p, i })).sort((u, v) => v.p - u.p).slice(0, 3);
    const name = (i: number): string => nc.CRAFTAX_ACTIONS[i] ?? (i === 43 ? "TARGET" : `#${i}`);
    sec.rows.push({
      title: `state ${k + 1}: ${c.label}`, match: oe ? match && compared > before : null,
      images: [{ label: "the board picture the crystal sees", src: png(`craftax-${k}-board.png`, F, F, grey(board), 2) }],
      lines: [`crystal plays <b>${name(mv.action)}</b>`, `top choices: ${top.map((t) => `${name(t.i)} ${(100 * t.p).toFixed(0)} %`).join(", ")}`, `memory bits set after the step: ${Array.from(agent.memory).filter((v) => v > 0.5).length} of ${agent.memory.length}`],
    });
    match = was && match;
  }
  eng.dispose(); oe?.dispose();
  sections.push(sec);
}

// ---- index.html
const esc = (s: string): string => s.replace(/&(?!amp;)/g, "&amp;");
const total = sections.flatMap((s) => s.rows), bad = total.filter((r) => r.match === false).length, cmp = total.filter((r) => r.match !== null).length;
const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Crystal Checks</title>
<style>:root{--bg:#fbfaf8;--fg:#1d1a18;--mut:#6b625b;--line:#e4ded8;--card:#f3efea;--ok:#2f7d4a;--bad:#b3392f}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#151312;--fg:#ece7e1;--mut:#a39b92;--line:#2c2826;--card:#1d1a18;--ok:#6cc08a;--bad:#e2786d}}
:root[data-theme="dark"]{--bg:#151312;--fg:#ece7e1;--mut:#a39b92;--line:#2c2826;--card:#1d1a18;--ok:#6cc08a;--bad:#e2786d}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}
h2{border-top:1px solid var(--line);padding-top:16px;margin-top:32px}.mut{color:var(--mut)}.row{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin:12px 0}
.imgs{display:flex;flex-wrap:wrap;gap:12px;margin:8px 0}.imgs figure{margin:0}.imgs img{display:block;image-rendering:pixelated;max-width:100%;height:auto;background:#000}.imgs figcaption{font-size:12px;color:var(--mut);max-width:300px}
.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}</style></head><body><main>
<h1>neuralcrystal-typescript: real crystals</h1>
<p class="mut">The published 16-bit crystals from Hugging Face, run by the new package on WebGPU (Dawn, in Node). ${old ? `Every input also went through the original engine: <span class="${bad ? "bad" : "ok"}">${cmp - bad} of ${cmp} answers identical</span>.` : "The original engine was not found, so nothing was compared."} ${new Date().toISOString().slice(0, 16).replace("T", " ")} UTC.</p>
${sections.map((s) => `<h2>${s.name}</h2><p class="mut">${s.about} Median time a read: ${[...s.ms].sort((a, b) => a - b)[Math.floor(s.ms.length / 2)]!.toFixed(0)} ms (the first read also compiles the GPU plans).</p>
${s.rows.map((r) => `<div class="row"><b>${esc(r.title)}</b> ${r.match === null ? "" : r.match ? '<span class="ok">same as the original engine</span>' : '<span class="bad">differs from the original engine</span>'}
<div class="imgs">${r.images.map((i) => `<figure><img src="${i.src}" alt="${i.label}"><figcaption>${i.label}</figcaption></figure>`).join("")}</div>${r.lines.map((l) => `<div>${l}</div>`).join("")}</div>`).join("\n")}`).join("\n")}
</main></body></html>`;
writeFileSync(join(OUT, "index.html"), html);
console.log(`${total.length} checks, ${cmp} compared with the original engine, ${bad} different → ${join(OUT, "index.html")}`);
for (const s of sections) console.log(`${s.name}: ${s.rows.map((r) => (r.match === false ? "DIFF " : "") + r.lines[0]!.replace(/<[^>]+>/g, "")).join(" | ")}`);
process.exit(0);
