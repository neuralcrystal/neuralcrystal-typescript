import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  CrystalEngine, DETECTOR_LAYOUTS, mirrorCrystal, solveTraced, Panel, StrokeInput, digitOf, loadCrystal, mnistKit,
  panelToField, parseShard, solve,
} from "../src/index.js";
import { paintDigitScreen } from "../src/problems/mnist/digit-screen.js";
import { fixture, relErr, realCrystal } from "./helpers.js";

const crystal = realCrystal("mnist-padrot.safetensors");
// the original data set, when NC_DATA points at it (a folder holding qmnist/test-000.json); the tests that need it skip otherwise
const DATA = process.env["NC_DATA"];
const shardFile = DATA ? join(DATA, "qmnist/test-000.json") : "";
const haveKitFixture = existsSync(fileURLToPath(new URL("./fixtures/mnist-kit.safetensors", import.meta.url)));

describe.runIf(existsSync(crystal) && existsSync(shardFile) && haveKitFixture)("MNIST kit against the trainer", () => {
  it("reads real test hands exactly as the trainer does", async () => {
    const f = fixture("mnist-kit");
    const meta = f.meta as { idx: number[]; pred: number[]; labels: number[] };
    const data = await loadCrystal(new Uint8Array(readFileSync(crystal)));
    const kit = mnistKit(data);
    expect(kit.layout.name).toBe("padrot");
    const eng = new CrystalEngine(data);
    const shard = parseShard(readFileSync(shardFile, "utf8"));
    const P = f.t("expect.p");
    const eff = f.t("expect.eff");
    for (let j = 0; j < meta.idx.length; j++) {
      const { answer } = await solve(kit, eng, digitOf(shard, meta.idx[j]!));
      expect(answer.pick).toBe(meta.pred[j]);
      expect(relErr(answer.scores, P.subarray(10 * j, 10 * j + 10))).toBeLessThan(2e-5);
      expect(answer.efficiency).toBeCloseTo(eff[j]!, 5);
    }
  });
});

describe.runIf(existsSync(crystal) && existsSync(shardFile))("the light trace", () => {
  it("ends at the solve's own exit field, one field per surface", async () => {
    const data = await loadCrystal(new Uint8Array(readFileSync(crystal)));
    const kit = mnistKit(data), eng = new CrystalEngine(data), shard = parseShard(readFileSync(shardFile, "utf8"));
    const frame = kit.frame(digitOf(shard, 3));
    const run = await eng.run(frame, kit.frameSize), tr = await eng.trace(frame, kit.frameSize), P = data.model.planes.length;
    expect(tr.leaving).toHaveLength(P);
    expect(tr.arriving).toHaveLength(P + 1);
    const ex = tr.arriving[P]!;
    let d = 0, n = 0;
    for (let k = 0; k < ex.re.length; k++) {
      d += (ex.re[k]! - run.exit.re[k]!) ** 2 + (ex.im[k]! - run.exit.im[k]!) ** 2;
      n += run.exit.re[k]! ** 2 + run.exit.im[k]! ** 2;
    }
    expect(Math.sqrt(d / n)).toBeLessThan(1e-9);
    const a = await solveTraced(kit, eng, digitOf(shard, 3)), b = await solve(kit, eng, digitOf(shard, 3));
    expect(a.trace).toBeDefined(); // one pass: the answer and the light to draw
    expect(a.answer.pick).toBe(b.answer.pick);
    expect(relErr(a.answer.scores, b.answer.scores)).toBeLessThan(1e-9);
  });
});

describe.runIf(existsSync(crystal) && existsSync(shardFile))("the mirrored crystal", () => {
  it("reads every digit exactly as the original, its light mirrored", async () => {
    const data = await loadCrystal(new Uint8Array(readFileSync(crystal)));
    const md = mirrorCrystal(data), shard = parseShard(readFileSync(shardFile, "utf8"));
    const k0 = mnistKit(data), k1 = mnistKit(md), e0 = new CrystalEngine(data), e1 = new CrystalEngine(md);
    for (const idx of [0, 7, 42, 99]) {
      const d = digitOf(shard, idx), a = await solve(k0, e0, d), b = await solve(k1, e1, d);
      expect(b.answer.pick).toBe(a.answer.pick);
      expect(relErr(b.answer.scores, a.answer.scores)).toBeLessThan(1e-6);
      const N = e0.geometry.N, E0 = a.run.exit, E1 = b.run.exit;
      let dd = 0, nn = 0;
      for (let y = 0; y < N; y++)
        for (let x = 0; x < N; x++) {
          const p = y * N + x, q = y * N + (N - 1 - x);
          dd += (E1.re[p]! - E0.re[q]!) ** 2 + (E1.im[p]! - E0.im[q]!) ** 2;
          nn += E0.re[q]! ** 2 + E0.im[q]! ** 2;
        }
      expect(Math.sqrt(dd / nn)).toBeLessThan(1e-6);
    }
  });
});

describe("detector layouts", () => {
  it("padrot is the pad turned 180°", () => {
    const pad = DETECTOR_LAYOUTS["pad"]!.centres, rot = DETECTOR_LAYOUTS["padrot"]!.centres;
    pad.forEach(([x, y], d) => {
      expect(rot[d]![0]).toBeCloseTo(-x);
      expect(rot[d]![1]).toBeCloseTo(-y);
    });
  });
});

describe("handwriting", () => {
  it("sizes and centres a small digit the way MNIST did", () => {
    const s = new StrokeInput();
    s.begin({ x: 3, y: 3, w: 2.1 });
    s.to({ x: 3, y: 11, w: 2.1 }); // a short "1" in the top-left corner
    const n = s.normalised()!;
    let top = 28, bottom = -1, mx = 0, my = 0, m = 0;
    for (let y = 0; y < 28; y++)
      for (let x = 0; x < 28; x++) {
        const v = n[y * 28 + x]!;
        if (v > 24 / 255) {
          top = Math.min(top, y);
          bottom = Math.max(bottom, y);
        }
        mx += (x + 0.5) * v;
        my += (y + 0.5) * v;
        m += v;
      }
    expect(bottom - top + 1).toBeGreaterThanOrEqual(19);
    expect(bottom - top + 1).toBeLessThanOrEqual(21);
    expect(mx / m).toBeCloseTo(14, 0);
    expect(my / m).toBeCloseTo(14, 0);
  });
  it("draws nothing when empty", () => {
    expect(new StrokeInput().normalised()).toBeNull();
  });
});

describe("the MNIST screen", () => {
  it("lights the pad cell of the digit read, with its figure knocked out", () => {
    const panel = new Panel();
    paintDigitScreen(panel, { digit: null, read: 5 });
    const at = (x: number, y: number): number => panel.px[y * 96 + x]!;
    expect(at(47 + 16 + 1, 4 + 22 + 1)).toBe(1); // the 5's cell (row 1, column 1) is filled
    expect(at(47 + 1, 4 + 1)).toBe(0); // the 1's cell is not
    let dark = 0;
    for (let y = 4 + 22; y < 4 + 43; y++) for (let x = 47 + 16; x < 47 + 31; x++) if (at(x, y) === 0) dark++;
    expect(dark).toBeGreaterThan(20); // the figure, knocked out of the fill
  });
  it("maps the drawing box to the digit field", () => {
    expect(panelToField(3, 21)).toEqual({ x: 4, y: 2 });
    expect(panelToField(2, 21)).toBeNull();
  });
});
