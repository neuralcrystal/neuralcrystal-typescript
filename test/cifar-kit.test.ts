import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { cifarKit, flipPicture } from "../src/index.js";
import { cifarFrame } from "../src/problems/cifar/frame.js";
import type { CrystalData } from "../src/index.js";

// The trainer's frames for test-000 pictures 0–2 (gcn 0.216, dark plane, mega, 5 levels, frame 256), as quarters
const ref = JSON.parse(readFileSync(new URL("./fixtures/cifar-frames.json", import.meta.url), "utf8")) as { pics: string; frames: string[] };
const b64 = (s: string): Uint8Array => Uint8Array.from(Buffer.from(s, "base64"));
const pics = b64(ref.pics);
const spec = { frame: 256, levels: 5, gcn: 0.216, p4: "dark" as const };

describe("the CIFAR kit", () => {
  it("draws the frame exactly as the trainer does (gcn, the dark plane, the mega code)", () => {
    ref.frames.forEach((f, i) => {
      const want = b64(f), got = cifarFrame(pics.subarray(i * 3072, (i + 1) * 3072), spec);
      let bad = 0;
      for (let k = 0; k < want.length; k++) if (Math.round(got[k]! * 4) !== want[k]) bad++;
      expect(bad).toBe(0);
    });
  });

  it("reads its frame spec and the 20 squares from the crystal file", () => {
    const model = { N: 1024, winUm: 576, tr: { det: "diff", cifar: { gcn: 0.216, p4: "dark", frame: 256, code: "mega" } }, quad: { mir: 2, levels: 5 }, planes: [] };
    const kit = cifarKit({ model } as unknown as CrystalData);
    expect(kit.spec).toEqual(spec);
    const r = kit.rects({ geometry: { N: 1024, winFrac: 576 / 1024 } } as never);
    expect(r.length).toBe(20);
    expect(r[0]).toEqual({ x0: 235, y0: 247, w: 92, h: 92 }); // 92 × 92 at N 1024, a 576 µm window
    expect(r[10]!.y0).toBe(546);
  });

  it("mirrors a picture left to right", () => {
    const p = pics.subarray(0, 3072), f = flipPicture(p);
    expect(f[0]).toBe(p[31 * 3]);
    expect(Array.from(flipPicture(f))).toEqual(Array.from(p));
  });
});
