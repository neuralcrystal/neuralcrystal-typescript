import { gzipSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import { loadCrystal, readCrystal, writeSafetensors } from "../src/index.js";

const model = { format: "neuralcrystal.model.v2", N: 8, dx_um: 1, planes: [{ kind: "learned", dist_mm: 0.1 }, { kind: "lens", param: 1 }] };
const phi0 = Float32Array.from({ length: 16 }, (_, i) => i / 10);
const phi2 = Float32Array.from({ length: 16 }, (_, i) => -i);
const v2 = writeSafetensors({ format: "neuralcrystal.model.v2", neuralcrystal: JSON.stringify(model) }, [
  ["plane.0.phi", phi0, [4, 4]],
  ["plane.0.phi.2", phi2, [4, 4]],
]);

describe("crystal files", () => {
  it("reads a v2 file: section 0 is plane.<i>.phi, missing sections are null", () => {
    const c = readCrystal(v2);
    expect(c.model.planes).toHaveLength(2);
    expect(c.mapN).toEqual([4, 0]);
    expect(Array.from(c.maps[0]![0]!)).toEqual(Array.from(phi0));
    expect(c.maps[0]![1]).toBeNull();
    expect(Array.from(c.maps[0]![2]!)).toEqual(Array.from(phi2));
    expect(c.maps[1]).toEqual([null]);
  });
  it("reads a v1 JSON file (base64 maps)", () => {
    const b64 = Buffer.from(phi0.buffer).toString("base64");
    const v1 = { ...model, format: "neuralcrystal.model.v1", planes: [{ ...model.planes[0], phi: b64, n: 4 }, model.planes[1]] };
    const c = readCrystal(new TextEncoder().encode(JSON.stringify(v1)));
    expect(Array.from(c.maps[0]![0]!)).toEqual(Array.from(phi0));
  });
  it("unpacks gzip by its magic bytes and reports progress", async () => {
    const gz = new Uint8Array(gzipSync(v2));
    const stages: string[] = [];
    const c = await loadCrystal(gz, { onProgress: (p) => stages.push(p.stage) });
    expect(Array.from(c.maps[0]![2]!)).toEqual(Array.from(phi2));
    expect(stages).toEqual(["unpack", "read"]);
  });
  it("streams a Response with download progress", async () => {
    const gz = new Uint8Array(gzipSync(v2));
    const res = new Response(gz, { headers: { "content-length": String(gz.length) } });
    let last = 0;
    const c = await loadCrystal(res, { onProgress: (p) => p.stage === "download" && (last = p.loaded) });
    expect(last).toBe(gz.length);
    expect(c.model.N).toBe(8);
  });
  it("rejects files that are not crystals", () => {
    expect(() => readCrystal(new TextEncoder().encode('{"hello":1}'))).toThrow(/not a NeuralCrystal/);
  });
});
