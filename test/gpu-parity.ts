// The WebGPU light path against the CPU reference and the Python package, in Node through Dawn (the `webgpu` package, the Metal/Vulkan/D3D
// backends Chrome uses). A plain script, not a vitest suite: Dawn's native module does not survive vitest's workers.
//   npm run test:gpu        — the 64² fixtures only, a few MB of GPU memory
import "./f16-polyfill.js";
import { existsSync, readFileSync } from "node:fs";
import { create, globals } from "webgpu";
import { CrystalEngine, WebGpuBackend, gridFits, loadCrystal, requestGpu } from "../src/index.js";
import { fieldErr, fixture, mismatch, realCrystal } from "./helpers.js";

Object.assign(globalThis, globals);
const instance = create([]) as unknown as GPU; // held: if the Dawn instance is collected mid-run, Dawn crashes
const gpu = await requestGpu(instance);
if (!gpu) throw new Error("no WebGPU adapter");
let failed = 0;
const check = (what: string, v: number, limit: number): void => {
  const ok = v < limit;
  if (!ok) failed++;
  process.stderr.write(`${ok ? "ok  " : "FAIL"} ${what}: ${v.toExponential(2)} (limit ${limit})\n`);
};

for (const name of ["quad", "single", "roles", "pupil"]) {
  process.stderr.write(`-- ${name}\n`);
  const fx = fixture(name);
  if (!gridFits(gpu.limits, fx.data.model.N)) throw new Error(`${name}: grid does not fit this GPU`);
  const cpu = new CrystalEngine(fx.data);
  const eng = new CrystalEngine(fx.data, (g) => new WebGpuBackend(gpu.device, g));
  for (let k = 0; k < eng.quad.hops; k++) {
    const pic = k === 0 ? fx.t("expect.frame") : fx.t(`expect.hidden.${k - 1}`);
    const plan = eng.plan(eng.quad.passSection[k] ?? k);
    const E = await eng.backend.runSection(eng.dmd(pic, fx.F, k > 0), plan);
    const C = await cpu.backend.runSection(cpu.dmd(pic, fx.F, k > 0), plan);
    check(`${name} pass ${k} GPU vs CPU`, fieldErr(E, Float32Array.from(C.re), Float32Array.from(C.im)), 5e-5);
    check(`${name} pass ${k} GPU vs Python`, fieldErr(E, fx.t(`expect.pass.${k}.re`), fx.t(`expect.pass.${k}.im`)), 5e-5);
  }
  if (eng.quad.hops === 1) {
    // the light trace: every surface's field in one GPU submission (runStages) against the CPU's surface-by-surface run
    const tg = await eng.trace(fx.t("expect.frame"), fx.F), tc = await cpu.trace(fx.t("expect.frame"), fx.F);
    let worst = 0;
    const cmp = (a: { re: Float64Array | Float32Array; im: Float64Array | Float32Array }, b: { re: Float64Array | Float32Array; im: Float64Array | Float32Array }): void => {
      worst = Math.max(worst, fieldErr(a as never, Float32Array.from(b.re), Float32Array.from(b.im)));
    };
    cmp(tg.arrival, tc.arrival);
    tg.leaving.forEach((f, i) => cmp(f, tc.leaving[i]!));
    tg.arriving.forEach((f, i) => cmp(f, tc.arriving[i]!));
    check(`${name} light trace GPU vs CPU (worst surface)`, worst, 5e-5);
  }
  if (eng.quad.hops > 1) {
    // the quad traced (traceStages): every pass surface by surface, read back over the window only (row copies), against the CPU's
    const tg = await eng.traceStages(fx.t("expect.frame"), fx.F), tc = await cpu.traceStages(fx.t("expect.frame"), fx.F);
    let worst = 0;
    tg.passes.forEach((p, k) => {
      const c = tc.passes[k]!;
      p.arriving.forEach((f, i) => (worst = Math.max(worst, fieldErr(f, Float32Array.from(c.arriving[i]!.re), Float32Array.from(c.arriving[i]!.im)))));
      p.leaving.forEach((f, i) => (worst = Math.max(worst, fieldErr(f, Float32Array.from(c.leaving[i]!.re), Float32Array.from(c.leaving[i]!.im)))));
    });
    check(`${name} quad trace over the window, GPU vs CPU (worst surface)`, worst, 5e-5);
    tg.hiddens.forEach((h, k) => check(`${name} quad trace hidden ${k} levels off`, mismatch(h, fx.t(`expect.hidden.${k}`)), 0.01));
    // a phone's half-precision trace (halfTraces): every surface within f16's rounding, the exit and the hidden pictures exactly as before
    eng.halfTraces = true;
    const th = await eng.traceStages(fx.t("expect.frame"), fx.F);
    eng.halfTraces = false;
    let worstH = 0, worstExit = 0;
    th.passes.forEach((p, k) => {
      const c = tc.passes[k]!, last = p.arriving.length - 1;
      p.arriving.forEach((f, i) => {
        const e = fieldErr(f, Float32Array.from(c.arriving[i]!.re), Float32Array.from(c.arriving[i]!.im));
        if (i === last) worstExit = Math.max(worstExit, e);
        else worstH = Math.max(worstH, e);
      });
      p.leaving.forEach((f, i) => (worstH = Math.max(worstH, fieldErr(f, Float32Array.from(c.leaving[i]!.re), Float32Array.from(c.leaving[i]!.im)))));
    });
    check(`${name} half quad trace, GPU vs CPU (worst surface)`, worstH, 2e-3);
    check(`${name} half quad trace exit, GPU vs CPU`, worstExit, 5e-5);
    th.hiddens.forEach((h, k) => check(`${name} half quad trace hidden ${k} levels off`, mismatch(h, fx.t(`expect.hidden.${k}`)), 0.01));
    // a phone's traced move in its kept scratch (reuseReads): twice in a row, the fields handed back between — the exit and the hidden
    // pictures exactly the plain trace's
    eng.reuseReads = eng.halfTraces = true;
    for (let rep = 0; rep < 2; rep++) {
      const ts = await eng.traceStages(fx.t("expect.frame"), fx.F);
      check(`${name} scratch quad trace ${rep} exit vs plain`, fieldErr(ts.run.exit, Float32Array.from(tg.run.exit.re), Float32Array.from(tg.run.exit.im)), 5e-6);
      ts.hiddens.forEach((h, k) => check(`${name} scratch quad trace ${rep} hidden ${k} vs plain`, mismatch(h, tg.hiddens[k]!), 1e-9));
      for (const p of ts.passes) eng.recycleFields([p.arrival, ...p.arriving.slice(0, -1), ...p.leaving]);
    }
    eng.reuseReads = eng.halfTraces = false;
  }
  const r = await eng.run(fx.t("expect.frame"), fx.F);
  r.hiddens.forEach((h, k) => check(`${name} hidden ${k} levels off`, mismatch(h, fx.t(`expect.hidden.${k}`)), 0.01));
  check(`${name} exit GPU vs Python`, fieldErr(r.exit, fx.t("expect.exit.re"), fx.t("expect.exit.im")), 5e-5);
  eng.dispose();
}
// a real trained crystal, when NC_CRYSTALS points at a folder holding it
const real = realCrystal("mnist-padrot.safetensors");
if (existsSync(real) && existsSync(new URL("./fixtures/mnist-real.safetensors", import.meta.url))) {
  process.stderr.write("-- mnist-padrot (real, 512²)\n");
  const f = fixture("mnist-real");
  const eng = new CrystalEngine(await loadCrystal(new Uint8Array(readFileSync(real))), (g) => new WebGpuBackend(gpu.device, g));
  const t0 = performance.now();
  const r = await eng.run(f.t("expect.frame"), f.F);
  const ms = performance.now() - t0;
  check("mnist-padrot exit GPU vs Python", fieldErr(r.exit, f.t("expect.exit.re"), f.t("expect.exit.im")), 5e-5);
  const t1 = performance.now();
  for (let i = 0; i < 5; i++) await eng.run(f.t("expect.frame"), f.F);
  process.stderr.write(`     first solve ${ms.toFixed(0)} ms (builds H, uploads maps), then ${((performance.now() - t1) / 5).toFixed(1)} ms a solve\n`);
  const tr = await eng.trace(f.t("expect.frame"), f.F);
  check("mnist-padrot trace exit vs Python", fieldErr(tr.arriving[tr.arriving.length - 1]!, f.t("expect.exit.re"), f.t("expect.exit.im")), 5e-5);
  const t2 = performance.now();
  for (let i = 0; i < 5; i++) await eng.trace(f.t("expect.frame"), f.F);
  process.stderr.write(`     light trace (11 fields, one readback): ${((performance.now() - t2) / 5).toFixed(1)} ms\n`);
  // the trace pool: a recycled trace is written over, not reallocated, and comes back right (scribbled on first, so stale values would show)
  const exitRe = tr.arriving[tr.arriving.length - 1]!.re;
  for (const fld of [tr.input, tr.arrival, ...tr.leaving, ...tr.arriving]) fld.re.fill(7);
  eng.recycle(tr);
  const tr2 = await eng.trace(f.t("expect.frame"), f.F);
  if (tr2.arriving[tr2.arriving.length - 1]!.re !== exitRe) throw new Error("trace pool: the recycled trace's fields were not reused");
  check("mnist-padrot recycled trace exit vs Python", fieldErr(tr2.arriving[tr2.arriving.length - 1]!, f.t("expect.exit.re"), f.t("expect.exit.im")), 5e-5);
  eng.dispose();
}
gpu.device.destroy();
void instance;
console.log(failed ? `${failed} FAILED` : "all passed");
process.exit(failed ? 1 : 0);
