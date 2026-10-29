// The WebGPU light path: a section's plan as row and column kernel dispatches in one command buffer, the field read back once at the
// exit. Transfer functions are built in float64 on the CPU (physics/transfer.ts) and kept on the GPU by hop length; phase maps are
// wrapped to [−π, π] in float64 before they become float32 (a trained map runs to −1000 rad, where f32 sin/cos lose digits).
import type { Crop, LightBackend } from "../light/backend.js";
import type { Field } from "../light/field.js";
import { Float16 } from "../light/field.js";
import type { Geometry } from "../physics/geometry.js";
import type { Op, SectionPlan } from "../physics/plan.js";
import { pupilH, transferH } from "../physics/transfer.js";
import { LruBuffers } from "./lru-buffers.js";
import { ROW_ABSORB, ROW_FORWARD, ROW_INVERSE, ROW_PHASE, colKernel, packKernel, rowKernel } from "./shaders.js";

const UNI_STRIDE = 256;
// GPUBufferUsage is read lazily: importing the engine must work where WebGPU does not exist (Node, server rendering)
const usage = (): typeof GPUBufferUsage => GPUBufferUsage;

interface Dispatch {
  kind: "row" | "col";
  flags: number;
  buf: GPUBuffer | null; // the phase map (row) or H (col)
  /** the phase map held as 8-bit steps (row8) */
  q8?: boolean;
}

export interface WebGpuOptions {
  /**
   * an 8-bit crystal's learned maps held on the GPU as their 8-bit steps, a quarter of the memory, the same light (for phones:
   * a chess crystal's 80 maps were 335 MB as 32-bit floats)
   */
  phase8?: boolean;
}

export class WebGpuBackend implements LightBackend {
  readonly name = "webgpu";
  private readonly N: number;
  private readonly row: GPUComputePipeline;
  private readonly col: GPUComputePipeline;
  private readonly x: GPUBuffer;
  private readonly tw: GPUBuffer;
  private readonly readBuf: GPUBuffer;
  private readonly dummy: GPUBuffer;
  private uni: GPUBuffer;
  private uniCap = 64;
  /** runStages' staging buffers, kept between traces */
  private readonly stages: GPUBuffer[] = [];
  /** the input's upload, kept */
  private upBuf: Float32Array<ArrayBuffer> | null = null;
  private readonly hBufs: LruBuffers;
  /** the half-precision readback's kernel, its window uniform and its packed field (made on the first half trace) */
  private pack: { pipe: GPUComputePipeline; uni: GPUBuffer; out: GPUBuffer } | null = null;
  /** the half trace's staging buffer, kept between passes (a phone: ~135 MB a chess pass, made and destroyed four times a move before) */
  private halfStage: GPUBuffer | null = null;
  /** half fields' 16-bit words handed back (recycleFields), by length: the next half trace is read into them */
  private halfPool: Uint16Array[] = [];
  private readonly phBufs: LruBuffers;

  private row8: GPUComputePipeline | null = null;
  /**
   * BIND GROUPS KEPT (an iPhone ran out of memory while chess thought): every dispatch used to make a new one — ~240 a chess read,
   * ~30,000 a look-ahead — each a native object freed only when Safari collects it. Now one per pipeline, buffer and dispatch slot, made
   * once; a buffer evicted from the LRU takes its bind groups with it (keyed weakly), a new uniform buffer clears them all.
   */
  private bindGroups = new WeakMap<GPUComputePipeline, WeakMap<GPUBuffer, GPUBindGroup[]>>();
  private packGroup: GPUBindGroup | null = null;
  /** the dispatches' uniforms, written here then copied to `uni` (kept: one array per encode before) */
  private ub = new ArrayBuffer(0);
  private readonly phase8: boolean;

  constructor(private readonly device: GPUDevice, private readonly g: Geometry, opts: WebGpuOptions = {}) {
    this.phase8 = !!opts.phase8;
    const N = (this.N = g.N);
    const mk = (code: string): GPUComputePipeline => device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
    this.row = mk(rowKernel(N));
    this.col = mk(colKernel(N));
    this.x = device.createBuffer({ size: 8 * N * N, usage: usage().STORAGE | usage().COPY_DST | usage().COPY_SRC });
    this.readBuf = device.createBuffer({ size: 8 * N * N, usage: usage().MAP_READ | usage().COPY_DST });
    this.dummy = device.createBuffer({ size: 16, usage: usage().STORAGE });
    const tw = new Float32Array(N);
    for (let m = 0; m < N / 2; m++) {
      tw[2 * m] = Math.cos((2 * Math.PI * m) / N);
      tw[2 * m + 1] = -Math.sin((2 * Math.PI * m) / N);
    }
    this.tw = this.upload(tw);
    this.uni = device.createBuffer({ size: UNI_STRIDE * this.uniCap, usage: usage().UNIFORM | usage().COPY_DST });
    this.hBufs = new LruBuffers(48);
    this.phBufs = new LruBuffers(160);
  }

  async runSection(input: Field, plan: SectionPlan, out?: Field): Promise<Field> {
    const N = this.N;
    if (!this.upBuf || this.upBuf.length !== 2 * N * N) this.upBuf = new Float32Array(2 * N * N);
    const a = this.upBuf;
    for (let k = 0; k < N * N; k++) {
      a[2 * k] = input.re[k]!;
      a[2 * k + 1] = input.im[k]!;
    }
    this.device.queue.writeBuffer(this.x, 0, a);
    this.encode(this.schedule(plan.ops));
    return this.readField(out);
  }

  /**
   * Several plans in turn, the field after each one kept (the light trace: every surface's arriving and leaving field). All of them go
   * in ONE command buffer, the field copied into a staging buffer after each plan, and read back ONCE — not a round trip per plan.
   */
  async runStages(input: Field, plans: SectionPlan[], into?: Field[], crop?: Crop, half = false, lastInto?: Field): Promise<Field[]> {
    const N = this.N, M = crop ? crop.n : N, bytes = 8 * M * M, n = plans.length;
    if (!n) return [];
    if (half && Float16 && n > 1) return this.runStagesHalf(input, plans, crop ?? { x0: 0, y0: 0, n: N }, lastInto);
    if (!this.upBuf || this.upBuf.length !== 2 * N * N) this.upBuf = new Float32Array(2 * N * N);
    const a = this.upBuf;
    for (let k = 0; k < N * N; k++) {
      a[2 * k] = input.re[k]!;
      a[2 * k + 1] = input.im[k]!;
    }
    this.device.queue.writeBuffer(this.x, 0, a);
    // the staging buffer is kept and reused (a trace a digit made and destroyed a 23 MB one each time), a second made only while one is mapped
    const size = bytes * n, stage = this.stages.pop() ?? this.device.createBuffer({ size, usage: usage().MAP_READ | usage().COPY_DST });
    if (stage.size < size) {
      stage.destroy();
      return this.runStages(input, plans, into, crop);
    }
    this.encode(plans.map((p) => this.schedule(p.ops)), stage, crop);
    await stage.mapAsync(GPUMapMode.READ, 0, size);
    const v = new Float32Array(stage.getMappedRange(0, size));
    const out: Field[] = [];
    for (let s = 0; s < n; s++) {
      // single precision, as the GPU computed it: half the memory of doubles (a trace is 11 fields — at 512², 23 MB not 46 a digit)
      const w = into?.[s], reuse = !!w && w.N === M && w.re instanceof Float32Array && w.im instanceof Float32Array;
      const re = reuse ? (w.re as Float32Array) : new Float32Array(M * M), im = reuse ? (w.im as Float32Array) : new Float32Array(M * M), o = (s * bytes) / 4;
      for (let k = 0; k < M * M; k++) {
        re[k] = v[o + 2 * k]!;
        im[k] = v[o + 2 * k + 1]!;
      }
      out.push({ N: M, re, im });
    }
    stage.unmap();
    if (this.stages.length < 2) this.stages.push(stage);
    else stage.destroy();
    return out;
  }

  /**
   * runStages in HALF PRECISION (for phones): each stage's window is packed to f16 on the GPU before it is copied out, so
   * the readback buffer and the fields are half the size; the last stage (the exit, what the next pass and the readout are made from)
   * comes back in single precision as before. Its staging buffer is made for the call and destroyed after it: nothing this size is kept.
   */
  private async runStagesHalf(input: Field, plans: SectionPlan[], crop: Crop, lastInto?: Field): Promise<Field[]> {
    const N = this.N, M = crop.n, MM = M * M, n = plans.length, hb = 4 * MM, size = hb * (n - 1) + 8 * MM;
    if (!this.upBuf || this.upBuf.length !== 2 * N * N) this.upBuf = new Float32Array(2 * N * N);
    const a = this.upBuf;
    for (let k = 0; k < N * N; k++) {
      a[2 * k] = input.re[k]!;
      a[2 * k + 1] = input.im[k]!;
    }
    this.device.queue.writeBuffer(this.x, 0, a);
    if (!this.pack || this.pack.out.size < hb) {
      this.pack?.uni.destroy();
      this.pack?.out.destroy();
      const pipe = this.pack?.pipe ?? this.device.createComputePipeline({ layout: "auto", compute: { module: this.device.createShaderModule({ code: packKernel() }), entryPoint: "main" } });
      this.pack = { pipe, uni: this.device.createBuffer({ size: 16, usage: usage().UNIFORM | usage().COPY_DST }), out: this.device.createBuffer({ size: hb, usage: usage().STORAGE | usage().COPY_SRC }) };
      this.packGroup = null; // it bound the old ones
    }
    this.device.queue.writeBuffer(this.pack.uni, 0, new Uint32Array([crop.x0, crop.y0, M, N]));
    // the kept staging buffer when it is free and big enough; one made for the call otherwise (another half trace has it mapped)
    let stage = this.halfStage;
    if (!stage || stage.size < size) {
      stage?.destroy();
      stage = this.halfStage = this.device.createBuffer({ size, usage: usage().MAP_READ | usage().COPY_DST });
    }
    this.halfStage = null;
    let kept = true;
    const words = (): Uint16Array => {
      const i = this.halfPool.findIndex((w) => w.length === MM);
      return i >= 0 ? this.halfPool.splice(i, 1)[0]! : new Uint16Array(MM);
    };
    try {
      this.encode(plans.map((p) => this.schedule(p.ops)), stage, crop, true);
      await stage.mapAsync(GPUMapMode.READ, 0, size);
      const buf = stage.getMappedRange(0, size), u32 = new Uint32Array(buf, 0, (n - 1) * MM), f32 = new Float32Array(buf, (n - 1) * hb, 2 * MM);
      const out: Field[] = [];
      for (let s = 0; s < n - 1; s++) {
        const re = words(), im = words(), o = s * MM;
        for (let k = 0; k < MM; k++) {
          const w = u32[o + k]!;
          re[k] = w & 0xffff;
          im[k] = w >>> 16;
        }
        out.push({ N: M, re: new Float16!(re.buffer as ArrayBuffer), im: new Float16!(im.buffer as ArrayBuffer) });
      }
      const intoLast = !!lastInto && lastInto.N === M && lastInto.re instanceof Float32Array && lastInto.im instanceof Float32Array;
      const re = intoLast ? (lastInto.re as Float32Array) : new Float32Array(MM), im = intoLast ? (lastInto.im as Float32Array) : new Float32Array(MM);
      for (let k = 0; k < MM; k++) {
        re[k] = f32[2 * k]!;
        im[k] = f32[2 * k + 1]!;
      }
      out.push({ N: M, re, im });
      stage.unmap();
      return out;
    } catch (e) {
      kept = false;
      throw e;
    } finally {
      if (kept && !this.halfStage) this.halfStage = stage;
      else stage.destroy();
    }
  }

  dispose(): void {
    this.hBufs.clear();
    this.phBufs.clear();
    for (const b of [this.x, this.tw, this.readBuf, this.dummy, this.uni, ...this.stages]) b.destroy();
    this.pack?.uni.destroy();
    this.pack?.out.destroy();
    this.pack = null;
    this.halfStage?.destroy();
    this.halfStage = null;
    this.halfPool.length = 0;
    this.stages.length = 0;
  }

  /** The plan as dispatches. State: "space" (x holds the field), "ready" (rows transformed, a column pass next), "after" (a hop done). */
  private schedule(ops: Op[]): Dispatch[] {
    const out: Dispatch[] = [];
    const abs = this.g.absorb ? ROW_ABSORB : 0;
    let s: "space" | "ready" | "after" = "space";
    for (let i = 0; i < ops.length; i++) {
      const op = ops[i]!;
      if (op.op === "phase") {
        const nextHop = ops[i + 1]?.op === "hop";
        const flags = ROW_PHASE | (s === "after" ? ROW_INVERSE | abs : 0) | (nextHop ? ROW_FORWARD : 0);
        const q8 = this.phase8 && op.levels === 256;
        out.push({ kind: "row", flags, buf: q8 ? this.phaseBuf8(op.key, op.phi, op.free, op.steps) : this.phaseBuf(op.key, op.phi, op.free), q8 });
        s = nextHop ? "ready" : "space";
        continue;
      }
      const H = this.hBuf(op.zSub);
      for (let k = 0; k < op.n; k++) {
        const Hk = op.pupil && k === 0 ? this.hBuf(op.zSub, true) : H;
        if (s === "space") out.push({ kind: "row", flags: ROW_FORWARD, buf: null });
        else if (s === "after") out.push({ kind: "row", flags: ROW_INVERSE | abs | ROW_FORWARD, buf: null });
        out.push({ kind: "col", flags: 0, buf: Hk });
        s = "after";
      }
    }
    if (s === "after") out.push({ kind: "row", flags: ROW_INVERSE | abs, buf: null });
    return out;
  }

  /** one command buffer: the dispatches (or several segments of them, the field copied into `stage` after each), then the readback copy */
  private encode(segs: Dispatch[] | Dispatch[][], stage: GPUBuffer | null = null, crop?: Crop, half = false): void {
    const { device, N } = this;
    const parts: Dispatch[][] = segs.length && Array.isArray(segs[0]) ? (segs as Dispatch[][]) : [segs as Dispatch[]];
    const ds = parts.flat();
    if (ds.length > this.uniCap) {
      this.uni.destroy();
      this.uniCap = Math.max(ds.length, 2 * this.uniCap);
      this.uni = device.createBuffer({ size: UNI_STRIDE * this.uniCap, usage: usage().UNIFORM | usage().COPY_DST });
      this.bindGroups = new WeakMap(); // they bound the old uniform buffer
    }
    if (this.ub.byteLength < UNI_STRIDE * ds.length) this.ub = new ArrayBuffer(UNI_STRIDE * this.uniCap);
    const ub = this.ub;
    const u32 = new Uint32Array(ub);
    const f32 = new Float32Array(ub);
    const box = this.g.absorb ?? { x0: 0, size: N };
    const enc = device.createCommandEncoder();
    let i0 = 0;
    parts.forEach((part, si) => {
    const cp = enc.beginComputePass();
    part.forEach((d, j) => {
      const i = i0 + j;
      const o = (i * UNI_STRIDE) / 4;
      u32[o] = d.flags;
      u32[o + 1] = box.x0;
      u32[o + 2] = box.size;
      f32[o + 4] = d.kind === "col" ? 1 / N : 1;
      const pipe = d.kind === "row" ? (d.q8 ? (this.row8 ??= this.device.createComputePipeline({ layout: "auto", compute: { module: this.device.createShaderModule({ code: rowKernel(N, true) }), entryPoint: "main" } })) : this.row) : this.col;
      const buf = d.buf ?? this.dummy;
      let byBuf = this.bindGroups.get(pipe);
      if (!byBuf) this.bindGroups.set(pipe, (byBuf = new WeakMap()));
      let slots = byBuf.get(buf);
      if (!slots) byBuf.set(buf, (slots = []));
      const bg = (slots[i] ??= device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.uni, offset: i * UNI_STRIDE, size: 32 } },
        { binding: 1, resource: { buffer: this.x } },
        { binding: 2, resource: { buffer: this.tw } },
        { binding: 3, resource: { buffer: buf } },
      ] }));
      cp.setPipeline(pipe);
      cp.setBindGroup(0, bg);
      cp.dispatchWorkgroups(N);
    });
    const packed = half && stage && crop && this.pack && si < parts.length - 1;
    if (packed) {
      // this stage's window to f16 pairs in pack.out (its own dispatch: it sees the FFTs' writes), copied out below
      const P = this.pack!, M = crop!.n;
      cp.setPipeline(P.pipe);
      cp.setBindGroup(0, (this.packGroup ??= device.createBindGroup({ layout: P.pipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: P.uni } }, { binding: 1, resource: { buffer: this.x } }, { binding: 2, resource: { buffer: P.out } }] })));
      cp.dispatchWorkgroups(Math.ceil((M * M) / 256));
    }
    cp.end();
    i0 += part.length;
    if (packed) enc.copyBufferToBuffer(this.pack!.out, 0, stage!, si * 4 * crop!.n * crop!.n, 4 * crop!.n * crop!.n);
    else if (stage && crop && half) {
      // the last stage of a half trace: single precision, after the packed ones
      const M = crop.n, base = 4 * M * M * (parts.length - 1);
      for (let r = 0; r < M; r++) enc.copyBufferToBuffer(this.x, 8 * ((crop.y0 + r) * N + crop.x0), stage, base + 8 * r * M, 8 * M);
    } else if (stage && crop) {
      // the window only: its rows, one copy each (a quad section's trace at 1024² is 8 MB a surface; its 512 window a quarter of that)
      const M = crop.n;
      for (let r = 0; r < M; r++) enc.copyBufferToBuffer(this.x, 8 * ((crop.y0 + r) * N + crop.x0), stage, si * 8 * M * M + 8 * r * M, 8 * M);
    } else if (stage) enc.copyBufferToBuffer(this.x, 0, stage, si * 8 * N * N, 8 * N * N); // this segment's field
    });
    device.queue.writeBuffer(this.uni, 0, ub, 0, UNI_STRIDE * ds.length);
    if (!stage) enc.copyBufferToBuffer(this.x, 0, this.readBuf, 0, 8 * N * N);
    device.queue.submit([enc.finish()]);
  }

  private async readField(out?: Field): Promise<Field> {
    const N = this.N;
    await this.readBuf.mapAsync(GPUMapMode.READ);
    const v = new Float32Array(this.readBuf.getMappedRange());
    const into = !!out && out.N === N && out.re instanceof Float64Array && out.im instanceof Float64Array;
    const re = into ? (out.re as Float64Array) : new Float64Array(N * N);
    const im = into ? (out.im as Float64Array) : new Float64Array(N * N);
    for (let k = 0; k < N * N; k++) {
      re[k] = v[2 * k]!;
      im[k] = v[2 * k + 1]!;
    }
    this.readBuf.unmap();
    return { N, re, im };
  }

  private hBuf(zSub: number, pupil = false): GPUBuffer {
    // made for the upload and let go: hBufs is the cache (a CPU copy of every kernel beside it doubled their memory — the iPhone)
    const z = Math.round(zSub * 1000) / 1000;
    return this.hBufs.get(`${pupil ? "HP" : "H"}:${zSub}`, () => this.upload(pupil ? pupilH(this.g, z) : transferH(this.g, z)));
  }

  private phaseBuf(key: string, phi: () => Float32Array, free?: () => void): GPUBuffer {
    return this.phBufs.get(key, () => {
      const p = phi();
      free?.(); // the plan's CPU copy goes once the GPU has its own (remade from the crystal's map if this buffer is ever evicted)
      const w = new Float32Array(p.length);
      const TAU = 2 * Math.PI;
      for (let k = 0; k < p.length; k++) w[k] = p[k]! - TAU * Math.round(p[k]! / TAU);
      return this.upload(w);
    });
  }

  /** half fields nothing reads any more: their 16-bit words kept for the next half trace (one pass's worth and a few over) */
  recycleFields(fields: Field[]): void {
    const seen = new Set<ArrayBufferLike>();
    for (const f of fields)
      for (const a of [f.re, f.im]) {
        if (!Float16 || !(a instanceof Float16) || seen.has(a.buffer) || this.halfPool.length >= 96) continue;
        seen.add(a.buffer);
        this.halfPool.push(new Uint16Array(a.buffer as ArrayBuffer));
      }
  }

  /** an 8-bit crystal's map as its steps k (phase k·2π/256), four to a u32: recovered exactly, whatever precision the map arrived in */
  private phaseBuf8(key: string, phi: () => Float32Array, free?: () => void, steps?: () => Uint8Array): GPUBuffer {
    return this.phBufs.get(`q8:${key}`, () => {
      if (steps) {
        // the crystal keeps its maps as steps (stepMaps): uploaded as they are, no radians made
        const s = steps(), b = this.device.createBuffer({ size: Math.max(16, Math.ceil(s.length / 4) * 4), usage: usage().STORAGE, mappedAtCreation: true });
        new Uint8Array(b.getMappedRange()).set(s);
        b.unmap();
        return b;
      }
      const p = phi();
      free?.();
      const q = new Uint8Array(Math.ceil(p.length / 4) * 4), TAU = 2 * Math.PI;
      for (let k = 0; k < p.length; k++) {
        const t = p[k]! / TAU;
        q[k] = Math.round((t - Math.floor(t)) * 256) & 255;
      }
      const b = this.device.createBuffer({ size: Math.max(16, q.byteLength), usage: usage().STORAGE, mappedAtCreation: true });
      new Uint8Array(b.getMappedRange()).set(q);
      b.unmap();
      return b;
    });
  }

  /** A storage buffer holding a copy of `a`. */
  private upload(a: Float32Array): GPUBuffer {
    const b = this.device.createBuffer({ size: Math.max(16, a.byteLength), usage: usage().STORAGE, mappedAtCreation: true });
    new Float32Array(b.getMappedRange(), 0, a.length).set(a);
    b.unmap();
    return b;
  }

}
