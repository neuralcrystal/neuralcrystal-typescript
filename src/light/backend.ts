// What a light backend does: run one section's plan on a field. The CPU backend is the reference; the WebGPU backend runs the
// same plans on the GPU. Everything around the light (the DMD, the camera, the FPGA controller's hand-off) is backend-independent.
import type { SectionPlan } from "../physics/plan.js";
import type { Field } from "./field.js";

/** a square of the grid (x0, y0, n × n): a trace read back over the window only, not the whole grid */
export interface Crop {
  x0: number;
  y0: number;
  n: number;
}

export interface LightBackend {
  readonly name: string;
  /** the field at the section's exit face; the input is not modified */
  /** out: a field to write the exit into instead of a new one (a phone's look-ahead reads — CrystalEngine.reuseReads); a backend may ignore it */
  runSection(input: Field, plan: SectionPlan, out?: Field): Promise<Field>;
  /** optional: several plans in turn, the field after each (one submission, one readback on the GPU) */
  /** into: fields to write the stages into (a recycled trace's), in order */
  /** half: every stage but the last read back in half precision (Float16Array) — the last one, the exit the next pass is made from, stays single */
  /** lastInto: a field (single precision, the crop's size) to read a half trace's last stage, its exit, into instead of a new one */
  runStages?(input: Field, plans: SectionPlan[], into?: Field[], crop?: Crop, half?: boolean, lastInto?: Field): Promise<Field[]>;
  /** optional: fields read back by runStages that nothing reads any more — kept to read the next ones into (CrystalEngine.recycleFields) */
  recycleFields?(fields: Field[]): void;
  /** free GPU buffers, caches */
  dispose(): void;
}
