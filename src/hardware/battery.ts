// A battery drawn down by the crystal's work: requests (see energy.ts) and the steady standby draw. Pure bookkeeping, no clock of its own:
// a caller feeds it requests and, for an animation that shows the hardware in slow motion, the animation's clock (steady), so the cell
// drains at the pace the viewer sees. Laser off is hardware off: nothing draws, and time spent off is never billed later.
import type { LightSpec, PowerSpec } from "./energy.js";
import { pulseJ, requestJ, standbyW } from "./energy.js";

/** J: alkaline cells, 2500 mAh and 1000 mAh at 1.5 V — about 13.5 kJ and 5.4 kJ */
export const BATT_AA_J = 2500e-3 * 1.5 * 3600;
export const BATT_AAA_J = 1000e-3 * 1.5 * 3600;
/** a fresh cell's charge, % (a new alkaline reads about here) */
export const BATT_FRESH_PCT = 99;

/** what the steady state needs from the scene each frame */
export interface SteadyInput {
  /** the animation's photon clock, cycles (one photon spacing each) */
  cycles: number;
  laserOn: boolean;
  /** spacings from one pulse to the next on screen (bursts: burstLen + burstGap; else 1) */
  period: number;
  /** how much slower than light the animation runs (the caller's own pacing) */
  sloMo: number;
  power: PowerSpec;
  light: LightSpec;
}

/** the cell; one can be shared by everything on a page that runs a crystal */
export class Battery {
  private slowAt = 0;
  private wallAt = 0;
  private anchored = false;
  private readonly listeners = new Set<(b: Battery) => void>();
  private shown = -1;

  constructor(public usedJ = 0) {}

  /** the charge left, % */
  get pct(): number {
    return Math.max(0, BATT_FRESH_PCT - (100 * this.usedJ) / BATT_AA_J);
  }

  /** draw J off the cell (a flat cell draws nothing more) */
  spend(J: number): void {
    if (!(J > 0) || this.pct <= 0) return;
    this.usedJ += J;
    const shown = Math.floor(this.pct * 100); // the print shows hundredths: tell only when that changes
    if (shown === this.shown) return;
    this.shown = shown;
    for (const f of this.listeners) f(this);
  }

  /** the hardware now drawing on it: what a request costs when the caller does not say */
  spec: { power: PowerSpec; light: LightSpec } | null = null;

  /** one request off the cell — on the given hardware, or `spec` (nothing, before one is set) */
  request(p?: PowerSpec, l?: LightSpec): void {
    const pw = p ?? this.spec?.power, li = l ?? this.spec?.light;
    if (pw && li) this.spend(requestJ(pw, li));
  }

  /** called when the shown hundredth changes; returns the unsubscribe */
  watch(f: (b: Battery) => void): () => void {
    this.listeners.add(f);
    return () => this.listeners.delete(f);
  }

  /**
   * One frame of the steady state: the always-on draw over the real time watched (slowed by sloMo), and a pulse's own cost for each pulse
   * the animation showed since the last charge. The laser off carries the anchors forward, so time spent dark is never billed.
   */
  steady(s: SteadyInput, nowMs = performance.now()): void {
    if (!s.laserOn || !this.anchored || s.cycles < this.slowAt) {
      this.slowAt = s.cycles;
      this.wallAt = nowMs;
      this.anchored = true;
      return;
    }
    const dC = s.cycles - this.slowAt;
    if (dC < 0.02) return; // about every 100 ms at 1×
    const wall = Math.min(5, (nowMs - this.wallAt) / 1000);
    this.slowAt = s.cycles;
    this.wallAt = nowMs;
    this.spend((standbyW(s.power) * wall) / s.sloMo + (dC / Math.max(0.05, s.period)) * pulseJ(s.power, s.light));
  }
}
