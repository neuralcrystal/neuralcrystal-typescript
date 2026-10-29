import { describe, expect, it } from "vitest";
import { BATT_AA_J, BENCH_POWER, Battery, awakeW, requestJ, pulseJ, standbyW } from "../src/index.js";

const quad = { N: 1024, lamUm: 0.617, frac: 0.2, passes: 4 };

describe("the hardware's AA battery", () => {
  it("draws the hardware's powers and a request's energy", () => {
    expect(standbyW(BENCH_POWER)).toBeCloseTo(3.52, 6);
    expect(awakeW(BENCH_POWER)).toBeCloseTo(1.5, 6);
    // four passes of load + pulse + frame at 1.5 W, and four pulses of light
    const act = 4 * (50e-6 + BENCH_POWER.pulseLen + 1 / BENCH_POWER.sensorFps);
    expect(requestJ(BENCH_POWER, quad)).toBeGreaterThan(1.5 * act);
    expect(requestJ(BENCH_POWER, quad)).toBeLessThan(1.5 * act * 1.01);
    expect(pulseJ(BENCH_POWER, quad)).toBeCloseTo(requestJ(BENCH_POWER, quad) - 1.5 * 50e-6, 9);
  });

  it("starts at 99 % fresh and drops by the J drawn", () => {
    const b = new Battery();
    expect(b.pct).toBe(99);
    b.spend(BATT_AA_J / 100);
    expect(b.pct).toBeCloseTo(98, 9);
  });

  it("charges the steady state per pulse on screen, and nothing with the laser off", () => {
    const b = new Battery(), s = { cycles: 0, laserOn: true, period: 0.804, sloMo: 2.998e11 / ((10 * 0.22 * 0.6) / 1), power: BENCH_POWER, light: quad };
    b.steady(s, 0);
    b.steady({ ...s, cycles: 8.04 }, 1000); // ten pulses crossed the hardware
    expect(b.usedJ).toBeCloseTo(10 * pulseJ(BENCH_POWER, quad) + standbyW(BENCH_POWER) / s.sloMo, 9);
    const u = b.usedJ;
    b.steady({ ...s, cycles: 20, laserOn: false }, 2000);
    b.steady({ ...s, cycles: 20 }, 60000); // back on: the dark minute is not billed
    expect(b.usedJ).toBe(u);
  });
});
