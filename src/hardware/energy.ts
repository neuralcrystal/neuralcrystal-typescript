// Energy per read: what a crystal's answer costs on real hardware. A request (a digit read, a move, a look-ahead read) wakes the
// electronics for the pattern load, the pulse and a frame read, plus the pulse's light; between requests only the standby draw remains.
// The light is sized so each sensor pixel collects enough electrons for a clean read after the crystal's losses.

/** electrons per sensor pixel for a clean read (~3 % shot noise), and the quantum efficiency at 617 nm */
export const SENSOR_E = 1000, SENSOR_QE = 0.6;
/** the pattern load, s: the mirrors settle before the pulse */
const LOAD_S = 50e-6;

/** the hardware around the glass, as power: the electronics, the DMD, the laser pulse and the imager */
export interface PowerSpec {
  /** W drawn by the electronics in all (DMD + controller, the imager, the laser and its driver) */
  elecW: number;
  /** the DMD and its controller stay powered between requests at dmdW (a DLPC900 does not sleep at 20 mW) */
  dmdOn: boolean;
  dmdW: number;
  /** s: the laser pulse */
  pulseLen: number;
  /** the imager's frame rate, frames a second */
  sensorFps: number;
  /** pulses a frame: each pulse carries a 1/ppf share of the frame's energy */
  pulsesPerFrame: number;
}

/** the electronics of neuralcrystal.com's demo hardware (a DLP DMD and controller, a fast imager, a pulsed 617 nm laser) */
export const DEMO_POWER: PowerSpec = { elecW: 5, dmdOn: true, dmdW: 3.5, pulseLen: 9.660508789898141e-6, sensorFps: 12077.294685990339, pulsesPerFrame: 1 };

/** what one request lights and reads: the sensor's pixels, the wavelength, the share of the light that reaches the sensor, the passes */
export interface LightSpec {
  /** the sensor's pixels a side (the grid's N) */
  N: number;
  lamUm: number;
  /** the light reaching the sensor over the light in (0–1) */
  frac: number;
  /** passes a request (a multi-stage crystal: one per stage) */
  passes: number;
}

/** W drawn between requests */
export function standbyW(p: PowerSpec): number {
  return 0.02 + (p.dmdOn ? p.dmdW : 0);
}

/** W that wakes for a request on top of that */
export function awakeW(p: PowerSpec): number {
  return p.dmdOn ? Math.max(0, p.elecW - p.dmdW) : p.elecW;
}

/** J of laser light a pulse, so each sensor pixel collects SENSOR_E electrons after the crystal's losses */
export function pulseLightJ(l: LightSpec): number {
  const hnu = (6.626e-34 * 2.998e8) / (l.lamUm * 1e-6);
  return (l.N * l.N * SENSOR_E * hnu) / SENSOR_QE / Math.max(1e-6, l.frac);
}

/** J: one request — a load, a pulse and a frame read a pass, and every pass's light */
export function requestJ(p: PowerSpec, l: LightSpec): number {
  const activeS = l.passes * (LOAD_S + p.pulseLen + 1 / p.sensorFps);
  return awakeW(p) * activeS + pulseLightJ(l) * l.passes;
}

/** J: one pulse's own cost in the steady state — a request without the pattern load, a 1/ppf share with several pulses a frame */
export function pulseJ(p: PowerSpec, l: LightSpec): number {
  return Math.max(0, requestJ(p, l) - awakeW(p) * LOAD_S) / Math.max(1, p.pulsesPerFrame);
}

/** @deprecated the old name of DEMO_POWER */
export const BENCH_POWER = DEMO_POWER;
