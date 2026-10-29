// Node 20 has no Float16Array: for the GPU test only, one that decodes the half floats into a Float32Array (the engine's half traces
// build theirs from the raw 16-bit words, which is all this has to stand in for). Imported before the engine, which reads it at load.
const g = globalThis as { Float16Array?: unknown };
if (!g.Float16Array) {
  g.Float16Array = function (buffer: ArrayBuffer): Float32Array {
    const u = new Uint16Array(buffer), f = new Float32Array(u.length);
    for (let i = 0; i < u.length; i++) {
      const h = u[i]!, s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
      f[i] = s * (e === 0 ? m * 2 ** -24 : e === 31 ? (m ? NaN : Infinity) : (1 + m / 1024) * 2 ** (e - 15));
    }
    return f;
  };
}
