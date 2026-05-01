// Minimal flat-buffer tensor helpers for the PoC.
// 4D tensors are stored as Float32Array with explicit shape; we never need
// general broadcasting so the API is intentionally narrow.

export interface T4 {
  data: Float32Array;
  shape: readonly [number, number, number, number];
}

export function numel(shape: readonly number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

export function zeros(shape: T4["shape"]): T4 {
  return { data: new Float32Array(numel(shape)), shape };
}

export function copy(t: T4): T4 {
  return { data: new Float32Array(t.data), shape: t.shape };
}

export function assertSameShape(a: T4, b: T4, ctx = "shape mismatch"): void {
  if (a.shape.length !== b.shape.length) {
    throw new Error(`${ctx}: rank ${a.shape.length} vs ${b.shape.length}`);
  }
  for (let i = 0; i < a.shape.length; i++) {
    if (a.shape[i] !== b.shape[i]) {
      throw new Error(`${ctx}: ${a.shape.join("x")} vs ${b.shape.join("x")}`);
    }
  }
}

// Element-wise: out[i] = a*x[i] + b*y[i]
export function axpby(a: number, x: T4, b: number, y: T4): T4 {
  assertSameShape(x, y, "axpby");
  const out = new Float32Array(x.data.length);
  for (let i = 0; i < out.length; i++) {
    out[i] = a * x.data[i]! + b * y.data[i]!;
  }
  return { data: out, shape: x.shape };
}

// Element-wise: out[i] = (a*x[i] - y[i]) / b   (used by _predict_eps_from_xstart)
export function epsFromXstart(
  a: number,
  x: T4,
  y: T4,
  b: number,
): T4 {
  assertSameShape(x, y, "epsFromXstart");
  const out = new Float32Array(x.data.length);
  for (let i = 0; i < out.length; i++) {
    out[i] = (a * x.data[i]! - y.data[i]!) / b;
  }
  return { data: out, shape: x.shape };
}

// Mean absolute error between two tensors of the same shape.
export function mae(a: T4, b: T4): number {
  assertSameShape(a, b, "mae");
  let sum = 0;
  for (let i = 0; i < a.data.length; i++) {
    sum += Math.abs(a.data[i]! - b.data[i]!);
  }
  return sum / a.data.length;
}

// Slice along the last axis. Negative `start` counts from the end.
// Used by AR loop: prefix = sample[..., -context_len:]
export function sliceLastAxis(t: T4, start: number, end?: number): T4 {
  const T = t.shape[3];
  const s = start < 0 ? T + start : start;
  const e = end === undefined ? T : end < 0 ? T + end : end;
  if (s < 0 || e > T || s >= e) {
    throw new Error(`sliceLastAxis: bad range [${s},${e}) for T=${T}`);
  }
  const newT = e - s;
  const [B, J, F] = [t.shape[0], t.shape[1], t.shape[2]];
  const out = new Float32Array(B * J * F * newT);
  // Source strides
  const strideF = T;
  const strideJ = F * T;
  const strideB = J * F * T;
  // Dst strides
  const dStrideF = newT;
  const dStrideJ = F * newT;
  const dStrideB = J * F * newT;
  for (let b = 0; b < B; b++) {
    for (let j = 0; j < J; j++) {
      for (let f = 0; f < F; f++) {
        const srcOff = b * strideB + j * strideJ + f * strideF + s;
        const dstOff = b * dStrideB + j * dStrideJ + f * dStrideF;
        for (let k = 0; k < newT; k++) {
          out[dstOff + k] = t.data[srcOff + k]!;
        }
      }
    }
  }
  return { data: out, shape: [B, J, F, newT] };
}

// Concat two T4 along the last axis.
export function concatLastAxis(a: T4, b: T4): T4 {
  if (a.shape[0] !== b.shape[0] || a.shape[1] !== b.shape[1] || a.shape[2] !== b.shape[2]) {
    throw new Error(
      `concatLastAxis: leading dims must match (${a.shape} vs ${b.shape})`,
    );
  }
  const [B, J, F] = [a.shape[0], a.shape[1], a.shape[2]];
  const Ta = a.shape[3];
  const Tb = b.shape[3];
  const T = Ta + Tb;
  const out = new Float32Array(B * J * F * T);
  for (let bi = 0; bi < B; bi++) {
    for (let j = 0; j < J; j++) {
      for (let f = 0; f < F; f++) {
        const dstOff = bi * J * F * T + j * F * T + f * T;
        const aOff = bi * J * F * Ta + j * F * Ta + f * Ta;
        const bOff = bi * J * F * Tb + j * F * Tb + f * Tb;
        for (let k = 0; k < Ta; k++) out[dstOff + k] = a.data[aOff + k]!;
        for (let k = 0; k < Tb; k++) out[dstOff + Ta + k] = b.data[bOff + k]!;
      }
    }
  }
  return { data: out, shape: [B, J, F, T] };
}

// Box-Muller standard normal sampler with seedable PRNG.
// Uses splitmix64-derived 32-bit state — adequate for PoC; matches no Python RNG.
export class SeededRng {
  private state: number;
  constructor(seed: number) {
    // Avoid 0 state.
    this.state = (seed >>> 0) || 0x9e3779b9;
  }
  // xorshift32
  private next32(): number {
    let x = this.state;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.state = x >>> 0;
    return this.state;
  }
  private nextFloat(): number {
    // (0, 1) — avoid zero so log() in Box-Muller is safe.
    return (this.next32() + 1) / 4294967297;
  }
  randn(out: Float32Array): void {
    for (let i = 0; i < out.length; i += 2) {
      const u1 = this.nextFloat();
      const u2 = this.nextFloat();
      const r = Math.sqrt(-2 * Math.log(u1));
      const theta = 2 * Math.PI * u2;
      out[i] = r * Math.cos(theta);
      if (i + 1 < out.length) out[i + 1] = r * Math.sin(theta);
    }
  }
}

export function randnT4(shape: T4["shape"], rng: SeededRng): T4 {
  const t = zeros(shape);
  rng.randn(t.data);
  return t;
}
