import tgpu, { d } from "typegpu";
import { abs, cos, floor, fract, max, min, sin, sqrt } from "typegpu/std";
import { MaterialReference } from "./types";
import { HitRecord } from "./types";

export const INF = d.f32(3.40282347e+38);
export const PI = d.f32(3.1415927);

export const Interval = d.struct({
  min: d.f32,
  max: d.f32,
});

export const interval = tgpu.fn([d.f32, d.f32], Interval)((min, max) => Interval({ min, max }));
export const surrounds = tgpu.fn([Interval, d.f32], d.bool)((i, num) => i.min < num && num < i.max);

export const combineIntervals = tgpu.fn([Interval, Interval], Interval)((a, b) =>
  interval(min(a.min, b.min), max(a.max, b.max)));

export const Ray = d.struct({
  origin: d.vec3f,
  direction: d.vec3f,
});

export const at = (ray: d.Infer<typeof Ray>, t: number) => {
  'use gpu';
  return ray.origin.add(ray.direction.mul(t));
}

export const didNotHit = () => {
  'use gpu';
  return HitRecord({
    isHit: false,
    position: d.vec3f(0, 0, 0),
    normal: d.vec3f(0, 0, 1),
    t: 0,
    isFrontFace: false,
    material: MaterialReference({ kind: 0, index: 0 }),
  });
}

export const Bounce = d.struct({
  // 0 = didn't bounce, 1 = did.
  // Compared to d.bool, this is both host-shareable and easy to sum.
  didBounce: d.u32,
  ray: Ray,
  attenuation: d.vec3f,
});

export const didNotBounce = () => {
  'use gpu';
  return Bounce({
    didBounce: 0,
    ray: Ray({ origin: d.vec3f(0, 0, 0), direction: d.vec3f(0, 0, 1) }),
    attenuation: d.vec3f(1, 1, 1),
  });
}

// Jorge Jimenez' "interleaved gradient noise", from https://medium.com/@jcowles/gpu-ray-tracing-in-one-weekend-3e7d874b3b0f
export const noise = tgpu.fn([d.f32, d.f32], d.f32)((i, j) =>
   fract(52.9829189 * fract(i * 0.06711056 + j * 0.00583715)));

// Adapted from https://stackoverflow.com/a/26127012
// i should be in the range [0, 1]
const fibonacciSphere = tgpu.fn([d.f32, d.f32], d.vec3f)`(i: f32, samples: f32) -> vec3f {
  let phi = 3.1415927 * (sqrt(5.0) - 1.0);
  let y = 1.0 - (i / (samples - 1.0)) * 2.0;
  let radius = sqrt(1.0 - y*y);
  let theta = phi * i;
  let x = cos(theta) * radius;
  let z = sin(theta) * radius;
  return vec3f(x, y, z);
}`;

export const randomUnitVector = tgpu.fn([d.f32], d.vec3f)((randomFloat) =>
  fibonacciSphere(floor(randomFloat * 1000), 1000));

export const randomInUnitDisk = tgpu.fn([d.f32, d.f32], d.vec3f)((randomFloat1, randomFloat2) => {
  const r = sqrt(randomFloat1);
  const theta = randomFloat2 * 2 * PI;
  return d.vec3f(r * cos(theta), r * sin(theta), 0);
});

export const isNearZero = tgpu.fn([d.vec3f], d.bool)(vec => {
  const s = 1e-8;
  return abs(vec.x) < s && abs(vec.y) < s && abs(vec.z) < s;
});

// Useful for debugging
export const isInvalidFloat = tgpu.fn([d.f32], d.bool)(/* wgsl */`(val) -> {
  let bits = bitcast<u32>(val);
  return (bits & 0x7fffffffu) >= 0x7f800000u;
}`);
