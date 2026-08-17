import tgpu, { d } from "typegpu";
import { cos, dot, fract, sin, sqrt } from "typegpu/std";

export const INF = d.f32(3.40282347e+38);
export const PI = d.f32(3.1415927);

export const Interval = d.struct({
  min: d.f32,
  max: d.f32,
});

export const interval = tgpu.fn([d.f32, d.f32], Interval)((min, max) => Interval({ min, max }));
export const surrounds = tgpu.fn([Interval, d.f32], d.bool)((i, num) => i.min < num && num < i.max);

export const Ray = d.struct({
  origin: d.vec3f,
  direction: d.vec3f,
});

export const at = (ray: d.Infer<typeof Ray>, t: number) => {
  'use gpu';
  return ray.origin.add(ray.direction.mul(t));
}

export const HitRecord = d.struct({
  isHit: d.bool,
  position: d.vec3f,
  normal: d.vec3f,
  t: d.f32,
  isFrontFace: d.bool,
});

export const didNotHit = () => {
  'use gpu';
  return HitRecord({
    isHit: false,
    position: d.vec3f(0, 0, 0),
    normal: d.vec3f(0, 0, 1),
    t: 0,
    isFrontFace: false,
  });
}

export const Bounce = d.struct({
  // 0 = didn't bounce, 1 = did.
  // Compared to d.bool, this is both host-shareable and easy to sum.
  didBounce: d.u32,
  ray: Ray,
  uv: d.vec2f, // pixel coords of the originating ray, used as seed for noise
});

export const didNotBounce = () => {
  'use gpu';
  return Bounce({
    didBounce: 0,
    ray: Ray({ origin: d.vec3f(0, 0, 0), direction: d.vec3f(0, 0, 1) }),
    uv: d.vec2f(0, 0),
  });
}

// Jorge Jimenez' "interleaved gradient noise", from https://medium.com/@jcowles/gpu-ray-tracing-in-one-weekend-3e7d874b3b0f
export const noise = tgpu.fn([d.f32, d.f32], d.f32)((i, j) =>
   fract(52.9829189 * fract(i * 0.06711056 + j * 0.00583715)));

// Adapted from https://stackoverflow.com/a/26127012
// i should be in the range [0, 1]
const fibonacciSphere = tgpu.fn([d.f32], d.vec3f)(i => {
  const phi = PI * (sqrt(5.0) - 1.0);
  const y = 1 - i * 2;
  const radius = sqrt(1 - y*y);
  const theta = phi * i;
  const x = cos(theta) * radius;
  const z = sin(theta) * radius;
  return d.vec3f(x, y, z);
});

const randomUnitVector = tgpu.fn([d.f32, d.f32], d.vec3f)((i, j) => fibonacciSphere(noise(i, j)));

export const randomOnHemisphere = tgpu.fn([d.f32, d.f32, d.vec3f], d.vec3f)((i, j, normal) => {
  const onUnitSphere = randomUnitVector(i, j);
  if (dot(onUnitSphere, normal) > 0.0) {
    return onUnitSphere;
  } else {
    return onUnitSphere.mul(-1);
  }
});
