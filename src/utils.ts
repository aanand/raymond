import tgpu, { d } from "typegpu";
import { fract } from "typegpu/std";

export const INF = d.f32(3.40282347e+38);

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

// Jorge Jimenez' "interleaved gradient noise", from https://medium.com/@jcowles/gpu-ray-tracing-in-one-weekend-3e7d874b3b0f
export const noise = tgpu.fn([d.f32, d.f32], d.f32)((x, y) =>
   fract(52.9829189 * fract(x * 0.06711056 + y * 0.00583715)));
