import tgpu, { d } from "typegpu";
import { combineIntervals, Interval, interval, Ray } from "./utils";

export const AABB = d.arrayOf(Interval, 3);

export const aabb = tgpu.fn([d.vec3f, d.vec3f], AABB)((a, b) =>
  [
    a.x <= b.x ? interval(a.x, b.x) : interval(b.x, a.x),
    a.y <= b.y ? interval(a.y, b.y) : interval(b.y, a.y),
    a.z <= b.z ? interval(a.z, b.z) : interval(b.z, a.z),
  ]
);

export const combineAABBs = tgpu.fn([AABB, AABB], AABB)((a, b) => [
  combineIntervals(a[0], b[0]),
  combineIntervals(a[1], b[1]),
  combineIntervals(a[2], b[2]),
]);

export const hitAABB = tgpu.fn([AABB, Ray, Interval], d.bool)((bb, ray, t) => {
  let tMin = t.min;
  let tMax = t.max;

  for (let axis = 0; axis < 3; axis++) {
    const ax = bb[axis];
    const adInv = 1.0 / ray.direction[axis];

    const t0 = (ax.min - ray.origin[axis]) * adInv;
    const t1 = (ax.max - ray.origin[axis]) * adInv;

    if (t0 < t1) {
      if (t0 > tMin) tMin = t0;
      if (t1 < tMax) tMax = t1;
    } else {
      if (t1 > tMin) tMin = t1;
      if (t0 < tMax) tMax = t0;
    }
  }

  return tMin < tMax;
});