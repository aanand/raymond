import tgpu, { d } from "typegpu";
import { dot, length, sqrt, select } from "typegpu/std";
import { Ray, Interval, didNotHit, surrounds, at } from "./utils";
import { HitRecord } from "./types";
import { MaterialReference } from "./types";

export const Sphere = d.struct({
  center: d.vec3f,
  radius: d.f32,
  material: MaterialReference,
});

export const hitSphere = tgpu.fn([Sphere, Ray, Interval], HitRecord)((sphere, ray, rayT) => {
  const oc = sphere.center.sub(ray.origin);
  const a = length(ray.direction) ** 2;
  const h = dot(ray.direction, oc);
  const c = length(oc) ** 2 - (sphere.radius * sphere.radius);
  const discriminant = h*h - a*c;

  if (discriminant < 0) {
    return didNotHit();
  }

  const sqrtd = sqrt(discriminant);

  // Find the nearest root that lies in the acceptable range.
  let root = (h - sqrtd) / a;
  if (!surrounds(rayT, root)) {
    root = (h + sqrtd) / a;
    if (!surrounds(rayT, root)) {
      return didNotHit();
    }
  }

  const t = root;
  const position = at(ray, t);
  const outwardNormal = position.sub(sphere.center).div(sphere.radius);
  const isFrontFace = dot(ray.direction, outwardNormal) < 0;
  // const normal = isFrontFace ? outwardNormal : outwardNormal.mul(-1);
  const normal = select(outwardNormal.mul(-1), outwardNormal, isFrontFace);

  return HitRecord({
    isHit: true,
    position,
    normal,
    t,
    isFrontFace,
    material: sphere.material,
  });
});