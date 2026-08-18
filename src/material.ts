import tgpu, { d } from "typegpu";
import { normalize } from "typegpu/std";

import { Bounce, HitRecord, randomUnitVector, Ray } from "./utils";

export const MATERIAL_LAMBERTIAN = 0;
export const MATERIAL_METAL = 1;

export const Lambertian = d.struct({
  albedo: d.vec3f,
});

export const scatterLambertian = tgpu.fn([Lambertian, HitRecord, d.f32, d.f32], Bounce)((lambertian, hitRecord, i, samples) => {
  const bounceDirection = normalize(hitRecord.normal.add(randomUnitVector(i, samples)));
  const bouncedRay = Ray({ origin: hitRecord.position, direction: bounceDirection });

  return Bounce({
    didBounce: 1,
    ray: bouncedRay,
    attenuation: lambertian.albedo,
  });
});

export const Metal = d.struct({
  albedo: d.vec3f,
});