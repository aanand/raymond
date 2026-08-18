import tgpu, { d } from "typegpu";
import { normalize, reflect } from "typegpu/std";

import { Bounce, HitRecord, randomUnitVector, Ray } from "./utils";

export const MATERIAL_LAMBERTIAN = 0;
export const MATERIAL_METAL = 1;

export const Lambertian = d.struct({
  albedo: d.vec3f,
});

export const scatterLambertian = tgpu.fn([Lambertian, HitRecord, d.f32], Bounce)((lambertian, hitRecord, randomFloat) => {
  const bounceDirection = normalize(hitRecord.normal.add(randomUnitVector(randomFloat)));
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

export const scatterMetal = tgpu.fn([Metal, Ray, HitRecord], Bounce)((metal, ray, hitRecord) => {
  const reflected = reflect(ray.direction, hitRecord.normal);
  const bouncedRay = Ray({ origin: hitRecord.position, direction: normalize(reflected) }); // normalize might not be necessary here?

  return Bounce({
    didBounce: 1,
    ray: bouncedRay,
    attenuation: metal.albedo,
  });
});