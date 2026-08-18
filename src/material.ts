import tgpu, { d } from "typegpu";
import { dot, normalize, reflect } from "typegpu/std";

import { Bounce, didNotBounce, HitRecord, randomUnitVector, Ray } from "./utils";

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
  fuzz: d.f32,
});

export const scatterMetal = tgpu.fn([Metal, Ray, HitRecord, d.f32], Bounce)((metal, ray, hitRecord, randomFloat) => {
  const reflected = reflect(ray.direction, hitRecord.normal).add(randomUnitVector(randomFloat).mul(metal.fuzz));
  const bouncedRay = Ray({ origin: hitRecord.position, direction: normalize(reflected) }); // normalize might not be necessary here?

  if (dot(bouncedRay.direction, hitRecord.normal) <= 0) {
    return didNotBounce();
  }

  return Bounce({
    didBounce: 1,
    ray: bouncedRay,
    attenuation: metal.albedo,
  });
});