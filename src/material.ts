import tgpu, { d } from "typegpu";
import { dot, normalize, reflect, refract, select } from "typegpu/std";
import { randf } from '@typegpu/noise';

import { Bounce, didNotBounce, isNearZero, Ray } from "./utils";
import { HitRecord } from "./types";

export const MATERIAL_LAMBERTIAN = 0;
export const MATERIAL_METAL = 1;
export const MATERIAL_DIELECTRIC = 2;

export const Lambertian = d.struct({
  albedo: d.vec3f,
});

export const scatterLambertian = tgpu.fn([Lambertian, HitRecord], Bounce)((lambertian, hitRecord) => {
  const perturbation = randf.onUnitSphere();
  const perturbed = hitRecord.normal.add(perturbation);
  const bounceDirection = select(perturbed, hitRecord.normal, isNearZero(perturbed));
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

export const scatterMetal = tgpu.fn([Metal, Ray, HitRecord], Bounce)((metal, ray, hitRecord) => {
  const reflected = reflect(ray.direction, hitRecord.normal).add(randf.onUnitSphere().mul(metal.fuzz));
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

export const Dielectric = d.struct({
  refractionIndex: d.f32,
});

export const scatterDielectric = tgpu.fn([Dielectric, Ray, HitRecord], Bounce)((dielectric, ray, hitRecord) => {
  const ri = hitRecord.isFrontFace ? (1.0/dielectric.refractionIndex) : dielectric.refractionIndex;

  const unitDirection = normalize(ray.direction);

  let direction = refract(unitDirection, normalize(hitRecord.normal), ri);
  if (isNearZero(direction)) {
    direction = reflect(unitDirection, hitRecord.normal);
  }

  const bouncedRay = Ray({ origin: hitRecord.position, direction });

  return Bounce({
    didBounce: 1,
    ray: bouncedRay,
    attenuation: d.vec3f(1, 1, 1),
  });
});
