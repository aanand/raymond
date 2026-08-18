import tgpu, { d } from "typegpu";
import { dot, min, normalize, pow, reflect, refract, sqrt } from "typegpu/std";

import { Bounce, didNotBounce, HitRecord, randomUnitVector, Ray } from "./utils";

export const MATERIAL_LAMBERTIAN = 0;
export const MATERIAL_METAL = 1;
export const MATERIAL_DIELECTRIC = 2;

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

export const Dielectric = d.struct({
  refractionIndex: d.f32,
});

export const scatterDielectric = tgpu.fn([Dielectric, Ray, HitRecord, d.f32], Bounce)((dielectric, ray, hitRecord, randomFloat) => {
  const ri = hitRecord.isFrontFace ? (1.0/dielectric.refractionIndex) : dielectric.refractionIndex;

  const unitDirection = normalize(ray.direction);
  const cosTheta = min(dot(unitDirection.mul(-1.0), hitRecord.normal), 1.0);
  const sinTheta = sqrt(1.0 - cosTheta*cosTheta);

  const cannotRefract = ri * sinTheta > 1.0;

  const direction = (cannotRefract || reflectance(cosTheta, ri) > randomFloat) ?
    reflect(unitDirection, hitRecord.normal) :
    refract(unitDirection, hitRecord.normal, ri);

  const bouncedRay = Ray({ origin: hitRecord.position, direction });

  return Bounce({
    didBounce: 1,
    ray: bouncedRay,
    attenuation: d.vec3f(1, 1, 1),
  });
});

const reflectance = tgpu.fn([d.f32, d.f32], d.f32)((cosine, refractionIndex) => {
  const r0 = pow((1.0 - refractionIndex) / (1.0 + refractionIndex), 2);
  return r0 + (1.0 - r0) * pow(1 - cosine, 5);
});