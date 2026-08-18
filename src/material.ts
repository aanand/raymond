import { d } from "typegpu";

export const MATERIAL_LAMBERTIAN = 0;
export const MATERIAL_METAL = 1;

export const Lambertian = d.struct({
  albedo: d.vec3f,
});

export const Metal = d.struct({
  albedo: d.vec3f,
});