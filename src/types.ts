import { d } from "typegpu";

export const MaterialReference = d.struct({
  kind: d.u32,
  index: d.u32,
});

export const HitRecord = d.struct({
  isHit: d.bool,
  position: d.vec3f,
  normal: d.vec3f,
  t: d.f32,
  isFrontFace: d.bool,
  material: MaterialReference,
});

export const ImageSize = d.struct({
  width: d.u32,
  height: d.u32,
  numPixels: d.u32,
});
