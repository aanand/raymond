import tgpu, { d } from "typegpu";
import { radians, tan, normalize, cross } from "typegpu/std";
import { ImageSize } from "./types";

// What the application provides
export const CameraInput = d.struct({
  vfov: d.f32,
  lookFrom: d.vec3f,
  lookAt: d.vec3f,
  vup: d.vec3f,

  defocusAngle: d.f32,
  focusDistance: d.f32,
});

export type CameraProps = d.Infer<typeof CameraInput>;

// What the GPU code uses
export const CameraStruct = d.struct({
  center: d.vec3f,

  pixel00Loc: d.vec3f,
  pixelDeltaU: d.vec3f,
  pixelDeltaV: d.vec3f,

  defocusAngle: d.f32,
  defocusDiskU: d.vec3f,
  defocusDiskV: d.vec3f,
});

export type Camera = d.Infer<typeof CameraStruct>;

export const setupCamera = tgpu.fn([ImageSize, CameraInput], CameraStruct)((
  size,
  cameraProps,
): Camera => {
  const center = cameraProps.lookFrom;
  const theta = radians(cameraProps.vfov);
  const h = tan(theta/2.0);
  const viewportHeight = 2.0 * h * cameraProps.focusDistance;
  const viewportWidth = d.f32(viewportHeight * (size.width/size.height));

  const w = normalize(cameraProps.lookFrom.sub(cameraProps.lookAt));
  const u = normalize(cross(cameraProps.vup, w));
  const v = cross(w, u);

  const viewportU = u.mul(viewportWidth);
  const viewportV = v.mul(-viewportHeight);

  const pixelDeltaU = viewportU.div(size.width);
  const pixelDeltaV = viewportV.div(size.height);

  const viewportUpperLeft = center
    .sub(w.mul(cameraProps.focusDistance))
    .sub(viewportU.div(2))
    .sub(viewportV.div(2));

  const pixel00Loc = viewportUpperLeft.add(pixelDeltaU.add(pixelDeltaV).mul(0.5));

  const defocusAngle = cameraProps.defocusAngle;
  const defocusRadius = cameraProps.focusDistance * tan(radians(defocusAngle / 2));
  const defocusDiskU = u.mul(defocusRadius);
  const defocusDiskV = v.mul(defocusRadius);

  return {
    center,

    pixel00Loc,
    pixelDeltaU,
    pixelDeltaV,

    defocusAngle,
    defocusDiskU,
    defocusDiskV,
  };
});