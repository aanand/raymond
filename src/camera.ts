import { d } from "typegpu";
import { length, radians, tan, normalize, cross } from "typegpu/std";

// What the application provides
export type CameraProps = {
  vfov: number,
  lookFrom: d.v3f,
  lookAt: d.v3f,
  vup: d.v3f,
}

// What the GPU code uses
export const CameraStruct = d.struct({
  center: d.vec3f,
  pixel00Loc: d.vec3f,
  pixelDeltaU: d.vec3f,
  pixelDeltaV: d.vec3f,
});

export type Camera = d.Infer<typeof CameraStruct>;

export const setupCamera = (
  imageWidth: number,
  imageHeight: number,
  cameraProps: CameraProps
): Camera => {
  const center = cameraProps.lookFrom;
  const focalLength = length(cameraProps.lookFrom.sub(cameraProps.lookAt));
  const theta = radians(cameraProps.vfov);
  const h = tan(theta/2.0);
  const viewportHeight = 2.0 * h * focalLength;
  const viewportWidth = d.f32(viewportHeight * (imageWidth/imageHeight));

  const w = normalize(cameraProps.lookFrom.sub(cameraProps.lookAt));
  const u = normalize(cross(cameraProps.vup, w));
  const v = cross(w, u);

  const viewportU = u.mul(viewportWidth);
  const viewportV = v.mul(-viewportHeight);

  const pixelDeltaU = viewportU.div(imageWidth);
  const pixelDeltaV = viewportV.div(imageHeight);

  const viewportUpperLeft = center
    .sub(w.mul(focalLength))
    .sub(viewportU.div(2))
    .sub(viewportV.div(2));

  const pixel00Loc = viewportUpperLeft.add(pixelDeltaU.add(pixelDeltaV).mul(0.5));

  return {
    center,
    pixel00Loc,
    pixelDeltaU,
    pixelDeltaV,
  };
}