import { d } from "typegpu";
import { cross, length, normalize, radians, tan } from "typegpu/std";

import type { World } from "./world";
import { makeGpuFunctions, type Camera } from "./gpuFunctions";

type CameraProps = {
  vfov: number,
  lookFrom: d.v3f,
  lookAt: d.v3f,
  vup: d.v3f,
}

export const createRenderer = async ({
  aspectRatio,
  imageWidth,
  
  cameraProps,

  samplesPerPixel,
  samplesPerPass,
  maxBounceDepth,
  
  world,
}: {
  aspectRatio: number,
  imageWidth: number,

  cameraProps: CameraProps,

  samplesPerPixel: number,
  samplesPerPass: number,
  maxBounceDepth: number,

  world: World,
}) => {
  const imageHeight = d.u32(Math.max(1, Math.floor(imageWidth/aspectRatio)));

  const setupCamera = (): Camera => {
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

  const gpuFunctions = await makeGpuFunctions({
    initialCamera: setupCamera(),
    imageWidth,
    imageHeight,
    world,
  });

  let numSamplesTaken = 0;
  let nextPassTimeout = 0;

  function renderNextPass() {
    if (numSamplesTaken >= samplesPerPixel) {
      return;
    }

    gpuFunctions.renderOnePass(samplesPerPass, maxBounceDepth);
    numSamplesTaken += samplesPerPass;

    nextPassTimeout = setTimeout(renderNextPass, 0);
  }

  function renderAllPasses() {
    numSamplesTaken = 0;

    if (nextPassTimeout) {
      clearTimeout(nextPassTimeout);
      nextPassTimeout = 0;
    }

    gpuFunctions.resetState();
    renderNextPass();
  }

  const render = async (canvas: HTMLCanvasElement) => {
    const imageData = await gpuFunctions.getPixelData();

    canvas.width = imageWidth;
    canvas.height = imageHeight;
    canvas.getContext('2d')!.putImageData(imageData, 0, 0);
  }

  const setCameraProps = (newProps: CameraProps) => {
    cameraProps = newProps;
    gpuFunctions.updateCamera(setupCamera());
    renderAllPasses();
  }

  renderAllPasses();

  return { render, setCameraProps };
}
