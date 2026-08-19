import { d } from "typegpu";

import type { World } from "./world";
import { makeGpuFunctions } from "./gpu";
import { setupCamera, type CameraProps } from "./camera";

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

  const gpuFunctions = await makeGpuFunctions({
    initialCamera: setupCamera(imageWidth, imageHeight, cameraProps),
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
    gpuFunctions.updateCamera(setupCamera(imageWidth, imageHeight, newProps));
    renderAllPasses();
  }

  renderAllPasses();

  return { render, setCameraProps };
}
