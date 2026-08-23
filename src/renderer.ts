import { d } from "typegpu";

import type { World } from "./world";
import { makeGpuFunctions } from "./gpu";
import { type CameraProps } from "./camera";

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
    initialCameraProps: cameraProps,
    imageWidth,
    imageHeight,
    world,
  });

  const render = async (canvas: HTMLCanvasElement) => {
    gpuFunctions.renderOnePass(samplesPerPixel, samplesPerPass, maxBounceDepth);

    const imageData = await gpuFunctions.getPixelData();

    canvas.width = imageWidth;
    canvas.height = imageHeight;
    canvas.getContext('2d')!.putImageData(imageData, 0, 0);
  }

  return {
    render,
    setCameraProps: gpuFunctions.updateCameraProps,
    setIsLoResMode: gpuFunctions.setIsLoResMode,
    getRenderTime: gpuFunctions.getRenderTime,
  };
}
