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

  let numSamplesTaken = 0;

  const render = async (canvas: HTMLCanvasElement) => {
    gpuFunctions.renderOnePass(samplesPerPixel, samplesPerPass, maxBounceDepth);
    numSamplesTaken += samplesPerPass;

    const imageData = await gpuFunctions.getPixelData();

    canvas.width = imageWidth;
    canvas.height = imageHeight;
    canvas.getContext('2d')!.putImageData(imageData, 0, 0);
  }

  const setCameraProps = (newProps: CameraProps) => {
    gpuFunctions.updateCameraProps(newProps);
  }

  const setIsLoResMode = (enabled: boolean) => {
    gpuFunctions.setIsLoResMode(enabled);
  };

  return {
    render,
    setCameraProps,
    setIsLoResMode: setIsLoResMode,
    getRenderTime: gpuFunctions.getRenderTime,
  };
}
