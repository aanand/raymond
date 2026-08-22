import tgpu, { d } from "typegpu";
import { pack4x8unorm, sqrt } from "typegpu/std";
import { randf } from "@typegpu/noise";

import { Ray } from "./utils";
import { buildRayTraceFunction, RayTraceResult } from "./rayTrace";
import type { World } from "./world";
import { CameraStruct, setupCamera, type CameraProps } from "./camera";
import { ImageSize } from "./types";

export const makeGpuFunctions = async ({
  initialCameraProps,
  imageWidth,
  imageHeight,
  world,
}: {
  initialCameraProps: CameraProps,
  imageWidth: number,
  imageHeight: number,
  world: World,
}) => {
  const numPixels = imageWidth * imageHeight;
  const sizeList = getSizes(imageWidth, imageHeight, 64);

  let sizeIndex = 0;
  let cameraProps = initialCameraProps;
  let isLoResMode = false;

  const State = d.struct({
    size: ImageSize,
    camera: CameraStruct,
    sampleIndex: d.u32,
    accumulatedSamples: d.arrayOf(d.vec4f, numPixels),
  });

  const initialState = () => ({
    size: sizeList[sizeIndex],
    camera: setupCamera(sizeList[sizeIndex], cameraProps),
    sampleIndex: 0,
    accumulatedSamples: new Float32Array(numPixels * 4),
  });

  const rayTrace = buildRayTraceFunction(world);

  const root = await tgpu.init({
    device: {
      optionalFeatures: [
        'timestamp-query',
      ],
    },
  });

  const state = root.createMutable(State, initialState());

  const defocusDiskSample = tgpu.fn([], d.vec3f)(() => {
    const p2d = randf.inUnitCircle();
    const p = d.vec3f(p2d.x, p2d.y, 0);

    return state.$.camera.center
      .add(state.$.camera.defocusDiskU.mul(p))
      .add(state.$.camera.defocusDiskV.mul(p));
  });

  const getPixelCoords = tgpu.fn([d.u32], d.vec2u)((pixelIndex) =>
    d.vec2u(
      d.u32(pixelIndex % state.$.size.width),
      d.u32(pixelIndex / state.$.size.width),
    ));

  const fireInitialRay = tgpu.fn([d.vec2u], RayTraceResult)((xy) => {
    'use gpu';

    const offsetX = randf.sample() - 0.5;
    const offsetY = randf.sample() - 0.5;

    const pixelCenter = state.$.camera.pixel00Loc
      .add(state.$.camera.pixelDeltaU.mul(d.f32(xy.x) + offsetX))
      .add(state.$.camera.pixelDeltaV.mul(d.f32(xy.y) + offsetY));

    const origin = defocusDiskSample();
    const ray = Ray({
      origin: origin,
      direction: pixelCenter.sub(origin),
    });

    return rayTrace(ray);
  });

  const processBounces = tgpu.fn([RayTraceResult, d.u32], RayTraceResult)((rayTraceResult, numBounces) => {
    'use gpu';

    let result = RayTraceResult(rayTraceResult);

    for (let i = d.u32(0); i < numBounces; i++) {
      const currentColor = result.color;
      const bounce = result.bounce;
      const bounceResult = rayTrace(bounce.ray);

      result = RayTraceResult({
        color: d.vec3f(
          currentColor.r * (bounce.didBounce ? bounceResult.color.r : 1.0),
          currentColor.g * (bounce.didBounce ? bounceResult.color.g : 1.0),
          currentColor.b * (bounce.didBounce ? bounceResult.color.b : 1.0),
        ),

        bounce: bounceResult.bounce,
      });
    }

    return result;
  });

  const sample = tgpu.fn([d.u32, d.u32])((pixelIndex, numBounces) => {
    'use gpu';

    const xy = getPixelCoords(pixelIndex);

    const initialResult = fireInitialRay(xy);
    const bouncedResult = processBounces(initialResult, numBounces);

    const pixelWidth = d.f32(imageWidth) / d.f32(state.$.size.width);
    const pixelHeight = d.f32(imageHeight) / d.f32(state.$.size.height);

    const fromImagePixel = d.vec2u(d.u32(d.f32(xy.x) * pixelWidth), d.u32(d.f32(xy.y) * pixelHeight));
    const toImagePixel = d.vec2u(d.u32(d.f32(xy.x + 1) * pixelWidth), d.u32(d.f32(xy.y + 1) * pixelHeight));

    // const debug = xy.x === d.u32(state.$.size.width/2) && xy.y === d.u32(state.$.size.height/2);
    // const debug = pixelIndex < 2;
    const debug = false;
    if (debug) {
      console.log('[pixelIndex=%s] xy=%s',
        pixelIndex, xy);
      console.log('[pixelIndex=%s] fromImagePixel=%s, toImagePixel=%s',
        pixelIndex, fromImagePixel, toImagePixel);
    }

    for (let imageY = fromImagePixel.y; imageY < toImagePixel.y; imageY++) {
      for (let imageX = fromImagePixel.x; imageX < toImagePixel.x; imageX++) {
        const imagePixelIndex = imageY * imageWidth + imageX;        

        if (debug) {
          console.log(
            '[pixelIndex=%s] imageX=%s, imageY=%s, imagePixelIndex=%s',
            pixelIndex, imageX, imageY, imagePixelIndex);
        }

        state.$.accumulatedSamples[imagePixelIndex] = state.$.accumulatedSamples[imagePixelIndex].add(d.vec4f(
          bouncedResult.color.r,
          bouncedResult.color.g,
          bouncedResult.color.b,
          1
        ));
      }
    }
  });

  const linearToGamma = tgpu.fn([d.f32], d.f32)(linear => {
    if (linear > 0) {
      return sqrt(linear);
    }

    return 0;
  });

  const pixelBuffer = root.createMutable(d.arrayOf(d.u32, numPixels));

  const writePixels = root.createGuardedComputePipeline((pixelIndex) => {
    'use gpu';
    const accumulation = state.$.accumulatedSamples[pixelIndex];
    pixelBuffer.$[pixelIndex] = pack4x8unorm(d.vec4f(
      linearToGamma(accumulation[0] / accumulation[3]),
      linearToGamma(accumulation[1] / accumulation[3]),
      linearToGamma(accumulation[2] / accumulation[3]),
      1.0
    ));
  });

  const passConfig = root.createMutable(d.struct({
    samplesPerPixel: d.u32,
    samplesPerPass: d.u32,
    numBounces: d.u32,
  }));

  const renderTimes: bigint[] = [];
  const drawTimes: number[] = [];
  const maxTimingSamples = 30;

  const accumulate = root.createGuardedComputePipeline((pixelIndex) => {
    'use gpu';

    randf.seed2(
      d.vec2f(
        (d.f32(pixelIndex) / d.f32(numPixels)) * 2000 - 1000,
        (d.f32(state.$.sampleIndex) / d.f32(passConfig.$.samplesPerPixel)) * 2000 - 1000));

    for (let s = 0; s < d.i32(passConfig.$.samplesPerPass); s++) {
      sample(pixelIndex, passConfig.$.numBounces);
      state.$.sampleIndex++;
    }
  })
  .withPerformanceCallback((start, end) => {
    renderTimes.unshift((end - start) / BigInt(1000000));
    renderTimes.splice(maxTimingSamples);
  });

  function renderOnePass(samplesPerPixel: number, samplesPerPass: number, numBounces: number) {
    if (isLoResMode) {
      sizeIndex = 0;
    }

    const size = sizeList[sizeIndex];

    state.patch({
      size: size,
      camera: setupCamera(size, cameraProps),
    });

    passConfig.write({ samplesPerPixel, samplesPerPass, numBounces })
    accumulate.dispatchThreads(size.numPixels);
    writePixels.dispatchThreads(numPixels);

    if (!isLoResMode) {
      sizeIndex = Math.min(sizeIndex + 1, sizeList.length - 1);
    }
  };

  return {
    renderOnePass,

    updateCameraProps(newValue: CameraProps) {
      cameraProps = newValue;
      sizeIndex = 0;
    },

    isLoResMode() {
      return isLoResMode;
    },

    setIsLoResMode(newValue: boolean) {
      isLoResMode = newValue;
    },

    resetState() {
      state.write(initialState());
    },

    async getPixelData() {
      const startTime = new Date().getTime();

      const value = await pixelBuffer.read();
      const imageData = new ImageData(new Uint8ClampedArray(new Uint32Array(value).buffer), imageWidth, imageHeight);

      const drawTime = new Date().getTime() - startTime;
      drawTimes.unshift(drawTime);
      drawTimes.splice(maxTimingSamples);

      return imageData;
    },

    getRenderTime() {
      return {
        render: renderTimes.reduce((a, b) => a + b, BigInt(0)),
        draw: drawTimes.reduce((a, b) => a + b, 0),
      };
    }
  };
};

const getSizes = (imageWidth: number, imageHeight: number, minSize = 1): d.Infer<typeof ImageSize>[] => {
  const sizes = [
    ImageSize({
      width: imageWidth,
      height: imageHeight,
      numPixels: imageWidth * imageHeight
    }),
  ];

  let w = imageWidth/2, h = imageHeight/2;

  while (w > minSize && h > minSize) {
    const width = Math.floor(w);
    const height = Math.floor(h);
    sizes.unshift(ImageSize({ width, height, numPixels: width * height }));
    w = w/2;
    h = h/2;
  }

  return sizes;
};
