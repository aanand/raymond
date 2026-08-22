import tgpu, { d } from "typegpu";
import { pack4x8unorm, sqrt } from "typegpu/std";

import { Ray } from "./utils";
import { buildRayTraceFunction, RayTraceResult } from "./rayTrace";
import type { World } from "./world";
import { CameraStruct, type Camera } from "./camera";
import { randf } from "@typegpu/noise";

export const makeGpuFunctions = async ({
  initialCamera,
  imageWidth,
  imageHeight,
  world,
}: {
  initialCamera: Camera,
  imageWidth: number,
  imageHeight: number,
  world: World,
}) => {
  const numPixels = imageWidth * imageHeight;

  const State = d.struct({
    camera: CameraStruct,
    sampleIndex: d.u32,
    accumulatedSamples: d.arrayOf(d.vec4f, numPixels),
  });

  let camera = initialCamera;

  const initialState = () => ({
    camera,
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

  const fireInitialRay = tgpu.fn([d.u32], RayTraceResult)((pixelIndex) => {
    'use gpu';

    const x = d.u32(pixelIndex % imageWidth);
    const y = d.u32(pixelIndex / imageWidth);

    const offsetX = randf.sample() - 0.5;
    const offsetY = randf.sample() - 0.5;

    const pixelCenter = state.$.camera.pixel00Loc
      .add(state.$.camera.pixelDeltaU.mul(d.f32(x) + offsetX))
      .add(state.$.camera.pixelDeltaV.mul(d.f32(y) + offsetY));

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

    const initialResult = fireInitialRay(pixelIndex);
    const bouncedResult = processBounces(initialResult, numBounces);

    state.$.accumulatedSamples[pixelIndex] = state.$.accumulatedSamples[pixelIndex].add(d.vec4f(
      bouncedResult.color.r,
      bouncedResult.color.g,
      bouncedResult.color.b,
      1
    ));
  });

  const linearToGamma = tgpu.fn([d.f32], d.f32)(linear => {
    if (linear > 0) {
      return sqrt(linear);
    }

    return 0;
  });

  const pixelBuffer = root.createMutable(d.arrayOf(d.u32, numPixels));

  const writePixels = tgpu.fn([d.u32])((pixelIndex) => {
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

  const pipeline = root.createGuardedComputePipeline((pixelIndex) => {
    'use gpu';

    randf.seed2(
      d.vec2f(
        (d.f32(pixelIndex) / d.f32(numPixels)) * 2000 - 1000,
        (d.f32(state.$.sampleIndex) / d.f32(passConfig.$.samplesPerPixel)) * 2000 - 1000));

    for (let s = 0; s < d.i32(passConfig.$.samplesPerPass); s++) {
      sample(pixelIndex, passConfig.$.numBounces);
      state.$.sampleIndex++;
    }

    writePixels(pixelIndex);
  })
  .withPerformanceCallback((start, end) => {
    renderTimes.unshift((end - start) / BigInt(1000000));
    renderTimes.splice(maxTimingSamples);
  });

  function renderOnePass(samplesPerPixel: number, samplesPerPass: number, numBounces: number) {
    passConfig.write({ samplesPerPixel, samplesPerPass, numBounces })
    pipeline.dispatchThreads(numPixels);
  };

  return {
    renderOnePass,

    updateCamera(newValue: Camera) {
      camera = newValue;
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