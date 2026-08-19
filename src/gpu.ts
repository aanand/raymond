import tgpu, { d } from "typegpu";
import { pack4x8unorm, sqrt } from "typegpu/std";

import { noise, Ray } from "./utils";
import { buildRayTraceFunction, RayTraceResult } from "./rayTrace";
import type { World } from "./world";
import { CameraStruct, type Camera } from "./camera";

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
  const numRandomValues = Math.floor(65536 / Uint32Array.BYTES_PER_ELEMENT);

  const generateRandomValues = () => {
    const values = new Uint32Array(numRandomValues);
    crypto.getRandomValues(values);
    return values;
  }

  const State = d.struct({
    camera: CameraStruct,
    sampleIndex: d.u32,
    accumulatedSamples: d.arrayOf(d.vec4f, numPixels),
    randomValues: d.arrayOf(d.u32, numRandomValues),
  });

  let camera = initialCamera;

  const initialState = () => ({
    camera,
    sampleIndex: 0,
    accumulatedSamples: new Float32Array(numPixels * 4),
    randomValues: new Uint32Array(numRandomValues),
  });

  const rayTrace = buildRayTraceFunction(world);

  const root = await tgpu.init();
  const state = root.createMutable(State, initialState());

  const randomFloat = tgpu.fn([d.u32], d.f32)(pixelIndex => {
    let float = state.$.randomValues[pixelIndex % numRandomValues] / 0xFFFFFFFF;

    // Ensure we never return 1.0. Does this bias it? Probably technically, yeah.
    if (float === 1.0) {
      return 0.0;
    }

    return float;
  });

  const fireInitialRay = tgpu.fn([d.u32], RayTraceResult)((pixelIndex) => {
    'use gpu';

    const x = d.u32(pixelIndex % imageWidth);
    const y = d.u32(pixelIndex / imageWidth);

    // const debug = x === 0 && y === 0; // x === d.u32(imageWidth/2) && y === d.u32(imageHeight/2);

    const offsetX = noise(d.f32(x + state.$.sampleIndex), d.f32(y)) - 0.5;
    const offsetY = noise(d.f32(x), d.f32(y + state.$.sampleIndex)) - 0.5;

    const pixelCenter = state.$.camera.pixel00Loc
      .add(state.$.camera.pixelDeltaU.mul(d.f32(x) + offsetX))
      .add(state.$.camera.pixelDeltaV.mul(d.f32(y) + offsetY));

    const ray = Ray({
      origin: state.$.camera.center,
      direction: pixelCenter.sub(state.$.camera.center)
    });

    return rayTrace(ray, randomFloat(pixelIndex));
  });

  const processBounces = tgpu.fn([d.u32, RayTraceResult, d.u32], RayTraceResult)((pixelIndex, rayTraceResult, numBounces) => {
    'use gpu';

    let result = RayTraceResult(rayTraceResult);

    for (let i = d.u32(0); i < numBounces; i++) {
      const currentColor = result.color;
      const bounce = result.bounce;
      const bounceResult = rayTrace(bounce.ray, randomFloat(pixelIndex));

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
    const bouncedResult = processBounces(pixelIndex, initialResult, numBounces);

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

  function renderOnePass(samplesPerPass: number, numBounces: number) {
    state.patch({ randomValues: generateRandomValues() });

    root.createGuardedComputePipeline((pixelIndex) => {
      'use gpu';

      for (let s = 0; s < samplesPerPass; s++) {
        state.$.sampleIndex++;
        sample(pixelIndex, numBounces);
      }

      writePixels(pixelIndex);
    }).dispatchThreads(numPixels);
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
      const value = await pixelBuffer.read();
      return new ImageData(new Uint8ClampedArray(new Uint32Array(value).buffer), imageWidth, imageHeight);
    }
  };
};