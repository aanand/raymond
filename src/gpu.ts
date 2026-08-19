import tgpu, { d } from "typegpu";
import { pack4x8unorm, sqrt } from "typegpu/std";

import { Bounce, didNotBounce, noise, Ray } from "./utils";
import { buildRayTraceFunction } from "./rayTrace";
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
    currentSample: d.arrayOf(d.vec3f, numPixels),
    accumulatedSamples: d.arrayOf(d.vec4f, numPixels),
    bounces: d.arrayOf(Bounce, numPixels),
    randomValues: d.arrayOf(d.u32, numRandomValues),
  });

  let camera = initialCamera;

  const initialState = () => ({
    camera,
    sampleIndex: 0,
    currentSample: new Float32Array(numPixels * 4),
    accumulatedSamples: new Float32Array(numPixels * 4),
    bounces: Array.from({ length: 100 }).map(didNotBounce),
    randomValues: new Uint32Array(numRandomValues),
  });

  const rayTrace = buildRayTraceFunction(world);

  const root = await tgpu.init();
  const state = root.createMutable(State, initialState());

  const randomFloat = tgpu.fn([d.u32], d.f32)(pixelIndex =>
    state.$.randomValues[pixelIndex % numRandomValues] / 0xFFFFFFFF);

  const fireInitialRay = tgpu.fn([d.u32])((pixelIndex) => {
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

    const result = rayTrace(ray, randomFloat(pixelIndex));

    state.$.currentSample[pixelIndex] = d.vec3f(result.color);
    state.$.bounces[pixelIndex] = Bounce(result.bounce);
  });

  const processBounces = tgpu.fn([d.u32])((pixelIndex) => {
    'use gpu';

    const bounce = state.$.bounces[pixelIndex];
    const bounceResult = rayTrace(bounce.ray, randomFloat(pixelIndex));

    const currentValue = state.$.currentSample[pixelIndex];
    state.$.currentSample[pixelIndex] = d.vec3f(
      bounce.didBounce ? currentValue[0] * bounceResult.color[0] : currentValue[0],
      bounce.didBounce ? currentValue[1] * bounceResult.color[1] : currentValue[1],
      bounce.didBounce ? currentValue[2] * bounceResult.color[2] : currentValue[2],
    );

    state.$.bounces[pixelIndex] = Bounce({
      didBounce: bounceResult.bounce.didBounce,
      ray: bounceResult.bounce.ray,
      attenuation: bounceResult.bounce.attenuation,
    });
  });

  const accumulateCurrentSample = tgpu.fn([d.u32])((pixelIndex) => {
    'use gpu';
    state.$.accumulatedSamples[pixelIndex] = state.$.accumulatedSamples[pixelIndex].add(d.vec4f(
      state.$.currentSample[pixelIndex].r,
      state.$.currentSample[pixelIndex].g,
      state.$.currentSample[pixelIndex].b,
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

  function renderOnePass(samplesPerPass: number, maxBounceDepth: number) {
    state.patch({ randomValues: generateRandomValues() });

    root.createGuardedComputePipeline((pixelIndex) => {
      'use gpu';

      for (let s = 0; s < samplesPerPass; s++) {
        state.$.sampleIndex++;
        fireInitialRay(pixelIndex);

        for (let i = 0; i < maxBounceDepth; i++) {
          processBounces(pixelIndex);
        }

        accumulateCurrentSample(pixelIndex);
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