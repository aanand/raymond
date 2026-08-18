import tgpu, { d } from "typegpu";
import { floor, length, normalize, pack4x8unorm, sqrt } from "typegpu/std";

import { hitSphere, Sphere } from "./sphere";
import { Ray, Interval, HitRecord, didNotHit, interval, INF, noise, Bounce, didNotBounce, randomUnitVector } from "./utils";

export const render = async ({ aspectRatio, imageWidth, samplesPerPixel, maxBounceDepth, world, canvas }: {
  aspectRatio: number,
  imageWidth: number,
  samplesPerPixel: number,
  maxBounceDepth: number,
  world: d.Infer<typeof Sphere>[],
  canvas: HTMLCanvasElement,
}) => {
  const imageHeight = d.u32(Math.max(1, Math.floor(imageWidth/aspectRatio)));

  const focalLength = d.f32(1.0);
  const viewportHeight = d.f32(2.0);
  const viewportWidth = d.f32(viewportHeight * (imageWidth/imageHeight));
  const cameraCenter = d.vec3f(0, 0, 0);

  const viewportU = d.vec3f(viewportWidth, 0, 0);
  const viewportV = d.vec3f(0, -viewportHeight, 0);

  const pixelDeltaU = viewportU.div(imageWidth);
  const pixelDeltaV = viewportV.div(imageHeight);

  const viewportUpperLeft = cameraCenter
    .sub(d.vec3f(0, 0, focalLength))
    .sub(viewportU.div(2))
    .sub(viewportV.div(2));

  const pixel00Loc = viewportUpperLeft.add(pixelDeltaU.add(pixelDeltaV).mul(0.5));

  const root = await tgpu.init();

  const rayTrace = buildRayTraceFunction(world);

  const numPixels = imageWidth * imageHeight;

  const numRandomValues = Math.floor(65536 / Uint32Array.BYTES_PER_ELEMENT);

  const generateRandomValues = () => {
    const values = new Uint32Array(numRandomValues);
    crypto.getRandomValues(values);
    return values;
  }

  const state = root.createMutable(d.struct({
    sampleIndex: d.u32,
    currentSample: d.arrayOf(d.vec3f, numPixels),
    accumulatedSamples: d.arrayOf(d.vec4f, numPixels),
    bounces: d.arrayOf(Bounce, numPixels),
    pixels: d.arrayOf(d.u32, numPixels),
    randomValues: d.arrayOf(d.u32, numRandomValues),
  }), {
    sampleIndex: 0,
    currentSample: new Float32Array(numPixels * 4),
    accumulatedSamples: new Float32Array(numPixels * 4),
    bounces: Array.from({ length: 100 }).map(didNotBounce),
    pixels: new Uint32Array(numPixels),
    randomValues: new Uint32Array(numRandomValues),
  });

  const fireInitialRays = root.createGuardedComputePipeline((pixelIndex) => {
    'use gpu';

    const x = d.u32(pixelIndex % imageWidth);
    const y = d.u32(pixelIndex / imageWidth);

    // const debug = x === 0 && y === 0; // x === d.u32(imageWidth/2) && y === d.u32(imageHeight/2);

    const offsetX = noise(d.f32(x + state.$.sampleIndex), d.f32(y)) - 0.5;
    const offsetY = noise(d.f32(x), d.f32(y + state.$.sampleIndex)) - 0.5;

    const pixelCenter = pixel00Loc
      .add(pixelDeltaU.mul(d.f32(x) + offsetX))
      .add(pixelDeltaV.mul(d.f32(y) + offsetY));

    const ray = Ray({
      origin: cameraCenter,
      direction: pixelCenter.sub(cameraCenter)
    });

    const randomFloat = state.$.randomValues[pixelIndex % numRandomValues] / 0xFFFFFFFF;
    const result = rayTrace(ray, world, floor(randomFloat * 1000), 1000);

    state.$.currentSample[pixelIndex] = d.vec3f(result.color);
    state.$.bounces[pixelIndex] = Bounce(result.bounce);
  });

  const processBounces = root.createGuardedComputePipeline((pixelIndex) => {
    'use gpu';

    const bounce = state.$.bounces[pixelIndex];

    // const x = d.u32(pixelIndex % imageWidth);
    // const y = d.u32(pixelIndex / imageWidth);
    // const debug = x === 0 && y === 0; // x === d.u32(imageWidth/2) && y === d.u32(imageHeight/2);

    const randomFloat = state.$.randomValues[pixelIndex % numRandomValues] / 0xFFFFFFFF;
    const bounceResult = rayTrace(bounce.ray, world, floor(randomFloat * 1000), 1000);
    const currentValue = state.$.currentSample[pixelIndex];

    state.$.currentSample[pixelIndex] = d.vec3f(
      bounce.didBounce ? currentValue[0] * bounceResult.color[0] : currentValue[0],
      bounce.didBounce ? currentValue[1] * bounceResult.color[1] : currentValue[1],
      bounce.didBounce ? currentValue[2] * bounceResult.color[2] : currentValue[2],
    );

    state.$.bounces[pixelIndex] = Bounce({
      didBounce: bounceResult.bounce.didBounce,
      ray: bounceResult.bounce.ray,
    });
  });

  const accumulateCurrentSample = root.createGuardedComputePipeline((pixelIndex) => {
    'use gpu';
    state.$.accumulatedSamples[pixelIndex] = state.$.accumulatedSamples[pixelIndex].add(d.vec4f(
      state.$.currentSample[pixelIndex].r,
      state.$.currentSample[pixelIndex].g,
      state.$.currentSample[pixelIndex].b,
      1
    ));
  });

  console.time('Render');
  for (let s = 0; s < samplesPerPixel; s++) {
    state.patch({ sampleIndex: s, randomValues: generateRandomValues() });
    fireInitialRays.dispatchThreads(numPixels);

    for (let i = 0; i < maxBounceDepth; i++) {
      state.patch({ randomValues: generateRandomValues() });
      processBounces.dispatchThreads(numPixels);
    }

    accumulateCurrentSample.dispatchThreads(numPixels);
  }

  const linearToGamma = tgpu.fn([d.f32], d.f32)(linear => {
    if (linear > 0) {
      return sqrt(linear);
    }

    return 0;
  });

  const writePixels = root.createGuardedComputePipeline((pixelIndex) => {
    'use gpu';
    const accumulation = state.$.accumulatedSamples[pixelIndex];
    state.$.pixels[pixelIndex] = pack4x8unorm(d.vec4f(
      linearToGamma(accumulation[0] / accumulation[3]),
      linearToGamma(accumulation[1] / accumulation[3]),
      linearToGamma(accumulation[2] / accumulation[3]),
      1.0
    ));
  });
  writePixels.dispatchThreads(numPixels);
  console.timeEnd('Render');

  const value = await state.read();
  const imageData = new ImageData(new Uint8ClampedArray(new Uint32Array(value.pixels).buffer), imageWidth, imageHeight);

  canvas.width = imageWidth;
  canvas.height = imageHeight;
  canvas.getContext('2d')!.putImageData(imageData, 0, 0);
}

const RayTraceResult = d.struct({
  color: d.vec3f,
  bounce: Bounce,
});

function buildRayTraceFunction(world: d.Infer<typeof Sphere>[]) {
  const World = d.arrayOf(Sphere, world.length);

  const hitWorld = tgpu.fn([World, Ray, Interval], HitRecord)((world, ray, rayT) => {
    let hitRecord = didNotHit();
    let closestSoFar = rayT.max;

    for (let i = 0; i < world.length; i++) {
      const sphereHit = hitSphere(world[i], ray, interval(rayT.min, closestSoFar));
      if (sphereHit.isHit) {
        hitRecord = HitRecord(sphereHit);
        closestSoFar = sphereHit.t;
      }
    }

    return hitRecord;
  });

  return tgpu.fn([Ray, World, d.f32, d.f32], RayTraceResult)((ray, world, i, samples) => {
    const hitRecord = hitWorld(world, ray, interval(0.001, INF));

    if (hitRecord.isHit) {
      const color = d.vec3f(0.5, 0.5, 0.5);
      const bounceDirection = normalize(hitRecord.normal.add(randomUnitVector(i, samples)));
      const bouncedRay = Ray({ origin: hitRecord.position, direction: bounceDirection });
      return RayTraceResult({
        color,
        bounce: Bounce({ didBounce: 1, ray: bouncedRay }),
      });
    } else {
      const unitDirection = normalize(ray.direction);
      const a = 0.5 * (unitDirection.y + 1.0);
      return RayTraceResult({
        color: d.vec3f(1, 1, 1).mul(1 - a).add(d.vec3f(0.5, 0.7, 1.0).mul(a)),
        bounce: didNotBounce(),
      });
    }
  });
}
