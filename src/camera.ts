import tgpu, { d } from "typegpu";
import { normalize, pack4x8unorm } from "typegpu/std";

import { hitSphere, Sphere } from "./sphere";
import { Ray, Interval, HitRecord, didNotHit, interval, INF, noise, Bounce, didNotBounce, randomOnHemisphere } from "./utils";

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

  const state = root.createMutable(d.struct({
    sampleIndex: d.u32,
    currentSample: d.arrayOf(d.vec3f, numPixels),
    accumulatedSamples: d.arrayOf(d.vec4f, numPixels),
    bounces: d.arrayOf(Bounce, numPixels),
    // numBounces: d.atomic(d.u32),
    pixels: d.arrayOf(d.u32, numPixels),
    time: d.u32,
  }), {
    sampleIndex: 0,
    currentSample: new Float32Array(numPixels * 3),
    accumulatedSamples: new Float32Array(numPixels * 4),
    bounces: Array.from({ length: 100 }).map(didNotBounce),
    // numBounces: 0,
    pixels: new Uint32Array(numPixels),
    time: 0,
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

    const uv = d.vec2f((d.f32(x) + offsetX), (d.f32(y) + offsetY));

    const ray = Ray({
      origin: cameraCenter,
      direction: pixelCenter.sub(cameraCenter)
    });

    const result = rayTrace(ray, world, uv, state.$.time);

    state.$.currentSample[pixelIndex] = d.vec3f(result.color);
    state.$.bounces[pixelIndex] = Bounce(result.bounce);
  });

  const processBounces = root.createGuardedComputePipeline((pixelIndex) => {
    'use gpu';

    const bounce = state.$.bounces[pixelIndex];

    // const x = d.u32(pixelIndex % imageWidth);
    // const y = d.u32(pixelIndex / imageWidth);
    // const debug = x === 0 && y === 0; // x === d.u32(imageWidth/2) && y === d.u32(imageHeight/2);

    const bounceResult = rayTrace(bounce.ray, world, bounce.uv, state.$.time);
    const currentValue = state.$.currentSample[pixelIndex];

    state.$.currentSample[pixelIndex] = d.vec3f(
      bounce.didBounce ? currentValue[0] * bounceResult.color[0] : currentValue[0],
      bounce.didBounce ? currentValue[1] * bounceResult.color[1] : currentValue[1],
      bounce.didBounce ? currentValue[2] * bounceResult.color[2] : currentValue[2],
    );

    state.$.bounces[pixelIndex] = Bounce({
      didBounce: bounceResult.bounce.didBounce,
      ray: bounceResult.bounce.ray,
      uv: bounce.uv,
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

  let time = 0;
  console.time('Render');
  for (let s = 0; s < samplesPerPixel; s++) {
    console.time('Fire initial rays');
    time += Math.floor(Math.random() * 1024);
    state.patch({ sampleIndex: s, time });
    fireInitialRays.dispatchThreads(numPixels);
    console.timeEnd('Fire initial rays');

    for (let i = 0; i < maxBounceDepth; i++) {
      console.time('Process bounces')
      time += Math.floor(Math.random() * 1024);
      state.patch({ time: time });
      processBounces.dispatchThreads(numPixels);
      console.timeEnd('Process bounces');
    }

    console.time('Accumulate current sample');
    accumulateCurrentSample.dispatchThreads(numPixels);
    console.timeEnd('Accumulate current sample');
  }

  console.time('Average samples together');
  const sum = root.createGuardedComputePipeline((pixelIndex) => {
    'use gpu';
    const accumulation = state.$.accumulatedSamples[pixelIndex];
    state.$.pixels[pixelIndex] = pack4x8unorm(d.vec4f(
      accumulation[0] / accumulation[3],
      accumulation[1] / accumulation[3],
      accumulation[2] / accumulation[3],
      1.0
    ));
  });
  sum.dispatchThreads(numPixels);
  console.timeEnd('Average samples together')
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

  return tgpu.fn([Ray, World, d.vec2f, d.u32], RayTraceResult)((ray, world, uv, time) => {
    const hitRecord = hitWorld(world, ray, interval(0.001, INF));

    if (hitRecord.isHit) {
      const color = d.vec3f(0.5, 0.5, 0.5);
      const bounceDirection = randomOnHemisphere(uv[0]*time, uv[1]*time, hitRecord.normal);
      const bouncedRay = Ray({ origin: hitRecord.position, direction: bounceDirection });
      return RayTraceResult({
        color,
        bounce: Bounce({ didBounce: 1, ray: bouncedRay, uv }),
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
