import tgpu, { d } from "typegpu";
import { normalize, pack4x8unorm } from "typegpu/std";

import { hitSphere, Sphere } from "./sphere";
import { Ray, Interval, HitRecord, didNotHit, interval, INF, noise } from "./utils";

export const render = async ({ aspectRatio, imageWidth, samplesPerPixel, world, canvas }: {
  aspectRatio: number,
  imageWidth: number,
  samplesPerPixel: number,
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

  const rayColor = buildRayColorFunction(world);

  const numPixels = imageWidth * imageHeight;
  const state = root.createMutable(d.struct({
    sampleIndex: d.u32,
    accumulator: d.arrayOf(d.vec4f, numPixels),
    pixels: d.arrayOf(d.u32, numPixels),
  }), {
    sampleIndex: 0,
    accumulator: new Float32Array(numPixels * 4),
    pixels: new Uint32Array(numPixels),
  });

  const accumulate = root.createGuardedComputePipeline((pixelIndex) => {
    'use gpu';

    const x = d.u32(pixelIndex % imageWidth);
    const y = d.u32(pixelIndex / imageWidth);

    const offsetX = noise(d.f32(x + state.$.sampleIndex), d.f32(y)) - 0.5;
    const offsetY = noise(d.f32(x), d.f32(y + state.$.sampleIndex)) - 0.5;

    const pixelCenter = pixel00Loc
      .add(pixelDeltaU.mul(d.f32(x) + offsetX))
      .add(pixelDeltaV.mul(d.f32(y) + offsetY));

    // if (x !== d.u32(imageWidth/2) || y !== d.u32(imageHeight/2)) {
    //   state.$.accumulator[pixelIndex] = d.vec4f(0, 0, 0, 1);
    //   return;
    // }

    const ray = Ray({
      origin: cameraCenter,
      direction: pixelCenter.sub(cameraCenter)
    });

    const color = rayColor(ray, world);

    const currentValue = state.$.accumulator[pixelIndex];
    state.$.accumulator[pixelIndex] = d.vec4f(
      currentValue[0] + color[0],
      currentValue[1] + color[1],
      currentValue[2] + color[2],
      currentValue[3] + 1,
    );
  });

  console.time('Render');
  for (let s = 0; s < samplesPerPixel; s++) {
    state.patch({ sampleIndex: s });
    accumulate.dispatchThreads(numPixels);
  }

  const sum = root.createGuardedComputePipeline((pixelIndex) => {
    'use gpu';
    const accumulation = state.$.accumulator[pixelIndex];
    state.$.pixels[pixelIndex] = pack4x8unorm(d.vec4f(
      accumulation[0] / accumulation[3],
      accumulation[1] / accumulation[3],
      accumulation[2] / accumulation[3],
      1.0
    ));
  });
  sum.dispatchThreads(numPixels);
  console.timeEnd('Render');

  const value = await state.read();
  const imageData = new ImageData(new Uint8ClampedArray(new Uint32Array(value.pixels).buffer), imageWidth, imageHeight);

  canvas.width = imageWidth;
  canvas.height = imageHeight;
  canvas.getContext('2d')!.putImageData(imageData, 0, 0);
}

function buildRayColorFunction(world: d.Infer<typeof Sphere>[]) {
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

  return tgpu.fn([Ray, World], d.vec3f)((ray, world) => {
    const hitRecord = hitWorld(world, ray, interval(0, INF));

    if (hitRecord.isHit) {
      const N = hitRecord.normal.add(1).div(2);
      return d.vec3f(N.x, N.y, N.z);
    } else {
      const unitDirection = normalize(ray.direction);
      const a = 0.5 * (unitDirection.y + 1.0);
      return d.vec3f(1, 1, 1).mul(1 - a).add(d.vec3f(0.5, 0.7, 1.0).mul(a));
    }
  });
}
