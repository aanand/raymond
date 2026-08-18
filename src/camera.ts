import tgpu, { d } from "typegpu";
import { cross, length, normalize, pack4x8unorm, radians, sqrt, tan } from "typegpu/std";

import { hitSphere, Sphere } from "./sphere";
import { Ray, Interval, HitRecord, didNotHit, interval, INF, noise, Bounce, didNotBounce } from "./utils";
import type { World } from "./world";
import { Dielectric, Lambertian, MATERIAL_DIELECTRIC, MATERIAL_LAMBERTIAN, MATERIAL_METAL, Metal, scatterDielectric, scatterLambertian, scatterMetal } from "./material";

export const createScene = async ({
  aspectRatio,
  imageWidth,
  
  vfov,
  lookFrom,
  lookAt,
  vup,

  samplesPerPixel,
  samplesPerPass,
  maxBounceDepth,
  
  world,
}: {
  aspectRatio: number,
  imageWidth: number,

  vfov: number,
  lookFrom: d.v3f,
  lookAt: d.v3f,
  vup: d.v3f,

  samplesPerPixel: number,
  samplesPerPass: number,
  maxBounceDepth: number,

  world: World,
}) => {
  const imageHeight = d.u32(Math.max(1, Math.floor(imageWidth/aspectRatio)));

  const cameraCenter = lookFrom;
  const focalLength = length(lookFrom.sub(lookAt));
  const theta = radians(vfov);
  const h = tan(theta/2.0);
  const viewportHeight = 2.0 * h * focalLength;
  const viewportWidth = d.f32(viewportHeight * (imageWidth/imageHeight));

  const w = normalize(lookFrom.sub(lookAt));
  const u = normalize(cross(vup, w));
  const v = cross(w, u);

  const viewportU = u.mul(viewportWidth);
  const viewportV = v.mul(-viewportHeight);

  const pixelDeltaU = viewportU.div(imageWidth);
  const pixelDeltaV = viewportV.div(imageHeight);

  const viewportUpperLeft = cameraCenter
    .sub(w.mul(focalLength))
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

  const pixelBuffer = root.createMutable(d.arrayOf(d.u32, numPixels));

  const state = root.createMutable(d.struct({
    sampleIndex: d.u32,
    currentSample: d.arrayOf(d.vec3f, numPixels),
    accumulatedSamples: d.arrayOf(d.vec4f, numPixels),
    bounces: d.arrayOf(Bounce, numPixels),
    randomValues: d.arrayOf(d.u32, numRandomValues),
  }), {
    sampleIndex: 0,
    currentSample: new Float32Array(numPixels * 4),
    accumulatedSamples: new Float32Array(numPixels * 4),
    bounces: Array.from({ length: 100 }).map(didNotBounce),
    randomValues: new Uint32Array(numRandomValues),
  });

  const randomFloat = tgpu.fn([d.u32], d.f32)(pixelIndex =>
    state.$.randomValues[pixelIndex % numRandomValues] / 0xFFFFFFFF);

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

    const result = rayTrace(ray, randomFloat(pixelIndex));

    state.$.currentSample[pixelIndex] = d.vec3f(result.color);
    state.$.bounces[pixelIndex] = Bounce(result.bounce);
  });

  const processBounces = root.createGuardedComputePipeline((pixelIndex) => {
    'use gpu';

    const bounce = state.$.bounces[pixelIndex];

    // const x = d.u32(pixelIndex % imageWidth);
    // const y = d.u32(pixelIndex / imageWidth);
    // const debug = x === 0 && y === 0; // x === d.u32(imageWidth/2) && y === d.u32(imageHeight/2);

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

  const accumulateCurrentSample = root.createGuardedComputePipeline((pixelIndex) => {
    'use gpu';
    state.$.accumulatedSamples[pixelIndex] = state.$.accumulatedSamples[pixelIndex].add(d.vec4f(
      state.$.currentSample[pixelIndex].r,
      state.$.currentSample[pixelIndex].g,
      state.$.currentSample[pixelIndex].b,
      1
    ));
  });

  function renderOnePass() {
    for (let s = 0; s < samplesPerPass; s++) {
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
      pixelBuffer.$[pixelIndex] = pack4x8unorm(d.vec4f(
        linearToGamma(accumulation[0] / accumulation[3]),
        linearToGamma(accumulation[1] / accumulation[3]),
        linearToGamma(accumulation[2] / accumulation[3]),
        1.0
      ));
    });
    writePixels.dispatchThreads(numPixels);
  }

  let numSamplesTaken = 0;
  function renderAllPasses() {
    if (numSamplesTaken >= samplesPerPixel) {
      console.log('Finished rendering');
      return;
    }

    renderOnePass();
    numSamplesTaken += samplesPerPass;

    setTimeout(renderAllPasses, 0);
  }

  renderAllPasses();

  return async (canvas: HTMLCanvasElement) => {
    const value = await pixelBuffer.read();
    const imageData = new ImageData(new Uint8ClampedArray(new Uint32Array(value).buffer), imageWidth, imageHeight);

    canvas.width = imageWidth;
    canvas.height = imageHeight;
    canvas.getContext('2d')!.putImageData(imageData, 0, 0);
  }
}

const RayTraceResult = d.struct({
  color: d.vec3f,
  bounce: Bounce,
});

function buildRayTraceFunction(world: World) {
  const spheres = tgpu.const(d.arrayOf(Sphere, world.spheres.length), world.spheres);
  const lambertians = tgpu.const(d.arrayOf(Lambertian, world.lambertians.length), world.lambertians);
  const metals = tgpu.const(d.arrayOf(Metal, world.metals.length), world.metals);
  const dielectrics = tgpu.const(d.arrayOf(Dielectric, world.dielectrics.length), world.dielectrics);

  const hitWorld = tgpu.fn([Ray, Interval], HitRecord)((ray, rayT) => {
    let hitRecord = didNotHit();
    let closestSoFar = rayT.max;

    for (let i = 0; i < spheres.$.length; i++) {
      const sphereHit = hitSphere(spheres.$[i], ray, interval(rayT.min, closestSoFar));
      if (sphereHit.isHit) {
        hitRecord = HitRecord(sphereHit);
        closestSoFar = sphereHit.t;
      }
    }

    return hitRecord;
  });

  return tgpu.fn([Ray, d.f32], RayTraceResult)((ray, randomFloat) => {
    const hitRecord = hitWorld(ray, interval(0.001, INF));

    if (hitRecord.isHit) {
      let bounce = didNotBounce();
      if (hitRecord.materialType === MATERIAL_LAMBERTIAN) {
        const lambertian = lambertians.$[hitRecord.materialIndex];
        bounce = scatterLambertian(lambertian, hitRecord, randomFloat);
      } else if (hitRecord.materialType === MATERIAL_METAL) {
        const metal = metals.$[hitRecord.materialIndex];
        bounce = scatterMetal(metal, ray, hitRecord, randomFloat);
      } else if (hitRecord.materialType === MATERIAL_DIELECTRIC) {
        const dielectric = dielectrics.$[hitRecord.materialIndex];
        bounce = scatterDielectric(dielectric, ray, hitRecord, randomFloat);
      }

      let color = d.vec3f(0, 0, 0);
      if (d.bool(bounce.didBounce)) {
        color = d.vec3f(bounce.attenuation);
      }

      return RayTraceResult({ color, bounce });
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
