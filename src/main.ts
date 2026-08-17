import { tgpu, d } from 'typegpu';
import { dot, length, normalize, pack4x8unorm, select, sqrt } from 'typegpu/std';

import './style.css'

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
<canvas></canvas>
`

const INF = d.f32(3.40282347e+38);

const aspectRatio = d.f32(16.0/9.0);
const imageWidth = d.u32(400);

const imageHeight = d.u32(Math.max(1, Math.floor(imageWidth/aspectRatio)));

const focalLength = d.f32(1.0);
const viewportHeight = d.f32(2.0);
const viewportWidth = d.f32(viewportHeight * (imageWidth/imageHeight));
const cameraCenter = d.vec3f(0, 0, 0);

const Interval = d.struct({
  min: d.f32,
  max: d.f32,
});

const interval = tgpu.fn([d.f32, d.f32], Interval)((min, max) => Interval({ min, max }));
const surrounds = tgpu.fn([Interval, d.f32], d.bool)((i, num) => i.min < num && num < i.max);

const Ray = d.struct({
  origin: d.vec3f,
  direction: d.vec3f,
});

const at = (ray: d.Infer<typeof Ray>, t: number) => {
  'use gpu';
  return ray.origin.add(ray.direction.mul(t));
}

const HitRecord = d.struct({
  isHit: d.bool,
  position: d.vec3f,
  normal: d.vec3f,
  t: d.f32,
  isFrontFace: d.bool,
});

const didNotHit = () => {
  'use gpu';
  return HitRecord({
    isHit: false,
    position: d.vec3f(0, 0, 0),
    normal: d.vec3f(0, 0, 1),
    t: 0,
    isFrontFace: false,
  });
}

const Sphere = d.struct({
  center: d.vec3f,
  radius: d.f32,
});

const hitSphere = tgpu.fn([Sphere, Ray, Interval], HitRecord)((sphere, ray, rayT) => {
  const oc = sphere.center.sub(ray.origin);
  const a = length(ray.direction) ** 2;
  const h = dot(ray.direction, oc);
  const c = length(oc) ** 2 - (sphere.radius * sphere.radius);
  const discriminant = h*h - a*c;

  if (discriminant < 0) {
    return didNotHit();
  }

  const sqrtd = sqrt(discriminant);

  // Find the nearest root that lies in the acceptable range.
  let root = (h - sqrtd) / a;
  if (!surrounds(rayT, root)) {
    root = (h + sqrtd) / a;
    if (!surrounds(rayT, root)) {
      return didNotHit();
    }
  }

  const t = root;
  const position = at(ray, t);
  const outwardNormal = position.sub(sphere.center).div(sphere.radius);
  const isFrontFace = dot(ray.direction, outwardNormal) < 0;
  // const normal = isFrontFace ? outwardNormal : outwardNormal.mul(-1);
  const normal = select(outwardNormal.mul(-1), outwardNormal, isFrontFace);

  return HitRecord({ isHit: true, position, normal, t, isFrontFace });
});

const hitWorld = tgpu.fn([d.arrayOf(Sphere, 2), Ray, Interval], HitRecord)((world, ray, rayT) => {
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

const rayColor = tgpu.fn([Ray, d.arrayOf(Sphere, 2)], d.vec4f)((ray, world) => {
  'use gpu';

  const hitRecord = hitWorld(world, ray, interval(0, INF));

  if (hitRecord.isHit) {
    const N = hitRecord.normal.add(1).div(2);
    return d.vec4f(N.x, N.y, N.z, 1.0);
  } else {
    const unitDirection = normalize(ray.direction);
    const a = 0.5 * (unitDirection.y + 1.0);
    return d.vec4f(1, 1, 1, 1).mul(1-a).add(d.vec4f(0.5, 0.7, 1.0, 1).mul(a));
  }
});

const world = d.arrayOf(Sphere, 2)([
  Sphere({ center: d.vec3f(0,    0,   -1), radius: -0.5 }),
  Sphere({ center: d.vec3f(0, -100.5, -1), radius:  100 }),
]);

async function initialize() {
  const viewportU = d.vec3f(viewportWidth, 0, 0);
  const viewportV = d.vec3f(0, -viewportHeight, 0);

  const pixelDeltaU = viewportU.div(imageWidth);
  const pixelDeltaV = viewportV.div(imageHeight);

  const viewportUpperLeft = cameraCenter
    .sub(d.vec3f(0, 0, focalLength))
    .sub(viewportU.div(2))
    .sub(viewportV.div(2));

  const pixel00Loc = viewportUpperLeft.add(pixelDeltaU.add(pixelDeltaV).mul(0.5));

  const canvas = document.querySelector('canvas') as HTMLCanvasElement;
  canvas.width = imageWidth;
  canvas.height = imageHeight;

  const context = canvas.getContext('2d');
  if (!context) {
    throw new Error("Failed to get canvas context");
  }

  const root = await tgpu.init();

  const numPixels = imageWidth * imageHeight;
  const state = root.createMutable(d.struct({
    pixels: d.arrayOf(d.u32, numPixels),
  }), {
    pixels: new Uint32Array(numPixels),
  });

  const program = root.createGuardedComputePipeline((threadId) => {
    'use gpu';

    const i = d.u32(threadId % imageWidth);
    const j = d.u32(threadId / imageWidth);

    // if (i !== d.u32(imageWidth/2) || j !== d.u32(imageHeight/2)) {
    //   state.$.pixels[threadId] = pack4x8unorm(d.vec4f(0, 0, 0, 1));
    //   return;
    // }

    const pixelCenter = pixel00Loc
      .add(pixelDeltaU.mul(d.f32(i)))
      .add(pixelDeltaV.mul(d.f32(j)));

    const ray = Ray({
      origin: cameraCenter,
      direction: pixelCenter.sub(cameraCenter)
    });

    const color = rayColor(ray, world);

    state.$.pixels[threadId] = pack4x8unorm(color);
  });

  program.dispatchThreads(numPixels);

  const value = await state.read();
  const imageData = new ImageData(new Uint8ClampedArray(new Uint32Array(value.pixels).buffer), imageWidth, imageHeight);
  context.putImageData(imageData, 0, 0);
}

initialize();
