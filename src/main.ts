import { tgpu, d } from 'typegpu';
import { dot, length, normalize, pack4x8unorm, sqrt } from 'typegpu/std';

import './style.css'

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
<canvas></canvas>
`

const aspectRatio = d.f32(16.0/9.0);
const imageWidth = d.u32(400);

const imageHeight = d.u32(Math.max(1, Math.floor(imageWidth/aspectRatio)));

const focalLength = d.f32(1.0);
const viewportHeight = d.f32(2.0);
const viewportWidth = d.f32(viewportHeight * (imageWidth/imageHeight));
const cameraCenter = d.vec3f(0, 0, 0);

const Ray = d.struct({
  origin: d.vec3f,
  direction: d.vec3f,
});

const at = (ray: d.Infer<typeof Ray>, t: number) => {
  'use gpu';
  return ray.origin.add(ray.direction.mul(t));
}

const hitSphere = (center: d.v3f, radius: number, ray: d.Infer<typeof Ray>) => {
  'use gpu';
  const oc = center.sub(ray.origin);
  const a = length(ray.direction) ** 2;
  const h = dot(ray.direction, oc);
  const c = length(oc) ** 2 - radius*radius;
  const discriminant = h*h - a*c;
  
  if (discriminant < 0) {
    return d.f32(-1.0);
  } else {
    return d.f32((h - sqrt(discriminant)) / a);
  }
}

const rayColor = (ray: d.Infer<typeof Ray>) => {
  'use gpu';

  const t = hitSphere(d.vec3f(0, 0, -1), 0.5, ray);

  if (t > 0.0) {
    const N = normalize(at(ray, t).sub(d.vec3f(0, 0, -1)));
    return d.vec4f((N.x + 1)/2, (N.y + 1)/2, (N.z + 1)/2, 1.0);
  } else {
    const unitDirection = normalize(ray.direction);
    const a = 0.5 * (unitDirection.y + 1.0);
    return d.vec4f(1, 1, 1, 1).mul(1-a).add(d.vec4f(0.5, 0.7, 1.0, 1).mul(a));
  }
}

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

    const pixelCenter = pixel00Loc
      .add(pixelDeltaU.mul(d.f32(i)))
      .add(pixelDeltaV.mul(d.f32(j)));

    const ray = Ray({
      origin: cameraCenter,
      direction: pixelCenter.sub(cameraCenter)
    });

    const color = rayColor(ray);

    state.$.pixels[threadId] = pack4x8unorm(color);
  });

  program.dispatchThreads(numPixels);

  const value = await state.read();
  const imageData = new ImageData(new Uint8ClampedArray(new Uint32Array(value.pixels).buffer), imageWidth, imageHeight);
  context.putImageData(imageData, 0, 0);
}

initialize();
