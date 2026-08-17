import { tgpu, d } from 'typegpu';
import { dot, normalize, pack4x8unorm } from 'typegpu/std';

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

const hitSphere = (center: d.v3f, radius: number, ray: d.Infer<typeof Ray>) => {
  'use gpu';
  const oc = center.sub(ray.origin);
  const a = dot(ray.direction, ray.direction);
  const b = -2.0 * dot(ray.direction, oc);
  const c = dot(oc, oc) - radius*radius;
  const discriminant = b*b - 4*a*c;
  return discriminant >= 0;
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

    if (hitSphere(d.vec3f(0, 0, -1), 0.5, ray)) {
      state.$.pixels[threadId] = pack4x8unorm(d.vec4f(1, 0, 0, 1));
    } else {
      const unitDirection = normalize(ray.direction);
      const a = 0.5 * (unitDirection.y + 1.0);
      const color = d.vec4f(1, 1, 1, 1).mul(1-a).add(d.vec4f(0.5, 0.7, 1.0, 1).mul(a));

      state.$.pixels[threadId] = pack4x8unorm(color);
    }
  });

  program.dispatchThreads(numPixels);

  const value = await state.read();
  const imageData = new ImageData(new Uint8ClampedArray(new Uint32Array(value.pixels).buffer), imageWidth, imageHeight);
  context.putImageData(imageData, 0, 0);
}

initialize();
