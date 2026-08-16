import { tgpu, d } from 'typegpu';
import { normalize, pack4x8unorm } from 'typegpu/std';

import './style.css'

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
<canvas></canvas>
`

const aspectRatio = 16/9;
const imageWidth = 400;

const imageHeight = Math.max(1, Math.floor(imageWidth/aspectRatio));

const focalLength = 1;
const viewportHeight = 2;
const viewportWidth = viewportHeight * (imageWidth/imageHeight);
const cameraCenter = d.vec3f(0, 0, 0);

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
    const j = d.u32(threadId / imageHeight);

    const pixelCenter = pixel00Loc
      .add(pixelDeltaU.mul(d.f32(i)))
      .add(pixelDeltaV.mul(d.f32(j)));

    const rayDirection = pixelCenter.sub(cameraCenter);

    const unitDirection = normalize(rayDirection);
    const a = 0.5 * (unitDirection.y + 1.0);
    const color = d.vec4f(1, 1, 1, 1).mul(1-a).add(d.vec4f(0.5, 0.7, 1.0, 1).mul(a));
    
    state.$.pixels[threadId] = pack4x8unorm(color);
  });

  program.dispatchThreads(numPixels);

  const value = await state.read();
  const imageData = new ImageData(new Uint8ClampedArray(new Uint32Array(value.pixels).buffer), imageWidth, imageHeight);
  context.putImageData(imageData, 0, 0);
}

initialize();
