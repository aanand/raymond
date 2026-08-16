import { tgpu, d } from 'typegpu';
import { pack4x8unorm } from 'typegpu/std';

import './style.css'

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
<canvas></canvas>
`

async function initialize() {
  const canvas = document.querySelector('canvas') as HTMLCanvasElement;
  const context = canvas.getContext('2d');
  if (!context) {
    throw new Error("Failed to get canvas context");
  }

  const circles = new Uint32Array([
    50, 50, 50,
    150, 50, 50,
  ]);
  const width = canvas.width;
  const height = canvas.height;
  const numPixels = width * height;

  const root = await tgpu.init();

  const state = root.createMutable(d.struct({
    circles: d.arrayOf(d.u32, circles.length),
    width: d.u32,
    pixels: d.arrayOf(d.u32, numPixels),
  }), {
    circles,
    width,
    pixels: new Uint32Array(numPixels),
  });

  const program = root.createGuardedComputePipeline((i) => {
    'use gpu';
    const x = i % state.$.width;
    const y = i / state.$.width;
    const circlesLength = state.$.circles.length;

    state.$.pixels[i] = pack4x8unorm(d.vec4f(0, 0, 0, 1));

    for (let ci = 0; ci < circlesLength; ci += 3) {
      const cx = state.$.circles[ci];
      const cy = state.$.circles[ci+1];
      const r  = state.$.circles[ci+2];

      const dx = cx - x;
      const dy = cy - y;

      if (dx*dx + dy*dy < r*r) {
        state.$.pixels[i] = pack4x8unorm(d.vec4f(1, 1, 1, 1));
      }
    }
  });

  program.dispatchThreads(numPixels);

  const value = await state.read();
  const imageData = new ImageData(new Uint8ClampedArray(new Uint32Array(value.pixels).buffer), width, height);
  context.putImageData(imageData, 0, 0);
}

initialize();
