import { d } from 'typegpu';

import { Sphere } from './sphere';
import { createScene } from './camera';

import type { World } from './world';
import { Lambertian, MATERIAL_LAMBERTIAN, MATERIAL_METAL, Metal } from './material';

import './style.css';

const aspectRatio = 16.0/9.0;
const imageWidth = 800;
const samplesPerPixel = 2000;
const samplesPerPass = 20;
const maxBounceDepth = 10;

const world: World = {
  spheres: [
    Sphere({
      center: d.vec3f(0, -100.5, -1),
      radius: 100,
      materialType: MATERIAL_LAMBERTIAN,
      materialIndex: 0,
    }),
    Sphere({
      center: d.vec3f(0, 0, -1.2),
      radius:  0.5,
      materialType: MATERIAL_LAMBERTIAN,
      materialIndex: 1,
    }),
    Sphere({
      center: d.vec3f(-1, 0, -1),
      radius: 0.5,
      materialType: MATERIAL_METAL,
      materialIndex: 0,
    }),
    Sphere({
      center: d.vec3f(1, 0, -1),
      radius: 0.5,
      materialType: MATERIAL_METAL,
      materialIndex: 1,
    }),
  ],
  lambertians: [
    Lambertian({ albedo: d.vec3f(0.8, 0.8, 0.0) }), // Ground
    Lambertian({ albedo: d.vec3f(0.1, 0.2, 0.5) }), // Middle sphere
  ],
  metals: [
    Metal({ albedo: d.vec3f(0.8, 0.8, 0.8), fuzz: 0.3 }), // Left sphere
    Metal({ albedo: d.vec3f(0.8, 0.6, 0.2), fuzz: 1.0 }), // Right sphere
  ],
}

const canvas = initializeCanvas();

const render = await createScene({
  aspectRatio,
  imageWidth,
  samplesPerPixel,
  samplesPerPass,
  maxBounceDepth,
  world,
});

function frame() {
  render(canvas);
  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);

function initializeCanvas() {
  document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <canvas></canvas>
  `;

  return document.querySelector('canvas') as HTMLCanvasElement;
}
