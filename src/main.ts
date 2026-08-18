import { d } from 'typegpu';

import { Sphere } from './sphere';
import { render } from './camera';

import './style.css';
import type { World } from './world';
import { Lambertian, MATERIAL_LAMBERTIAN, MATERIAL_METAL, Metal } from './material';

const aspectRatio = d.f32(16.0/9.0);
const imageWidth = d.u32(800);
const samplesPerPixel = d.u32(40);
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
    Metal({ albedo: d.vec3f(0.8, 0.8, 0.8) }), // Left sphere
    Metal({ albedo: d.vec3f(0.8, 0.6, 0.2) }), // Right sphere
  ],
}

const canvas = initializeCanvas();

const draw = () =>
  render({ aspectRatio, imageWidth, samplesPerPixel, maxBounceDepth, world, canvas });

canvas.addEventListener('click', draw);
draw();

function initializeCanvas() {
  document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <canvas></canvas>
  `;

  return document.querySelector('canvas') as HTMLCanvasElement;
}
