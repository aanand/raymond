import { d } from 'typegpu';

import { Sphere } from './sphere';
import { render } from './camera';

import './style.css';

const aspectRatio = d.f32(16.0/9.0);
const imageWidth = d.u32(800);

const world = d.arrayOf(Sphere, 2)([
  Sphere({ center: d.vec3f(0,    0,   -1), radius: -0.5 }),
  Sphere({ center: d.vec3f(0, -100.5, -1), radius:  100 }),
]);

render({ aspectRatio, imageWidth, world, canvas: initializeCanvas() });

function initializeCanvas() {
  document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <canvas></canvas>
  `;

  return document.querySelector('canvas') as HTMLCanvasElement;
}
