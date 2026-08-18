import { d } from 'typegpu';

import { Sphere } from './sphere';
import { createScene } from './camera';

import type { World } from './world';
import { Dielectric, Lambertian, MATERIAL_DIELECTRIC, MATERIAL_LAMBERTIAN, MATERIAL_METAL, Metal } from './material';

import './style.css';
import { clamp, cos, cross, dot, mix, sin } from 'typegpu/std';

const aspectRatio = 16.0/9.0;
const imageWidth = 800;
const vfov = 40;
const lookAt = d.vec3f(0, 0, -1);
const vup = d.vec3f(0, 1, 0);
const samplesPerPixel = 2000;
const samplesPerPass = 5;
const maxBounceDepth = 10;

const calculateLookFrom = (lookAt: d.v3f, cameraDistance: number, azimuth: number, elevation: number) => {
  const rotateAxis = (v: d.v3f, axis: d.v3f, angleRadians: number) =>
    mix(axis.mul(dot(axis, v)), v, cos(angleRadians)).add(cross(axis, v).mul(sin(angleRadians)));

  let cameraDirection = d.vec3f(0, 0, cameraDistance);
  cameraDirection = rotateAxis(cameraDirection, d.vec3f(1, 0, 0), -elevation);
  cameraDirection = rotateAxis(cameraDirection, d.vec3f(0, 1, 0), -azimuth);

  return lookAt.add(cameraDirection);
}

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
      materialType: MATERIAL_DIELECTRIC,
      materialIndex: 0,
    }),
    Sphere({
      center: d.vec3f(-1, 0, -1),
      radius: 0.4,
      materialType: MATERIAL_DIELECTRIC,
      materialIndex: 1,
    }),
    Sphere({
      center: d.vec3f(1, 0, -1),
      radius: 0.5,
      materialType: MATERIAL_METAL,
      materialIndex: 0,
    }),
  ],
  lambertians: [
    Lambertian({ albedo: d.vec3f(0.8, 0.8, 0.0) }), // Ground
    Lambertian({ albedo: d.vec3f(0.1, 0.2, 0.5) }), // Middle sphere
  ],
  metals: [
    Metal({ albedo: d.vec3f(0.8, 0.6, 0.2), fuzz: 1.0 }), // Right sphere
  ],
  dielectrics: [
    Dielectric({ refractionIndex: 1.5 }),     // Left sphere (outside)
    Dielectric({ refractionIndex: 1.0/1.5 }), // Left sphere (inside)
  ]
}

const canvas = initializeCanvas();

// (0, 0) means (lookAt - lookFrom) = (0, 0, cameraDistance)
// Positive azimuth moves the camera LEFT
// Positive elevation moves the camera UP
const cameraDistance = 3.5;
let azimuth = Math.PI/4;
let elevation = Math.PI/8;
let lookFrom = calculateLookFrom(lookAt, cameraDistance, azimuth, elevation);

const { render, setCameraProps } = await createScene({
  aspectRatio,
  imageWidth,

  cameraProps: {
    vfov,
    lookFrom,
    lookAt,
    vup,
  },

  samplesPerPixel,
  samplesPerPass,
  maxBounceDepth,

  world,
});

const updateCamera = () => {
  lookFrom = calculateLookFrom(lookAt, cameraDistance, azimuth, elevation);
  setCameraProps({ vfov, lookFrom, lookAt, vup });
};

updateCamera();

// Radians per pixel
const movementSpeed = Math.PI / 256;

let isMoving = false;
canvas.addEventListener('mousedown', () => { isMoving = true });
canvas.addEventListener('mouseup', () => { isMoving = false });
canvas.addEventListener('mousemove', event => {
  if (isMoving) {
    azimuth = (azimuth + event.movementX * movementSpeed) % (Math.PI * 2);
    elevation = clamp(elevation + event.movementY * movementSpeed, 0, Math.PI * 0.4);
    updateCamera();
  }
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
