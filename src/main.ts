import { d } from 'typegpu';
import { clamp, cos, cross, dot, mix, sin } from 'typegpu/std';

import { Sphere } from './sphere';
import { createRenderer } from './renderer';

import type { World } from './world';
import { Dielectric, Lambertian, MATERIAL_DIELECTRIC, MATERIAL_LAMBERTIAN, MATERIAL_METAL, Metal } from './material';
import type { CameraProps } from './camera';

import './style.css';

const aspectRatio = 16.0/9.0;
const imageWidth = 1000;
const samplesPerPixel = 2000;
const samplesPerPass = 5;
const maxBounceDepth = 10;

const vfov = 40;
const lookAt = d.vec3f(0, 0, -1);
const vup = d.vec3f(0, 1, 0);

const cameraDistance = 3.5;
const defocusAngle = 2;
const focusDistance = cameraDistance;

const calculateLookFrom = (lookAt: d.v3f, cameraDistance: number, azimuth: number, elevation: number) => {
  const rotateAxis = (v: d.v3f, axis: d.v3f, angleRadians: number) =>
    mix(axis.mul(dot(axis, v)), v, cos(angleRadians)).add(cross(axis, v).mul(sin(angleRadians)));

  let cameraDirection = d.vec3f(0, 0, cameraDistance);
  cameraDirection = rotateAxis(cameraDirection, d.vec3f(1, 0, 0), -elevation);
  cameraDirection = rotateAxis(cameraDirection, d.vec3f(0, 1, 0), -azimuth);

  return lookAt.add(cameraDirection);
}

// (0, 0) means (lookAt - lookFrom) = (0, 0, cameraDistance), i.e. "front" view.
// Positive azimuth moves the camera LEFT
// Positive elevation moves the camera UP
let azimuth = Math.PI/4;
let elevation = Math.PI/8;
let lookFrom = calculateLookFrom(lookAt, cameraDistance, azimuth, elevation);

const getCameraProps = (): CameraProps => ({
  vfov,
  lookAt,
  lookFrom,
  vup,
  defocusAngle,
  focusDistance,
});

const world: World = {
  spheres: [
    Sphere({
      center: d.vec3f(0, -100.5, -1),
      radius: 100,
      materialType: MATERIAL_LAMBERTIAN,
      materialIndex: 0,
    }),
    Sphere({
      center: d.vec3f(0, 0, -1),
      radius:  0.5,
      materialType: MATERIAL_METAL,
      materialIndex: 0,
    }),
    Sphere({
      center: d.vec3f(-1.25, 0, -1),
      radius: 0.5,
      materialType: MATERIAL_DIELECTRIC,
      materialIndex: 0,
    }),
    Sphere({
      center: d.vec3f(-1.25, 0, -1),
      radius: 0.4,
      materialType: MATERIAL_DIELECTRIC,
      materialIndex: 1,
    }),
    Sphere({
      center: d.vec3f(1.25, 0, -1),
      radius: 0.5,
      materialType: MATERIAL_LAMBERTIAN,
      materialIndex: 1,
    }),
  ],
  lambertians: [
    Lambertian({ albedo: d.vec3f(0.8, 0.8, 0.0) }), // Ground
    Lambertian({ albedo: d.vec3f(0.1, 0.2, 0.5) }), // Right sphere
  ],
  metals: [
    Metal({ albedo: d.vec3f(0.8, 0.6, 0.4), fuzz: 0.1 }), // Middle sphere
  ],
  dielectrics: [
    Dielectric({ refractionIndex: 1.5 }),     // Left sphere (outside)
    Dielectric({ refractionIndex: 1.0/1.5 }), // Left sphere (inside)
  ]
}

const canvas = initializeCanvas();

const { render, setCameraProps } = await createRenderer({
  aspectRatio,
  imageWidth,

  cameraProps: getCameraProps(),

  samplesPerPixel,
  samplesPerPass,
  maxBounceDepth,

  world,
});

const updateCamera = () => {
  lookFrom = calculateLookFrom(lookAt, cameraDistance, azimuth, elevation);
  setCameraProps(getCameraProps());
};

updateCamera();

// Radians per pixel
const movementSpeed = Math.PI / 256;

let isMoving = false;
canvas.addEventListener('mousedown', () => { isMoving = true });
window.addEventListener('mouseup', () => { isMoving = false });
canvas.addEventListener('mousemove', event => {
  if (isMoving) {
    azimuth = (azimuth + event.movementX * movementSpeed) % (Math.PI * 2);
    elevation = clamp(elevation + event.movementY * movementSpeed, Math.PI * -0.03, Math.PI * 0.3);
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
