import { d } from 'typegpu';
import { clamp } from 'typegpu/std';

import { createRenderer } from './renderer';
import type { CameraProps } from './camera';
import { world } from './world';
import { rotateAxis } from './utils';

import './style.css';

const aspectRatio = 1;
const imageWidth = 1000;
const samplesPerPixel = 2000;
const samplesPerPass = 1;
const maxBounceDepth = 5;

const vfov = 40;
const lookAt = d.vec3f(0, 1, 0);
const vup = d.vec3f(0, 1, 0);

const cameraDistance = 7;
const defocusAngle = 2;
const focusDistance = cameraDistance;

const calculateLookFrom = (lookAt: d.v3f, cameraDistance: number, azimuth: number, elevation: number) => {
  let cameraDirection = d.vec3f(0, 0, cameraDistance);
  cameraDirection = rotateAxis(cameraDirection, d.vec3f(1, 0, 0), -elevation);
  cameraDirection = rotateAxis(cameraDirection, d.vec3f(0, 1, 0), -azimuth);

  return lookAt.add(cameraDirection);
}

// (0, 0) means (lookAt - lookFrom) = (0, 0, cameraDistance), i.e. "front" view.
// Positive azimuth moves the camera LEFT
// Positive elevation moves the camera UP
let azimuth = 0;
let elevation = Math.PI/16;
let lookFrom = calculateLookFrom(lookAt, cameraDistance, azimuth, elevation);

const getCameraProps = (): CameraProps => ({
  vfov,
  lookAt,
  lookFrom,
  vup,
  defocusAngle,
  focusDistance,
});

const renderer = await createRenderer({
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
  renderer.setCameraProps(getCameraProps());
};

updateCamera();

const canvas = document.querySelector('canvas') as HTMLCanvasElement;
const pre = document.querySelector('pre') as HTMLElement;
const debug = Array.from(new URLSearchParams(window.location.search).keys()).includes('debug');

// Radians per pixel
const movementSpeed = Math.PI / 256;

let isMoving = false;
let lastX = 0;
let lastY = 0;

const startMoving = (pageX: number, pageY: number): void => {
  isMoving = true;
  lastX = pageX;
  lastY = pageY; 
}

const stopMoving = () => {
  isMoving = false;
  lastX = 0;
  lastY = 0;
};

const moveTo = (pageX: number, pageY: number): void => {
  if (!isMoving) { return; }

  const deltaX = pageX - lastX;
  const deltaY = pageY - lastY;

  azimuth = (azimuth + deltaX * movementSpeed) % (Math.PI * 2);
  elevation = clamp(elevation + deltaY * movementSpeed, 0, Math.PI * 0.3);

  lastX = pageX;
  lastY = pageY;

  updateCamera();
}

canvas.addEventListener('mousedown', event => startMoving(event.pageX, event.pageY));
canvas.addEventListener('touchstart', event => startMoving(event.changedTouches[0].pageX, event.changedTouches[0].pageY));

window.addEventListener('mouseup', stopMoving);
window.addEventListener('touchend', stopMoving)

window.addEventListener('mousemove', event => {
  event.preventDefault();
  moveTo(event.pageX, event.pageY);
});
window.addEventListener('touchmove', event => {
  event.preventDefault();
  moveTo(event.changedTouches[0].pageX, event.changedTouches[0].pageY);
});

function frame() {
  renderer.render(canvas);

  if (debug) {
    const renderTimes = renderer.getRenderTime();
    pre.textContent = `render: ${renderTimes.render.toString()}, draw: ${renderTimes.draw.toString()}`;
  }

  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
