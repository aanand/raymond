import { d } from 'typegpu';

import { createRenderer } from './renderer';
import type { CameraProps } from './camera';
import { world } from './world';
import { rotateAxis } from './utils';
import { trackMovement } from './movement';

import './style.css';

const aspectRatio = 1;
const imageWidth = 600;
const samplesPerPixel = 2000;
const samplesPerPass = 1;
const maxBounceDepth = 5;

const vfov = 40;
const lookAt = d.vec3f(0, 1, 0);
const vup = d.vec3f(0, 1, 0);

const cameraDistance = 7;
const defocusAngle = 2;
const focusDistance = cameraDistance;

// (0, 0) means (lookAt - lookFrom) = (0, 0, cameraDistance), i.e. "front" view.
// Positive azimuth moves the camera LEFT
// Positive elevation moves the camera UP
const startingAzimuth = 0;
const startingElevation = Math.PI/16;

// Radians per pixel
const movementSpeed = Math.PI / 256;

const calculateLookFrom = (azimuth: number, elevation: number) => {
  let cameraDirection = d.vec3f(0, 0, cameraDistance);
  cameraDirection = rotateAxis(cameraDirection, d.vec3f(1, 0, 0), -elevation);
  cameraDirection = rotateAxis(cameraDirection, d.vec3f(0, 1, 0), -azimuth);

  return lookAt.add(cameraDirection);
}

let lookFrom = calculateLookFrom(startingAzimuth, startingElevation);

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
  renderer.setCameraProps(getCameraProps());
};

updateCamera();

const canvas = document.querySelector('canvas') as HTMLCanvasElement;
const pre = document.querySelector('pre') as HTMLElement;
const debug = Array.from(new URLSearchParams(window.location.search).keys()).includes('debug');

trackMovement({
  startingAzimuth,
  startingElevation,
  movementSpeed,
  targetElement: canvas,
  onStartMoving: () => renderer.setIsLoResMode(true),
  onStopMoving: () => renderer.setIsLoResMode(false),
  onMove: (azimuth, elevation) => {
    lookFrom = calculateLookFrom(azimuth, elevation);
    updateCamera();
  },
});

let frameCount = 0;
let fps = 0;

setInterval(() => {
  fps = frameCount;
  frameCount = 0;
}, 1000);

function frame() {
  renderer.render(canvas);
  frameCount++;

  if (debug) {
    const renderTimes = renderer.getRenderTime();
    pre.textContent = [
      `render: ${renderTimes.render.toFixed(0)}`,
      `draw: ${renderTimes.draw.toFixed(0)}`,
      `fps: ${fps.toFixed(0)}`
    ].join(', ');
  }

  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
