import { clamp } from "typegpu/std";

export function trackMovement({
  startingAzimuth,
  startingElevation,
  movementSpeed,
  targetElement,
  onStartMoving,
  onStopMoving,
  onMove,
}: {
  startingAzimuth: number,
  startingElevation: number,
  movementSpeed: number,
  targetElement: HTMLElement,
  onStartMoving: () => void,
  onStopMoving: () => void,
  onMove: (newAzimuth: number, newElevation: number) => void,
}) {
  let azimuth = startingAzimuth;
  let elevation = startingElevation;
  let isMoving = false;
  let lastX = 0;
  let lastY = 0;

  const startMoving = (pageX: number, pageY: number): void => {
    onStartMoving();

    isMoving = true;
    lastX = pageX;
    lastY = pageY; 
  }

  const stopMoving = () => {
    onStopMoving();

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

    onMove(azimuth, elevation);
  }

  targetElement.addEventListener('mousedown', event => {
    if (event.button === 0) {
      startMoving(event.pageX, event.pageY);
    }
  });
  targetElement.addEventListener('touchstart', event => startMoving(event.changedTouches[0].pageX, event.changedTouches[0].pageY));

  window.addEventListener('mouseup', event => {
    if (event.button === 0) {
      stopMoving();
    }
  });
  window.addEventListener('touchend', stopMoving)

  window.addEventListener('mousemove', event => {
    event.preventDefault();
    moveTo(event.pageX, event.pageY);
  });
  window.addEventListener('touchmove', event => {
    event.preventDefault();
    moveTo(event.changedTouches[0].pageX, event.changedTouches[0].pageY);
  });
}
