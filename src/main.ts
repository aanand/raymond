import './style.css'

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
<canvas></canvas>
`

async function initialize() {
  const canvas = document.querySelector('canvas') as HTMLCanvasElement;
  const adapter = await navigator.gpu?.requestAdapter({});
  const device = await adapter?.requestDevice();
  if (!device) {
    throw new Error("Failed to request device");
  }

  const context = canvas.getContext('2d');
  if (!context) {
    throw new Error("Failed to get canvas context");
  }

  const module = device.createShaderModule({
    label: 'doubling compute module',
    code: /* wgsl */`
      @group(0) @binding(0) var<storage, read> circles: array<u32>; // array of [x, y, radius, x, y, radius, ...]
      @group(0) @binding(1) var<storage, read> width: u32;
      // @group(0) @binding(2) var<storage, read> height: u32;
      @group(0) @binding(2) var<storage, read_write> pixels: array<u32>;

      @compute @workgroup_size(1) fn computeSomething(
        @builtin(global_invocation_id) id: vec3u
      ) {
        let i = id.x;
        let x = i % width;
        let y = i / width;
        let numCircles = arrayLength(&circles);

        pixels[i] = pack4xU8(vec4(0, 0, 0, 255));

        for (var ci = u32(0); ci < numCircles; ci += 3) {
          let cx = circles[ci];
          let cy = circles[ci+1];
          let r  = circles[ci+2];

          let dx = cx - x;
          let dy = cy - y;

          if (dx*dx + dy*dy < r*r) {
            pixels[i] = pack4xU8(vec4(255, 255, 255, 255));
          }
        }
      }
    `,
  });

  const pipeline = device.createComputePipeline({
    label: 'doubling compute pipeline',
    layout: 'auto',
    compute: {
      module,
    },
  });

  const circles = new Uint32Array([
    50, 50, 50,
    150, 50, 50,
  ]);
  const width = canvas.width;
  const height = canvas.height;
  const numPixels = width * height;

  const circlesBuffer = device.createBuffer({
    label: 'circles buffer',
    size: circles.byteLength,
    // @ts-expect-error: GPUBufferUsage not defined
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(circlesBuffer, 0, circles);

  const widthBuffer = device.createBuffer({
    label: 'width buffer',
    size: Uint32Array.BYTES_PER_ELEMENT,
    // @ts-expect-error: GPUBufferUsage not defined
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(widthBuffer, 0, new Uint32Array([width]));

  // const heightBuffer = device.createBuffer({
  //   label: 'height buffer',
  //   size: Uint32Array.BYTES_PER_ELEMENT,
  //   // @ts-expect-error: GPUBufferUsage not defined
  //   usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  // });
  // device.queue.writeBuffer(heightBuffer, 0, new Uint32Array([height]));

  const pixelsBuffer = device.createBuffer({
    label: 'pixels buffer',
    size: Uint32Array.BYTES_PER_ELEMENT * numPixels,
    // @ts-expect-error: GPUBufferUsage not defined
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const resultBuffer = device.createBuffer({
    label: 'result buffer',
    size: Uint32Array.BYTES_PER_ELEMENT * numPixels,
    // @ts-expect-error: GPUBufferUsage not defined
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const bindGroup = device.createBindGroup({
    label: 'bindGroup for work buffer',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: circlesBuffer },
      { binding: 1, resource: widthBuffer },
      // { binding: 2, resource: heightBuffer },
      { binding: 2, resource: pixelsBuffer },
    ],
  });

  const encoder = device.createCommandEncoder({
    label: 'compute encoder',
  });
  const pass = encoder.beginComputePass({
    label: 'compute pass',
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(numPixels);
  pass.end();

  encoder.copyBufferToBuffer(pixelsBuffer, 0, resultBuffer, 0, resultBuffer.size);

  const commandBuffer = encoder.finish();
  device.queue.submit([commandBuffer]);

  // @ts-expect-error: GPUMapMode not defined
  await resultBuffer.mapAsync(GPUMapMode.READ);
  const result = new Uint8ClampedArray(resultBuffer.getMappedRange());

  context.putImageData(new ImageData(result, width, height), 0, 0);
  
  resultBuffer.unmap();
}

initialize();
