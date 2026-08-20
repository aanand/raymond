import tgpu, { d } from "typegpu";
import { normalize } from "typegpu/std";

import { Lambertian, Metal, Dielectric, MATERIAL_LAMBERTIAN, scatterLambertian, MATERIAL_METAL, scatterMetal, MATERIAL_DIELECTRIC, scatterDielectric } from "./material";
import { Sphere, hitSphere } from "./sphere";
import { Bounce, Ray, Interval, didNotHit, interval, INF, didNotBounce } from "./utils";
import { HitRecord } from "./types";
import type { World } from "./world";
import { BVHNode, makeBVH } from "./bvh";
import { hitAABB } from "./aabb";

export const RayTraceResult = d.struct({
  color: d.vec3f,
  bounce: Bounce,
});

export function buildRayTraceFunction(world: World) {
  const spheres = tgpu.const(d.arrayOf(Sphere, world.spheres.length), world.spheres);

  const { nodes, depth } = makeBVH(world.spheres);
  const bvh = tgpu.const(d.arrayOf(BVHNode, nodes.length), nodes);

  const lambertians = tgpu.const(d.arrayOf(Lambertian, world.lambertians.length), world.lambertians);
  const metals = tgpu.const(d.arrayOf(Metal, world.metals.length), world.metals);
  const dielectrics = tgpu.const(d.arrayOf(Dielectric, world.dielectrics.length), world.dielectrics);

  const hitWorld = tgpu.fn([Ray, Interval], HitRecord)((ray, rayT) => {
    const stack = d.arrayOf(d.u32, depth)();
    stack[0] = 0;
    let stackHead = 0;

    let hitRecord = didNotHit();
    let closestSoFar = rayT.max;

    while (stackHead >= 0) {
      const nodeIndex = stack[stackHead];
      stackHead--;

      const node = bvh.$[nodeIndex];
      // @ts-expect-error: node.bbox is readonly
      const hit = hitAABB(node.bbox, ray, interval(rayT.min, closestSoFar));

      if (!hit) {
        continue;
      }

      if (node.isLeaf) {
        const object = spheres.$[node.objectIndex];
        // @ts-expect-error: object is readonly
        const sphereHit = hitSphere(object, ray, interval(rayT.min, closestSoFar));
        if (sphereHit.isHit) {
          hitRecord = HitRecord(sphereHit);
          closestSoFar = sphereHit.t;
        }
        continue;
      }

      stackHead++;
      stack[stackHead] = node.leftIndex;

      stackHead++;
      stack[stackHead] = node.rightIndex;
    }

    return hitRecord;
  });

  return tgpu.fn([Ray, d.f32], RayTraceResult)((ray, randomFloat) => {
    const hitRecord = hitWorld(ray, interval(0.001, INF));

    if (hitRecord.isHit) {
      let bounce = didNotBounce();
      if (hitRecord.material.kind === MATERIAL_LAMBERTIAN) {
        const lambertian = lambertians.$[hitRecord.material.index];
        bounce = scatterLambertian(lambertian, hitRecord, randomFloat);
      } else if (hitRecord.material.kind === MATERIAL_METAL) {
        const metal = metals.$[hitRecord.material.index];
        bounce = scatterMetal(metal, ray, hitRecord, randomFloat);
      } else if (hitRecord.material.kind === MATERIAL_DIELECTRIC) {
        const dielectric = dielectrics.$[hitRecord.material.index];
        bounce = scatterDielectric(dielectric, ray, hitRecord, randomFloat);
      }

      let color = d.vec3f(0, 0, 0);
      if (d.bool(bounce.didBounce)) {
        color = d.vec3f(bounce.attenuation);
      }

      return RayTraceResult({ color, bounce });
    } else {
      const unitDirection = normalize(ray.direction);
      const a = 0.5 * (unitDirection.y + 1.0);
      return RayTraceResult({
        color: d.vec3f(1, 1, 1).mul(1 - a).add(d.vec3f(0.5, 0.7, 1.0).mul(a)),
        bounce: didNotBounce(),
      });
    }
  });
}
