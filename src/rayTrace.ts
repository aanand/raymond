import tgpu, { d } from "typegpu";
import { normalize } from "typegpu/std";

import { Lambertian, Metal, Dielectric, MATERIAL_LAMBERTIAN, scatterLambertian, MATERIAL_METAL, scatterMetal, MATERIAL_DIELECTRIC, scatterDielectric } from "./material";
import { Sphere, hitSphere } from "./sphere";
import { Bounce, Ray, Interval, HitRecord, didNotHit, interval, INF, didNotBounce } from "./utils";
import type { World } from "./world";

const RayTraceResult = d.struct({
  color: d.vec3f,
  bounce: Bounce,
});

export function buildRayTraceFunction(world: World) {
  const spheres = tgpu.const(d.arrayOf(Sphere, world.spheres.length), world.spheres);
  const lambertians = tgpu.const(d.arrayOf(Lambertian, world.lambertians.length), world.lambertians);
  const metals = tgpu.const(d.arrayOf(Metal, world.metals.length), world.metals);
  const dielectrics = tgpu.const(d.arrayOf(Dielectric, world.dielectrics.length), world.dielectrics);

  const hitWorld = tgpu.fn([Ray, Interval], HitRecord)((ray, rayT) => {
    let hitRecord = didNotHit();
    let closestSoFar = rayT.max;

    for (let i = 0; i < spheres.$.length; i++) {
      const sphereHit = hitSphere(spheres.$[i], ray, interval(rayT.min, closestSoFar));
      if (sphereHit.isHit) {
        hitRecord = HitRecord(sphereHit);
        closestSoFar = sphereHit.t;
      }
    }

    return hitRecord;
  });

  return tgpu.fn([Ray, d.f32], RayTraceResult)((ray, randomFloat) => {
    const hitRecord = hitWorld(ray, interval(0.001, INF));

    if (hitRecord.isHit) {
      let bounce = didNotBounce();
      if (hitRecord.materialType === MATERIAL_LAMBERTIAN) {
        const lambertian = lambertians.$[hitRecord.materialIndex];
        bounce = scatterLambertian(lambertian, hitRecord, randomFloat);
      } else if (hitRecord.materialType === MATERIAL_METAL) {
        const metal = metals.$[hitRecord.materialIndex];
        bounce = scatterMetal(metal, ray, hitRecord, randomFloat);
      } else if (hitRecord.materialType === MATERIAL_DIELECTRIC) {
        const dielectric = dielectrics.$[hitRecord.materialIndex];
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
