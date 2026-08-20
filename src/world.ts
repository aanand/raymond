import { d } from "typegpu";
import { length } from "typegpu/std";

import { Sphere } from "./sphere";
import { MATERIAL_DIELECTRIC, MATERIAL_LAMBERTIAN, MATERIAL_METAL, Dielectric, Lambertian, Metal } from "./material";
import { MaterialReference } from "./types";

export type World = {
  spheres: d.Infer<typeof Sphere>[],
  lambertians: d.Infer<typeof Lambertian>[],
  metals: d.Infer<typeof Metal>[],
  dielectrics: d.Infer<typeof Dielectric>[],
};

const spheres: d.Infer<typeof Sphere>[] = [];
const lambertians: d.Infer<typeof Lambertian>[] = [];
const metals: d.Infer<typeof Metal>[] = [];
const dielectrics: d.Infer<typeof Dielectric>[] = [];

const makeLambertian = (props: d.Infer<typeof Lambertian>): d.Infer<typeof MaterialReference> => {
  lambertians.push(Lambertian(props));
  return { kind: MATERIAL_LAMBERTIAN, index: lambertians.length - 1 };
}

const makeMetal = (props: d.Infer<typeof Metal>): d.Infer<typeof MaterialReference> => {
  metals.push(Metal(props));
  return { kind: MATERIAL_METAL, index: metals.length - 1 };
}

const makeDielectric = (props: d.Infer<typeof Dielectric>): d.Infer<typeof MaterialReference> => {
  dielectrics.push(Dielectric(props));
  return { kind: MATERIAL_DIELECTRIC, index: dielectrics.length - 1};
}

const makeSphere = (center: d.v3f, radius: number, material: d.Infer<typeof MaterialReference>) => {
  spheres.push(Sphere({ center, radius, material }));
}

const randomColor = (min = 0, max = 1): d.v3f =>
  d.vec3f(
    min + Math.random() * (max - min),
    min + Math.random() * (max - min),
    min + Math.random() * (max - min),
  );

const groundMaterial = makeLambertian({ albedo: d.vec3f(0.5, 0.5, 0.5) });
makeSphere(d.vec3f(0, -1000, 0), 1000, groundMaterial);

const glass = makeDielectric({ refractionIndex: 0.5 });

for (let a = -11; a < 11; a++) {
  for (let b = -11; b < 11; b++) {
    const chooseMat = Math.random();
    const center = d.vec3f(a + 0.9*Math.random(), 0.2, b + 0.9*Math.random());

    if (length(center.sub(d.vec3f(4, 0.2, 0))) > 0.9) {
      let sphereMaterial: d.Infer<typeof MaterialReference>;

      if (chooseMat < 0.8) {
        const albedo = randomColor().mul(randomColor());
        sphereMaterial = makeLambertian({ albedo });
      } else if (chooseMat < 0.95) {
        const albedo = randomColor(0.5, 1);
        const fuzz = 0.5 * Math.random();
        sphereMaterial = makeMetal({ albedo, fuzz });
      } else {
        sphereMaterial = glass;
      }

      makeSphere(center, 0.2, sphereMaterial);
    }
  }
}

export const world: World = {
  spheres,
  lambertians,
  metals,
  dielectrics,
};
