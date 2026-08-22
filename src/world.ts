import { d } from "typegpu";
import { length, radians } from "typegpu/std";

import { sphere, Sphere } from "./sphere";
import { MATERIAL_DIELECTRIC, MATERIAL_LAMBERTIAN, MATERIAL_METAL, Dielectric, Lambertian, Metal } from "./material";
import { MaterialReference } from "./types";
import { rotateAxis } from "./utils";

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
  spheres.push(sphere({ center, radius, material }));
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

const spread = 1.8;

for (let a = -11; a < 11; a++) {
  for (let b = -11; b < 11; b++) {
    const chooseMat = Math.random();
    const center = d.vec3f(a + spread*Math.random(), 0.2, b + spread*Math.random());

    if (length(center.sub(d.vec3f(0, 0, 0))) > spread) {
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

const bigSphereRadius = 0.75;
const back = d.vec3f(0, 0, -bigSphereRadius*1.16);
const left = rotateAxis(back, d.vec3f(0, 1, 0), radians(-360/3));
const right = rotateAxis(back, d.vec3f(0, 1, 0), radians(360/3));
const center = d.vec3f(0, bigSphereRadius, 0);

makeSphere(center.add(left), bigSphereRadius, makeDielectric({ refractionIndex: 1.5 }));
makeSphere(center.add(back), bigSphereRadius, makeLambertian({ albedo: d.vec3f(0.25, 0.45, 0.55) }));
makeSphere(center.add(right), bigSphereRadius, makeMetal({ albedo: d.vec3f(0.7, 0.6, 0.5), fuzz: 0 }));

export const world: World = {
  spheres,
  lambertians,
  metals,
  dielectrics,
};
