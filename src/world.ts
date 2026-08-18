import type { d } from "typegpu";

import type { Sphere } from "./sphere";
import type { Lambertian, Metal } from "./material";

export type World = {
  spheres: d.Infer<typeof Sphere>[],
  lambertians: d.Infer<typeof Lambertian>[],
  metals: d.Infer<typeof Metal>[],
};
