import { d } from "typegpu";

import { AABB, combineAABBs } from "./aabb";
import type { Sphere } from "./sphere";

type Tree = {
  isLeaf: false,
  depth: number,
  bbox: d.Infer<typeof AABB>,
  left: Node,
  right: Node,
};

type Leaf = {
  isLeaf: true,
  depth: number,
  bbox: d.Infer<typeof AABB>,
  object: d.Infer<typeof Sphere>,
}

type Node = Tree | Leaf;

const makeComparator = (axis: number) =>
  (a: d.Infer<typeof Sphere>, b: d.Infer<typeof Sphere>) =>
    a.bbox[axis].min - b.bbox[axis].min;

function makeTree(
  objects: d.Infer<typeof Sphere>[],
): Node {
  if (objects.length === 1) {
    return {
      isLeaf: true,
      depth: 1,
      bbox: objects[0].bbox,
      object: objects[0],
    };
  } else if (objects.length === 2) {
    return {
      isLeaf: false,
      bbox: combineAABBs(objects[0].bbox, objects[1].bbox),
      depth: 2,
      left: {
        isLeaf: true,
        depth: 1,
        bbox: objects[0].bbox,
        object: objects[0],
      },
      right: {
        isLeaf: true,
        depth: 1,
        bbox: objects[1].bbox,
        object: objects[1]
      },
    };
  } else {
    const axis = Math.floor(Math.random() * 3);
    const sorted = Array.from(objects).sort(makeComparator(axis));
    const mid = Math.floor(objects.length/2);

    const left = makeTree(sorted.slice(0, mid));
    const right = makeTree(sorted.slice(mid));

    return {
      isLeaf: false,
      depth: Math.max(left.depth, right.depth) + 1,
      bbox: combineAABBs(left.bbox, right.bbox),
      left,
      right,
    };
  }
}

export const BVHNode = d.struct({
  isLeaf: d.bool,
  bbox: AABB,

  // Only used by leaf nodes
  objectIndex: d.u32,

  // Only used by non-leaf nodes
  leftIndex: d.u32,
  rightIndex: d.u32,
});

const serialize = (
  root: Node,
  objects: d.Infer<typeof Sphere>[],
  offset: number,
): d.Infer<typeof BVHNode>[] => {
  if (root.isLeaf) {
    return [
      BVHNode({
        isLeaf: true,
        bbox: root.bbox,
        objectIndex: objects.indexOf(root.object),
        leftIndex: 0,
        rightIndex: 0,
      })
    ]
  }
  
  const leftIndex = offset + 1;
  const leftList = serialize(root.left, objects, leftIndex);
  const rightIndex = offset + 1 + leftList.length;
  const rightList = serialize(root.right, objects, rightIndex);

  return [
    {
      isLeaf: false,
      bbox: root.bbox,
      objectIndex: 0,
      leftIndex,
      rightIndex,
    },
    ...leftList,
    ...rightList,
  ];
}

export const makeBVH = (objects: d.Infer<typeof Sphere>[]): {
  nodes: d.Infer<typeof BVHNode>[],
  depth: number,
} => {
  const tree = makeTree(objects);
  return {
    nodes: serialize(tree, objects, 0),
    depth: tree.depth,
  };
}