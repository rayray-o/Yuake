import type {
  Point2D,
  Point3D
} from "./types";

export function distance3D(
  a: Point3D,
  b: Point3D
): number {
  return Math.hypot(
    a.x - b.x,
    a.y - b.y,
    a.z - b.z
  );
}

export function distance2D(
  a: Point2D,
  b: Point2D
): number {
  return Math.hypot(
    a.x - b.x,
    a.y - b.y
  );
}

export function subtract3D(
  a: Point3D,
  b: Point3D
): Point3D {
  return {
    x: a.x - b.x,
    y: a.y - b.y,
    z: a.z - b.z
  };
}

export function add3D(
  a: Point3D,
  b: Point3D
): Point3D {
  return {
    x: a.x + b.x,
    y: a.y + b.y,
    z: a.z + b.z
  };
}

export function multiply3D(
  a: Point3D,
  scalar: number
): Point3D {
  return {
    x: a.x * scalar,
    y: a.y * scalar,
    z: a.z * scalar
  };
}

export function midpoint3D(
  a: Point3D,
  b: Point3D
): Point3D {
  return {
    x: (a.x + b.x) * 0.5,
    y: (a.y + b.y) * 0.5,
    z: (a.z + b.z) * 0.5
  };
}

export function midpoint2D(
  a: Point2D,
  b: Point2D
): Point2D {
  return {
    x: (a.x + b.x) * 0.5,
    y: (a.y + b.y) * 0.5
  };
}

export function cross3D(
  a: Point3D,
  b: Point3D
): Point3D {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x
  };
}

export function normalize3D(
  vector: Point3D
): Point3D {
  const length = Math.hypot(
    vector.x,
    vector.y,
    vector.z
  );

  if (length < 0.000001) {
    return {
      x: 0,
      y: 0,
      z: 0
    };
  }

  return {
    x: vector.x / length,
    y: vector.y / length,
    z: vector.z / length
  };
}

export function clamp(
  value: number,
  min: number,
  max: number
): number {
  return Math.max(
    min,
    Math.min(max, value)
  );
}

/*
 * Converts thumb/index distance into a
 * hand-relative measurement.
 *
 * This means the same physical pinch can
 * work when the hand is near or far from
 * the camera.
 */
export function normalizedFingerDistance(
  distance: number,
  handScale: number
): number {
  return distance / Math.max(handScale, 0.0001);
}

export function calculateHandScale(
  landmarks: Point3D[]
): number {
  if (
    landmarks.length < 10
  ) {
    return 0.1;
  }

  /*
   * Wrist -> middle MCP.
   *
   * This is stable enough to act as a
   * relative hand-size reference.
   */
  return Math.max(
    distance3D(
      landmarks[0],
      landmarks[9]
    ),
    0.0001
  );
}

export function calculateFingerBend(
  landmarks: Point3D[],
  mcp: number,
  pip: number,
  dip: number,
  tip: number
): number {
  const first =
    distance3D(
      landmarks[mcp],
      landmarks[pip]
    );

  const second =
    distance3D(
      landmarks[pip],
      landmarks[dip]
    );

  const third =
    distance3D(
      landmarks[dip],
      landmarks[tip]
    );

  const direct =
    distance3D(
      landmarks[mcp],
      landmarks[tip]
    );

  const chain =
    first +
    second +
    third;

  if (chain < 0.000001) {
    return 1;
  }

  return clamp(
    1 - direct / chain,
    0,
    1
  );
}

export function calculatePalm(
  landmarks: Point3D[]
): Point3D {
  const indices = [
    0,
    5,
    9,
    13,
    17
  ];

  let x = 0;
  let y = 0;
  let z = 0;

  for (const index of indices) {
    const point =
      landmarks[index];

    x += point.x;
    y += point.y;
    z += point.z;
  }

  const count =
    indices.length;

  return {
    x: x / count,
    y: y / count,
    z: z / count
  };
}

export function calculatePalmNormal(
  landmarks: Point3D[]
): Point3D {
  if (landmarks.length < 18) {
    return {
      x: 0,
      y: 0,
      z: 0
    };
  }

  const across =
    subtract3D(
      landmarks[17],
      landmarks[5]
    );

  const upward =
    subtract3D(
      landmarks[9],
      landmarks[0]
    );

  return normalize3D(
    cross3D(
      across,
      upward
    )
  );
}
