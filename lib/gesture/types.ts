export type Point3D = {
  x: number;
  y: number;
  z: number;
};

export type Point2D = {
  x: number;
  y: number;
};

export type FingerName =
  | "thumb"
  | "index"
  | "middle"
  | "ring"
  | "pinky";

export type Handedness =
  | "Left"
  | "Right"
  | "Unknown";

export type Pose =
  | "none"
  | "open"
  | "fist"
  | "point"
  | "pinch";

export type FingerState = {
  extended: boolean;
  touch: boolean;
  distance: number;
  strength: number;
};

export type FingerStates = Record<
  FingerName,
  FingerState
>;

export type GestureHand = {
  id: string;
  handedness: Handedness;

  landmarks: Point3D[];
  worldLandmarks: Point3D[];

  wrist: Point3D;
  palm: Point3D;

  cursor: Point2D;

  velocity: Point2D;
  speed: number;

  pinch: boolean;
  pinchStrength: number;

  pose: Pose;

  fingers: FingerStates;

  palmNormal: Point3D;

  confidence: number;

  visible: boolean;

  lastSeen: number;
};

export type GestureFrame = {
  timestamp: number;

  detected: boolean;

  hands: GestureHand[];

  primaryHand: GestureHand | null;

  frameTime: number;

  processingTime: number;
};

export type TrackerStatus =
  | "idle"
  | "initializing"
  | "ready"
  | "running"
  | "error";
