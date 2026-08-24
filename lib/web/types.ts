import type { GestureHand, Point2D, Point3D } from "../gesture/types";

export type WebState = "idle" | "armed" | "firing" | "attached" | "retracting";

export type WebTarget = {
  point: Point2D;
  depth: number;
  confidence: number;
  normal: Point3D;
};

export type WebSample = {
  x: number;
  y: number;
  z: number;
};

export type WebDiagnostics = {
  poseConfidence: number;
  targetConfidence: number;
  speed: number;
  distance: number;
  tension: number;
  state: WebState;
};

export type WebConfig = {
  maxRangePx: number;
  launchSpeedPxPerSecond: number;
  retractSpeedPxPerSecond: number;
  drag: number;
  spring: number;
  damping: number;
};

export type WebUpdateInput = {
  hand: GestureHand | null;
  video: HTMLVideoElement | null;
  now: number;
  viewportWidth: number;
  viewportHeight: number;
};
