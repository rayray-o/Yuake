import {
  FilesetResolver,
  HandLandmarker,
  type HandLandmarkerResult
} from "@mediapipe/tasks-vision";

import {
  HAND_LOST_GRACE_MS,
  HAND_MODEL_URL,
  MAX_HANDS,
  MEDIAPIPE_WASM_URL
} from "./constants";

import {
  calculatePalm,
  calculatePalmNormal
} from "./geometry";

import {
  OneEuroPointFilter
} from "./oneEuro";

import {
  GestureClassifier
} from "./classify";

import type {
  GestureFrame,
  GestureHand,
  Handedness,
  Point3D
} from "./types";

type Point2D = {
  x: number;
  y: number;
};

type RuntimeState = {
  filter: OneEuroPointFilter;
  classifier: GestureClassifier;

  previousCursor: Point2D;
  previousTime: number;

  lastSeen: number;
};

const INDEX_TIP = 8;

/*
 * IMPORTANT:
 *
 * These constants assume the filter operates on
 * PIXEL coordinates, not normalized 0..1 values.
 *
 * beta is a speed-adaptive term: it lowers the
 * filter's smoothing (cutoff goes up) in proportion
 * to how fast the tracked point is moving, so the
 * cursor "catches up" during fast motion instead of
 * lagging behind at a constant smoothing strength.
 *
 * That only works if beta is scaled to match the
 * units of the derivative. On a 0..1 normalized
 * scale, finger velocity is a tiny number
 * (e.g. ~2 "screens per second" during a fast swipe),
 * so a pixel-tuned beta like 0.01 barely moves the
 * cutoff at all -> the filter behaves like a constant
 * low-pass filter regardless of speed -> the cursor
 * always trails behind instead of locking on.
 *
 * Fixing this by filtering in PIXEL space (see
 * convertResult below) lets us use standard,
 * well-tested 1-euro constants.
 */
const FILTER_MIN_CUTOFF = 1.0;
const FILTER_BETA = 0.03;
const FILTER_D_CUTOFF = 1.0;

/*
 * Keep the fast mobile processing pipeline.
 *
 * The visible camera remains full resolution.
 * MediaPipe receives a smaller same-aspect-ratio
 * processing frame.
 */
const MAX_PROCESSING_WIDTH = 480;

let cachedViewport:
  | {
      width: number;
      height: number;
      left: number;
      top: number;
      time: number;
    }
  | null = null;

function clamp(
  value: number,
  min: number,
  max: number
) {
  return Math.max(
    min,
    Math.min(max, value)
  );
}

function clamp01(
  value: number
) {
  return clamp(
    value,
    0,
    1
  );
}

function toPoint3D(
  point: {
    x: number;
    y: number;
    z: number;
  }
): Point3D {
  return {
    x: Number.isFinite(point.x)
      ? point.x
      : 0,

    y: Number.isFinite(point.y)
      ? point.y
      : 0,

    z: Number.isFinite(point.z)
      ? point.z
      : 0
  };
}

function getHandedness(
  result: HandLandmarkerResult,
  index: number
): Handedness {
  const label =
    result.handednesses?.[
      index
    ]?.[0]?.categoryName;

  if (
    label === "Left" ||
    label === "Right"
  ) {
    return label;
  }

  return "Unknown";
}

/*
 * Get the actual DOM rectangle occupied by
 * the video element.
 *
 * IMPORTANT:
 * This is NOT the visible image rectangle.
 *
 * With object-fit: cover, the source image
 * is scaled until it completely covers this
 * rectangle and some source pixels are cropped.
 */
function getViewportRect(
  video: HTMLVideoElement
) {
  const now =
    performance.now();

  const element =
    video.getBoundingClientRect();

  if (
    element.width <= 0 ||
    element.height <= 0
  ) {
    return null;
  }

  if (
    cachedViewport &&
    now -
      cachedViewport.time 
      100 &&
    cachedViewport.width ===
      element.width &&
    cachedViewport.height ===
