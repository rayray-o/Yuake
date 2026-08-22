export const MEDIAPIPE_VERSION = "0.10.35";

export const MEDIAPIPE_WASM_URL =
  `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;

export const HAND_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

export const MAX_HANDS = 2;

export const DETECTION_CONFIDENCE = 0.55;

export const PRESENCE_CONFIDENCE = 0.55;

export const TRACKING_CONFIDENCE = 0.55;

/*
 * Pinch thresholds are intentionally separated.
 *
 * Enter pinch:
 *     normalized distance < PINCH_START
 *
 * Leave pinch:
 *     normalized distance > PINCH_END
 *
 * This is hysteresis and prevents rapid
 * pinch / release flickering.
 */
export const PINCH_START = 0.34;

export const PINCH_END = 0.46;

/*
 * How long we keep a hand alive after
 * MediaPipe temporarily loses it.
 */
export const HAND_LOST_GRACE_MS = 100;

/*
 * One-Euro filter configuration.
 */
export const FILTER_MIN_CUTOFF = 1.15;

export const FILTER_BETA = 0.035;

export const FILTER_D_CUTOFF = 1.0;

/*
 * Finger contact uses the same principle
 * as pinch hysteresis.
 */
export const FINGER_TOUCH_START = 0.36;

export const FINGER_TOUCH_END = 0.50;

/*
 * A hand with at least this many extended
 * fingers is considered open.
 */
export const OPEN_FINGER_COUNT = 3;

/*
 * Video processing is deliberately bounded.
 * We don't need to process every camera frame
 * if the device is producing more than we can use.
 */
export const MAX_PROCESSING_FPS = 30;
