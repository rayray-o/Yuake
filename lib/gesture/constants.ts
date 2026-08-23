export const MEDIAPIPE_VERSION = "1.0.0";

export const MEDIAPIPE_WASM_URL =
  `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;

export const HAND_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

export const MAX_HANDS = 2;

export const DETECTION_CONFIDENCE = 0.55;
export const PRESENCE_CONFIDENCE = 0.55;
export const TRACKING_CONFIDENCE = 0.55;

export const PINCH_START = 0.34;
export const PINCH_END = 0.46;

export const HAND_LOST_GRACE_MS = 100;

export const FILTER_MIN_CUTOFF = 1.15;
export const FILTER_BETA = 0.035;
export const FILTER_D_CUTOFF = 1.0;

export const FINGER_TOUCH_START = 0.36;
export const FINGER_TOUCH_END = 0.50;

export const OPEN_FINGER_COUNT = 3;

/*
 * DIAGNOSTIC MODE
 *
 * Do not artificially cap processing.
 *
 * The old value was 30 FPS.
 * We need to measure what the device
 * and browser can actually sustain.
 */
export const MAX_PROCESSING_FPS = 120;
