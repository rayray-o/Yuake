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

const FILTER_MIN_CUTOFF = 1.0;
const FILTER_BETA = 0.03;
const FILTER_D_CUTOFF = 1.0;

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

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function clamp01(value: number) {
  return clamp(value, 0, 1);
}

function toPoint3D(point: { x: number; y: number; z: number }): Point3D {
  return {
    x: Number.isFinite(point.x) ? point.x : 0,
    y: Number.isFinite(point.y) ? point.y : 0,
    z: Number.isFinite(point.z) ? point.z : 0
  };
}

function getHandedness(result: HandLandmarkerResult, index: number): Handedness {
  const label = result.handednesses?.[index]?.[0]?.categoryName;

  if (label === "Left" || label === "Right") {
    return label;
  }

  return "Unknown";
}

function getViewportRect(video: HTMLVideoElement) {
  const now = performance.now();
  const element = video.getBoundingClientRect();

  if (element.width <= 0 || element.height <= 0) {
    return null;
  }

  if (
    cachedViewport &&
    now - cachedViewport.time < 100 &&
    cachedViewport.width === element.width &&
    cachedViewport.height === element.height &&
    cachedViewport.left === element.left &&
    cachedViewport.top === element.top
  ) {
    return cachedViewport;
  }

  cachedViewport = {
    left: element.left,
    top: element.top,
    width: element.width,
    height: element.height,
    time: now
  };

  return cachedViewport;
}

function cameraToViewport(video: HTMLVideoElement, point: Point3D): Point2D {
  const viewport = getViewportRect(video);

  if (!viewport || !video.videoWidth || !video.videoHeight) {
    return {
      x: clamp01(point.x),
      y: clamp01(point.y)
    };
  }

  const sourceWidth = video.videoWidth;
  const sourceHeight = video.videoHeight;
  const viewportWidth = viewport.width;
  const viewportHeight = viewport.height;

  const scale = Math.max(
    viewportWidth / sourceWidth,
    viewportHeight / sourceHeight
  );

  const scaledWidth = sourceWidth * scale;
  const scaledHeight = sourceHeight * scale;

  const cropX = (scaledWidth - viewportWidth) / 2;
  const cropY = (scaledHeight - viewportHeight) / 2;

  const sourceX = clamp01(point.x) * sourceWidth;
  const sourceY = clamp01(point.y) * sourceHeight;

  const displayedX = sourceX * scale - cropX;
  const displayedY = sourceY * scale - cropY;

  const screenX = viewport.left + displayedX;
  const screenY = viewport.top + displayedY;

  return {
    x: clamp01(screenX / window.innerWidth),
    y: clamp01(screenY / window.innerHeight)
  };
}

export class YuakeGestureTracker {
  private detector: HandLandmarker | null = null;

  private runtime = new Map<string, RuntimeState>();

  private initialized = false;
  private processingTime = 0;
  private lastTimestamp = 0;

  private processingCanvas: HTMLCanvasElement | null = null;
  private processingContext: CanvasRenderingContext2D | null = null;
  private processingWidth = 0;
  private processingHeight = 0;

  async initialize() {
    if (this.initialized) {
      return;
    }

    const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);

    this.detector = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: HAND_MODEL_URL,
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numHands: MAX_HANDS
    });

    this.processingCanvas = document.createElement("canvas");

    this.processingContext = this.processingCanvas.getContext("2d", {
      alpha: false,
      desynchronized: true
    });

    if (!this.processingContext) {
      throw new Error("Unable to create hand-tracking processing canvas.");
    }

    this.processingContext.imageSmoothingEnabled = false;

    this.initialized = true;
  }

  private prepareProcessingFrame(video: HTMLVideoElement) {
    if (!this.processingCanvas || !this.processingContext) {
      throw new Error("Processing canvas is not initialized.");
    }

    const sourceWidth = video.videoWidth;
    const sourceHeight = video.videoHeight;

    if (!sourceWidth || !sourceHeight) {
      throw new Error("Camera dimensions are unavailable.");
    }

    const scale = Math.min(1, MAX_PROCESSING_WIDTH / sourceWidth);

    const width = Math.max(1, Math.round(sourceWidth * scale));
    const height = Math.max(1, Math.round(sourceHeight * scale));

    if (width !== this.processingWidth || height !== this.processingHeight) {
      this.processingWidth = width;
      this.processingHeight = height;

      this.processingCanvas.width = width;
      this.processingCanvas.height = height;

      this.processingContext = this.processingCanvas.getContext("2d", {
        alpha: false,
        desynchronized: true
      });

      if (!this.processingContext) {
        throw new Error("Unable to recreate processing canvas.");
      }

      this.processingContext.imageSmoothingEnabled = false;
    }

    this.processingContext.drawImage(video, 0, 0, width, height);

    return this.processingCanvas;
  }

  process(video: HTMLVideoElement, timestamp: number): GestureFrame {
    if (!this.detector) {
      throw new Error("Gesture tracker has not been initialized.");
    }

    const start = performance.now();

    const safeTimestamp = Math.max(timestamp, this.lastTimestamp + 0.001);

    const processingFrame = this.prepareProcessingFrame(video);

    const result = this.detector.detectForVideo(processingFrame, safeTimestamp);

    const hands = this.convertResult(result, video, safeTimestamp);

    this.lastTimestamp = safeTimestamp;
    this.processingTime = performance.now() - start;

    return {
      timestamp: safeTimestamp,
      detected: hands.length > 0,
      hands,
      primaryHand: this.selectPrimaryHand(hands),
      frameTime: safeTimestamp,
      processingTime: this.processingTime
    };
  }

  private convertResult(
    result: HandLandmarkerResult,
    video: HTMLVideoElement,
    timestamp: number
  ): GestureHand[] {
    const hands: GestureHand[] = [];

    for (let index = 0; index < result.landmarks.length; index++) {
      const raw = result.landmarks[index];

      if (!raw || raw.length < 21) {
        continue;
      }

      const landmarks = raw.map(toPoint3D);

      const worldRaw = result.worldLandmarks?.[index];

      const worldLandmarks =
        worldRaw?.length === 21
          ? worldRaw.map(toPoint3D)
          : landmarks.map(point => ({ ...point }));

      const handedness = getHandedness(result, index);

      const id = handedness === "Unknown" ? `hand-${index}` : handedness;

      let runtime = this.runtime.get(id);

      if (!runtime) {
        runtime = {
          filter: new OneEuroPointFilter({
            minCutoff: FILTER_MIN_CUTOFF,
            beta: FILTER_BETA,
            dCutoff: FILTER_D_CUTOFF
          }),
          classifier: new GestureClassifier(),
          previousCursor: { x: 0, y: 0 },
          previousTime: timestamp,
          lastSeen: timestamp
        };

        this.runtime.set(id, runtime);
      }

      runtime.lastSeen = timestamp;

      const fingertip = landmarks[INDEX_TIP];

      const rawCursor = cameraToViewport(video, fingertip);

      const pixelCursor = {
        x: rawCursor.x * window.innerWidth,
        y: rawCursor.y * window.innerHeight
      };

      const filteredPixels = runtime.filter.filter(pixelCursor, timestamp / 1000);

      const filtered = {
        x: filteredPixels.x / window.innerWidth,
        y: filteredPixels.y / window.innerHeight
      };

      const dt = clamp((timestamp - runtime.previousTime) / 1000, 0.001, 0.1);

      const velocity = {
        x: (rawCursor.x - runtime.previousCursor.x) / dt,
        y: (rawCursor.y - runtime.previousCursor.y) / dt
      };

      const speed = Math.hypot(velocity.x, velocity.y);

      const cursor = {
        x: clamp01(filtered.x),
        y: clamp01(filtered.y)
      };

      const classification = runtime.classifier.classify(landmarks);

      const palm = calculatePalm(worldLandmarks);
      const palmNormal = calculatePalmNormal(worldLandmarks);

      const confidence = this.getConfidence(result, index);

      hands.push({
        id,
        handedness,
        landmarks,
        worldLandmarks,
        wrist: worldLandmarks[0],
        palm,
        palmNormal,
        cursor,
        velocity,
        speed,
        pinch: classification.pinch,
        pinchStrength: classification.pinchStrength,
        pose: classification.pose,
        fingers: classification.fingers,
        confidence,
        visible: true,
        lastSeen: timestamp
      });

      runtime.previousCursor = rawCursor;
      runtime.previousTime = timestamp;
    }

    this.removeLostHands(timestamp);

    return hands;
  }

  private getConfidence(result: HandLandmarkerResult, index: number) {
    return clamp01(result.handednesses?.[index]?.[0]?.score ?? 0.8);
  }

  private selectPrimaryHand(hands: GestureHand[]) {
    if (hands.length === 0) {
      return null;
    }

    const pinching = hands.find(hand => hand.pinch);

    if (pinching) {
      return pinching;
    }

    return [...hands].sort((a, b) => b.confidence - a.confidence)[0];
  }

  private removeLostHands(timestamp: number) {
    for (const [id, runtime] of this.runtime) {
      if (timestamp - runtime.lastSeen > HAND_LOST_GRACE_MS) {
        runtime.filter.reset();
        runtime.classifier.reset();
        this.runtime.delete(id);
      }
    }
  }

  close() {
    this.detector?.close();
    this.detector = null;

    this.runtime.clear();

    this.initialized = false;
    this.processingTime = 0;
    this.lastTimestamp = 0;

    this.processingCanvas = null;
    this.processingContext = null;
    this.processingWidth = 0;
    this.processingHeight = 0;

    cachedViewport = null;
  }
  }
