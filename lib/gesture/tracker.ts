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

type VideoRect = {
  left: number;
  top: number;
  width: number;
  height: number;
};

const INDEX_TIP = 8;

/*
 * Match GestureWatcher's low-latency
 * One-Euro configuration.
 */
const FILTER_MIN_CUTOFF = 1.2;
const FILTER_BETA = 0.01;
const FILTER_D_CUTOFF = 1.0;

/*
 * --------------------------------------------------
 * PROCESSING CANVAS
 * --------------------------------------------------
 *
 * The visible camera stays full resolution.
 *
 * MediaPipe receives a much smaller copy with
 * exactly the same aspect ratio.
 *
 * This reduces the amount of image data that
 * has to be copied and processed while preserving
 * the landmark coordinate relationship with the
 * original camera frame.
 *
 * 480 is intentionally conservative on mobile.
 */
const MAX_PROCESSING_WIDTH = 480;

let cachedVideoRect:
  | {
      rect: VideoRect | null;
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
 * --------------------------------------------------
 * DISPLAYED CAMERA RECTANGLE
 * --------------------------------------------------
 *
 * MediaPipe coordinates are normalized against
 * the complete camera frame.
 *
 * The visible video uses object-fit: cover,
 * so the actual visible camera rectangle must
 * be calculated before converting coordinates.
 */
function getVideoContentRect(
  video: HTMLVideoElement
): VideoRect | null {
  if (
    !video.videoWidth ||
    !video.videoHeight ||
    !video.isConnected
  ) {
    return null;
  }

  const now =
    performance.now();

  if (
    cachedVideoRect &&
    now -
      cachedVideoRect.time <
      250
  ) {
    return cachedVideoRect.rect;
  }

  const element =
    video.getBoundingClientRect();

  if (
    element.width <= 0 ||
    element.height <= 0
  ) {
    return null;
  }

  const sourceAspect =
    video.videoWidth /
    video.videoHeight;

  const destinationAspect =
    element.width /
    element.height;

  let width =
    element.width;

  let height =
    element.height;

  if (
    destinationAspect >
    sourceAspect
  ) {
    height =
      element.height;

    width =
      height *
      sourceAspect;
  } else {
    width =
      element.width;

    height =
      width /
      sourceAspect;
  }

  const left =
    element.left +
    (
      element.width -
      width
    ) /
      2;

  const top =
    element.top +
    (
      element.height -
      height
    ) /
      2;

  const rect = {
    left,
    top,
    width,
    height
  };

  cachedVideoRect = {
    rect,
    time: now
  };

  return rect;
}

/*
 * --------------------------------------------------
 * CAMERA → VIEWPORT
 * --------------------------------------------------
 *
 * Rear camera is NOT mirrored.
 */
function cameraToViewport(
  video: HTMLVideoElement,
  point: Point3D
): Point2D {
  const rect =
    getVideoContentRect(
      video
    );

  if (!rect) {
    return {
      x:
        point.x *
        window.innerWidth,

      y:
        point.y *
        window.innerHeight
    };
  }

  const viewportX =
    rect.left +
    point.x *
      rect.width;

  const viewportY =
    rect.top +
    point.y *
      rect.height;

  return {
    x: clamp01(
      viewportX /
        window.innerWidth
    ),

    y: clamp01(
      viewportY /
        window.innerHeight
    )
  };
}

export class YuakeGestureTracker {
  private detector:
    HandLandmarker | null =
      null;

  private runtime =
    new Map<
      string,
      RuntimeState
    >();

  private initialized =
    false;

  private processingTime =
    0;

  private lastTimestamp =
    0;

  /*
   * Small processing canvas.
   *
   * It is never inserted into the DOM.
   */
  private processingCanvas:
    HTMLCanvasElement | null =
      null;

  private processingContext:
    CanvasRenderingContext2D | null =
      null;

  private processingWidth =
    0;

  private processingHeight =
    0;

  async initialize() {
    if (
      this.initialized
    ) {
      return;
    }

    const vision =
      await FilesetResolver.forVisionTasks(
        MEDIAPIPE_WASM_URL
      );

    this.detector =
      await HandLandmarker.createFromOptions(
        vision,
        {
          baseOptions: {
            modelAssetPath:
              HAND_MODEL_URL,

            delegate:
              "GPU"
          },

          runningMode:
            "VIDEO",

          numHands:
            MAX_HANDS
        }
      );

    /*
     * Create the processing canvas once.
     */
    this.processingCanvas =
      document.createElement(
        "canvas"
      );

    this.processingContext =
      this.processingCanvas.getContext(
        "2d",
        {
          alpha: false,
          desynchronized: true
        }
      );

    if (
      !this.processingContext
    ) {
      throw new Error(
        "Unable to create the hand-tracking processing canvas."
      );
    }

    /*
     * We do not need color interpolation quality
     * for hand landmarks.
     *
     * Disable it to reduce canvas work.
     */
    this.processingContext.imageSmoothingEnabled =
      false;

    this.initialized =
      true;
  }

  /*
   * --------------------------------------------------
   * PREPARE LOW-RES FRAME
   * --------------------------------------------------
   */
  private prepareProcessingFrame(
    video: HTMLVideoElement
  ): HTMLCanvasElement {
    if (
      !this.processingCanvas ||
      !this.processingContext
    ) {
      throw new Error(
        "Processing canvas has not been initialized."
      );
    }

    const sourceWidth =
      video.videoWidth;

    const sourceHeight =
      video.videoHeight;

    if (
      !sourceWidth ||
      !sourceHeight
    ) {
      throw new Error(
        "Camera video dimensions are unavailable."
      );
    }

    /*
     * Preserve the camera aspect ratio.
     */
    const scale =
      Math.min(
        1,
        MAX_PROCESSING_WIDTH /
          sourceWidth
      );

    const width =
      Math.max(
        1,
        Math.round(
          sourceWidth *
            scale
        )
      );

    const height =
      Math.max(
        1,
        Math.round(
          sourceHeight *
            scale
        )
      );

    if (
      width !==
        this.processingWidth ||
      height !==
        this.processingHeight
    ) {
      this.processingWidth =
        width;

      this.processingHeight =
        height;

      this.processingCanvas.width =
        width;

      this.processingCanvas.height =
        height;

      /*
       * Canvas resizing resets the context.
       */
      this.processingContext =
        this.processingCanvas.getContext(
          "2d",
          {
            alpha: false,
            desynchronized: true
          }
        );

      if (
        !this.processingContext
      ) {
        throw new Error(
          "Unable to recreate the processing canvas context."
        );
      }

      this.processingContext.imageSmoothingEnabled =
        false;
    }

    /*
     * Copy the current camera frame into
     * the smaller canvas.
     *
     * Same aspect ratio = same normalized
     * MediaPipe coordinates.
     */
    this.processingContext.drawImage(
      video,
      0,
      0,
      width,
      height
    );

    return this.processingCanvas;
  }

  process(
    video: HTMLVideoElement,
    timestamp: number
  ): GestureFrame {
    if (
      !this.detector
    ) {
      throw new Error(
        "Gesture tracker has not been initialized."
      );
    }

    const start =
      performance.now();

    /*
     * MediaPipe requires monotonically increasing
     * timestamps.
     */
    const safeTimestamp =
      Math.max(
        timestamp,
        this.lastTimestamp +
          0.001
      );

    /*
     * ------------------------------------------------
     * IMPORTANT:
     *
     * MediaPipe no longer receives the full
     * 1080×1920 camera frame.
     *
     * It receives the smaller same-aspect-ratio
     * processing canvas.
     * ------------------------------------------------
     */
    const processingFrame =
      this.prepareProcessingFrame(
        video
      );

    const result =
      this.detector.detectForVideo(
        processingFrame,
        safeTimestamp
      );

    /*
     * Coordinates returned by MediaPipe are still
     * normalized to the processing canvas.
     *
     * Because the canvas preserves the exact camera
     * aspect ratio, those normalized coordinates map
     * directly back onto the original camera frame.
     */
    const hands =
      this.convertResult(
        result,
        video,
        safeTimestamp
      );

    this.lastTimestamp =
      safeTimestamp;

    this.processingTime =
      performance.now() -
      start;

    return {
      timestamp:
        safeTimestamp,

      detected:
        hands.length >
        0,

      hands,

      primaryHand:
        this.selectPrimaryHand(
          hands
        ),

      frameTime:
        safeTimestamp,

      processingTime:
        this.processingTime
    };
  }

  private convertResult(
    result: HandLandmarkerResult,
    video: HTMLVideoElement,
    timestamp: number
  ): GestureHand[] {
    const hands:
      GestureHand[] = [];

    for (
      let index = 0;
      index <
        result.landmarks.length;
      index++
    ) {
      const raw =
        result.landmarks[
          index
        ];

      if (
        !raw ||
        raw.length <
          21
      ) {
        continue;
      }

      const landmarks =
        raw.map(
          toPoint3D
        );

      const worldRaw =
        result.worldLandmarks?.[
          index
        ];

      const worldLandmarks =
        worldRaw?.length === 21
          ? worldRaw.map(
              toPoint3D
            )
          : landmarks.map(
              point => ({
                ...point
              })
            );

      const handedness =
        getHandedness(
          result,
          index
        );

      const id =
        handedness ===
        "Unknown"
          ? `hand-${index}`
          : handedness;

      let runtime =
        this.runtime.get(
          id
        );

      if (!runtime) {
        runtime = {
          filter:
            new OneEuroPointFilter(
              {
                minCutoff:
                  FILTER_MIN_CUTOFF,

                beta:
                  FILTER_BETA,

                dCutoff:
                  FILTER_D_CUTOFF
              }
            ),

          classifier:
            new GestureClassifier(),

          previousCursor: {
            x: 0,
            y: 0
          },

          previousTime:
            timestamp,

          lastSeen:
            timestamp
        };

        this.runtime.set(
          id,
          runtime
        );
      }

      runtime.lastSeen =
        timestamp;

      /*
       * ------------------------------------------------
       * ACTUAL INDEX FINGERTIP
       * ------------------------------------------------
       *
       * Landmark 8 = index fingertip.
       *
       * No thumb midpoint.
       * No artificial offset.
       */
      const fingertip =
        landmarks[
          INDEX_TIP
        ];

      /*
       * Map the raw fingertip into the
       * actual visible camera rectangle.
       */
      const rawCursor =
        cameraToViewport(
          video,
          fingertip
        );

      /*
       * GestureWatcher-style One-Euro filtering.
       *
       * We filter screen-space coordinates,
       * because that's what the user sees.
       */
      const filtered =
        runtime.filter.filter(
          rawCursor,
          timestamp /
            1000
        );

      /*
       * ------------------------------------------------
       * VELOCITY
       * ------------------------------------------------
       *
       * Used for the UI/debug information.
       *
       * It does NOT alter the cursor.
       */
      const dt =
        clamp(
          (
            timestamp -
            runtime.previousTime
          ) /
            1000,

          0.001,
          0.1
        );

      const velocity = {
        x:
          (
            rawCursor.x -
            runtime.previousCursor.x
          ) /
          dt,

        y:
          (
            rawCursor.y -
            runtime.previousCursor.y
          ) /
          dt
      };

      const speed =
        Math.hypot(
          velocity.x,
          velocity.y
        );

      /*
       * ------------------------------------------------
       * CURSOR
       * ------------------------------------------------
       *
       * The filtered fingertip itself is the cursor.
       *
       * No trailing interpolation.
       * No dead-zone.
       * No arbitrary offset.
       * No fake cursor movement.
       */
      const cursor = {
        x: clamp01(
          filtered.x
        ),

        y: clamp01(
          filtered.y
        )
      };

      const classification =
        runtime.classifier.classify(
          landmarks
        );

      const palm =
        calculatePalm(
          worldLandmarks
        );

      const palmNormal =
        calculatePalmNormal(
          worldLandmarks
        );

      const confidence =
        this.getConfidence(
          result,
          index
        );

      hands.push({
        id,

        handedness,

        landmarks,

        worldLandmarks,

        wrist:
          worldLandmarks[0],

        palm,

        palmNormal,

        cursor,

        velocity,

        speed,

        pinch:
          classification.pinch,

        pinchStrength:
          classification.pinchStrength,

        pose:
          classification.pose,

        fingers:
          classification.fingers,

        confidence,

        visible:
          true,

        lastSeen:
          timestamp
      });

      runtime.previousCursor =
        rawCursor;

      runtime.previousTime =
        timestamp;
    }

    this.removeLostHands(
      timestamp
    );

    return hands;
  }

  private getConfidence(
    result: HandLandmarkerResult,
    index: number
  ) {
    return clamp01(
      result.handednesses?.[
        index
      ]?.[0]?.score ??
        0.8
    );
  }

  private selectPrimaryHand(
    hands: GestureHand[]
  ) {
    if (
      hands.length ===
      0
    ) {
      return null;
    }

    const pinching =
      hands.find(
        hand =>
          hand.pinch
      );

    if (pinching) {
      return pinching;
    }

    return [...hands].sort(
      (a, b) =>
        b.confidence -
        a.confidence
    )[0];
  }

  private removeLostHands(
    timestamp: number
  ) {
    for (
      const [
        id,
        runtime
      ] of this.runtime
    ) {
      if (
        timestamp -
          runtime.lastSeen >
        HAND_LOST_GRACE_MS
      ) {
        runtime.filter.reset();

        runtime.classifier.reset();

        this.runtime.delete(
          id
        );
      }
    }
  }

  close() {
    this.detector?.close();

    this.detector =
      null;

    this.runtime.clear();

    this.initialized =
      false;

    this.processingTime =
      0;

    this.lastTimestamp =
      0;

    this.processingCanvas =
      null;

    this.processingContext =
      null;

    this.processingWidth =
      0;

    this.processingHeight =
      0;

    cachedVideoRect =
      null;
  }
      }
