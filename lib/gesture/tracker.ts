import {
  FilesetResolver,
  HandLandmarker,
  type HandLandmarkerResult
} from "@mediapipe/tasks-vision";

import {
  DETECTION_CONFIDENCE,
  FILTER_BETA,
  FILTER_D_CUTOFF,
  FILTER_MIN_CUTOFF,
  HAND_LOST_GRACE_MS,
  HAND_MODEL_URL,
  MAX_HANDS,
  MEDIAPIPE_WASM_URL,
  PRESENCE_CONFIDENCE,
  TRACKING_CONFIDENCE
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

  previousRaw: Point2D;
  previousTimestamp: number;

  lastSeen: number;
  initialized: boolean;
};

type VideoRect = {
  left: number;
  top: number;
  width: number;
  height: number;
};

const INDEX_TIP = 8;

const MIN_DT = 0.001;
const MAX_DT = 0.1;

/*
 * Cache the displayed camera rectangle.
 *
 * The camera is CSS object-fit: cover.
 * MediaPipe coordinates are relative to
 * the ORIGINAL camera frame.
 *
 * These are NOT the same coordinate space.
 */
let rectCache: {
  time: number;
  rect: VideoRect | null;
} = {
  time: 0,
  rect: null
};

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
  const category =
    result.handednesses?.[
      index
    ]?.[0]?.categoryName;

  if (
    category === "Left" ||
    category === "Right"
  ) {
    return category;
  }

  return "Unknown";
}

/**
 * Calculate the actual visible portion
 * of a video using CSS object-fit: cover.
 *
 * Example:
 *
 * Camera = 16:9
 * Phone screen = 9:20
 *
 * The camera gets cropped heavily on
 * the left/right.
 *
 * We must account for that crop.
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
    now -
      rectCache.time <
    100
  ) {
    return rectCache.rect;
  }

  const element =
    video.getBoundingClientRect();

  if (
    element.width <= 0 ||
    element.height <= 0
  ) {
    return null;
  }

  const videoAspect =
    video.videoWidth /
    video.videoHeight;

  const elementAspect =
    element.width /
    element.height;

  let renderedWidth =
    element.width;

  let renderedHeight =
    element.height;

  /*
   * object-fit: cover means:
   *
   * whichever dimension needs
   * MORE scaling determines the scale.
   */
  if (
    elementAspect >
    videoAspect
  ) {
    /*
     * Element is relatively wider
     * than the camera frame.
     *
     * Height determines scale.
     */
    renderedHeight =
      element.height;

    renderedWidth =
      renderedHeight *
      videoAspect;
  } else {
    /*
     * Element is relatively taller
     * than the camera frame.
     *
     * Width determines scale.
     */
    renderedWidth =
      element.width;

    renderedHeight =
      renderedWidth /
      videoAspect;
  }

  /*
   * Center the scaled video inside
   * the element.
   *
   * The negative overflow represents
   * the crop produced by object-fit cover.
   */
  const left =
    element.left +
    (
      element.width -
      renderedWidth
    ) /
      2;

  const top =
    element.top +
    (
      element.height -
      renderedHeight
    ) /
      2;

  rectCache = {
    time: now,

    rect: {
      left,
      top,

      width:
        renderedWidth,

      height:
        renderedHeight
    }
  };

  return rectCache.rect;
}

/**
 * Convert MediaPipe camera-frame coordinates
 * to actual viewport coordinates.
 *
 * IMPORTANT:
 *
 * YUAKE uses the rear camera and the page
 * intentionally does NOT mirror it.
 *
 * Therefore:
 *
 * x = landmark.x
 *
 * NOT:
 *
 * x = 1 - landmark.x
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

  const screenX =
    rect.left +
    point.x *
      rect.width;

  const screenY =
    rect.top +
    point.y *
      rect.height;

  /*
   * GestureHand.cursor historically
   * uses normalized viewport coordinates.
   *
   * Keep that API so the rest of YUAKE
   * does not need to change.
   */
  return {
    x: clamp01(
      screenX /
        window.innerWidth
    ),

    y: clamp01(
      screenY /
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

  private lastFrameTime =
    0;

  private processingTime =
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
            MAX_HANDS,

          minHandDetectionConfidence:
            DETECTION_CONFIDENCE,

          minHandPresenceConfidence:
            PRESENCE_CONFIDENCE,

          minTrackingConfidence:
            TRACKING_CONFIDENCE
        }
      );

    this.initialized =
      true;
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

    const safeTimestamp =
      Math.max(
        timestamp,
        this.lastFrameTime +
          0.001
      );

    const result =
      this.detector.detectForVideo(
        video,
        safeTimestamp
      );

    const hands =
      this.convertResult(
        result,
        video,
        safeTimestamp
      );

    this.lastFrameTime =
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
    const output:
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

      let state =
        this.runtime.get(
          id
        );

      if (!state) {
        state = {
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

          previousRaw: {
            x: 0,
            y: 0
          },

          previousTimestamp:
            timestamp,

          lastSeen:
            timestamp,

          initialized:
            false
        };

        this.runtime.set(
          id,
          state
        );
      }

      state.lastSeen =
        timestamp;

      /*
       * =====================================================
       * THE IMPORTANT PART
       * =====================================================
       *
       * Landmark #8 is the physical
       * index fingertip.
       */
      const fingertip =
        landmarks[
          INDEX_TIP
        ];

      /*
       * Convert the fingertip from
       * CAMERA SPACE → SCREEN SPACE.
       *
       * This accounts for object-fit: cover.
       */
      const rawCursor =
        cameraToViewport(
          video,
          fingertip
        );

      if (
        !state.initialized
      ) {
        state.previousRaw =
          rawCursor;

        state.previousTimestamp =
          timestamp;

        state.filter.reset();

        /*
         * Prime filter.
         *
         * This filter is NOT used for
         * the visual cursor.
         */
        state.filter.filter(
          rawCursor,
          timestamp /
            1000
        );

        state.initialized =
          true;
      }

      const dt =
        clamp(
          (
            timestamp -
            state.previousTimestamp
          ) /
            1000,

          MIN_DT,
          MAX_DT
        );

      /*
       * Velocity is calculated from
       * the ACTUAL screen-space fingertip.
       */
      const velocity =
        {
          x:
            (
              rawCursor.x -
              state.previousRaw.x
            ) /
            dt,

          y:
            (
              rawCursor.y -
              state.previousRaw.y
            ) /
            dt
        };

      const speed =
        Math.hypot(
          velocity.x,
          velocity.y
        );

      /*
       * Feed the filter for the
       * non-cursor systems.
       *
       * The visual cursor does NOT
       * use this value.
       */
      state.filter.filter(
        rawCursor,
        timestamp /
          1000
      );

      /*
       * IMPORTANT:
       *
       * NO:
       *
       * lerp()
       * prediction
       * velocity compensation
       * adaptive blend
       * dead zone
       *
       * The cursor is literally the
       * mapped fingertip.
       */
      const cursor =
        rawCursor;

      const classification =
        state.classifier.classify(
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
        this.calculateConfidence(
          result,
          index
        );

      const hand:
        GestureHand = {
        id,

        handedness,

        landmarks,

        worldLandmarks,

        wrist:
          worldLandmarks[0],

        palm,

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

        palmNormal,

        confidence,

        visible:
          true,

        lastSeen:
          timestamp
      };

      state.previousRaw =
        rawCursor;

      state.previousTimestamp =
        timestamp;

      output.push(
        hand
      );
    }

    this.removeLostHands(
      timestamp
    );

    return output;
  }

  private calculateConfidence(
    result: HandLandmarkerResult,
    index: number
  ) {
    const category =
      result.handednesses?.[
        index
      ]?.[0];

    return clamp01(
      category?.score ??
        0.8
    );
  }

  private selectPrimaryHand(
    hands: GestureHand[]
  ) {
    if (
      hands.length === 0
    ) {
      return null;
    }

    /*
     * Preserve the current behavior:
     * pinch takes priority.
     */
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
        state
      ] of this.runtime
    ) {
      if (
        timestamp -
          state.lastSeen >
        HAND_LOST_GRACE_MS
      ) {
        state.filter.reset();

        state.classifier.reset();

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

    this.lastFrameTime =
      0;

    this.processingTime =
      0;

    rectCache = {
      time: 0,
      rect: null
    };
  }
        }
