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

type HandRuntime = {
  filter: OneEuroPointFilter;

  classifier: GestureClassifier;

  previousRaw: Point2D;

  previousCursor: Point2D;

  previousTimestamp: number;

  velocity: Point2D;

  lastSeen: number;

  initialized: boolean;
};

const INDEX_TIP =
  8;

const MIN_DT =
  0.001;

const MAX_DT =
  0.100;

/*
 * At low speed we want the
 * filtered position.
 *
 * At high speed we want the
 * raw fingertip.
 *
 * This is intentionally NOT
 * prediction.
 *
 * Prediction can make the cursor
 * lead the fingertip and feel wrong.
 */
const RAW_BLEND_START =
  0.35;

const RAW_BLEND_FULL =
  2.2;

/*
 * Small dead-zone for tiny
 * MediaPipe landmark noise.
 *
 * This is applied only to
 * very slow movement.
 */
const MICRO_MOVEMENT =
  0.0012;

function clamp(
  value: number,
  min: number,
  max: number
): number {
  return Math.max(
    min,
    Math.min(
      max,
      value
    )
  );
}

function clamp01(
  value: number
): number {
  return clamp(
    value,
    0,
    1
  );
}

function lerp(
  a: number,
  b: number,
  amount: number
): number {
  return (
    a +
    (b - a) *
      amount
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

function handednessFromResult(
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

function movementBlend(
  speed: number
): number {
  /*
   * speed is normalized
   * screen-space units / second.
   *
   * The transition is smooth:
   *
   * 0.00 -> filtered
   * 0.35 -> begin trusting raw
   * 2.20 -> almost completely raw
   */
  return clamp01(
    (
      speed -
      RAW_BLEND_START
    ) /
    (
      RAW_BLEND_FULL -
      RAW_BLEND_START
    )
  );
}

export class YuakeGestureTracker {
  private detector:
    HandLandmarker | null =
      null;

  private runtime =
    new Map<
      string,
      HandRuntime
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
        hands.length > 0,

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
        handednessFromResult(
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
                /*
                 * Low enough to remove
                 * camera jitter.
                 *
                 * Fast movement is
                 * handled by the
                 * raw/filtered blend
                 * below.
                 */
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
            x: 0.5,
            y: 0.5
          },

          previousCursor: {
            x: 0.5,
            y: 0.5
          },

          previousTimestamp:
            timestamp,

          velocity: {
            x: 0,
            y: 0
          },

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
       * THE ACTUAL FINGERTIP.
       *
       * MediaPipe landmark 8 =
       * index finger tip.
       */
      const fingertip =
        landmarks[
          INDEX_TIP
        ];

      /*
       * Rear camera is not mirrored.
       *
       * Keep X exactly as MediaPipe
       * reports it.
       */
      const rawCursor: Point2D = {
        x: clamp01(
          fingertip.x
        ),

        y: clamp01(
          fingertip.y
        )
      };

      if (
        !state.initialized
      ) {
        state.filter.reset();

        /*
         * First frame snaps
         * directly to the fingertip.
         */
        state.previousRaw =
          rawCursor;

        state.previousCursor =
          rawCursor;

        state.previousTimestamp =
          timestamp;

        state.velocity = {
          x: 0,
          y: 0
        };

        /*
         * Prime the filter without
         * creating an initial offset.
         */
        state.filter.filter(
          rawCursor,
          timestamp / 1000
        );

        state.initialized =
          true;
      }

      /*
       * Filtered position gives us
       * stability during tiny hand
       * movements.
       */
      const filtered =
        state.filter.filter(
          rawCursor,
          timestamp / 1000
        );

      const deltaSeconds =
        clamp(
          (
            timestamp -
            state.previousTimestamp
          ) / 1000,

          MIN_DT,
          MAX_DT
        );

      /*
       * IMPORTANT:
       *
       * Velocity is measured from
       * RAW fingertip movement.
       *
       * Measuring it from the filtered
       * cursor would hide fast movement
       * and make the cursor trail.
       */
      const rawVelocity: Point2D = {
        x:
          (
            rawCursor.x -
            state.previousRaw.x
          ) /
          deltaSeconds,

        y:
          (
            rawCursor.y -
            state.previousRaw.y
          ) /
          deltaSeconds
      };

      const speed =
        Math.hypot(
          rawVelocity.x,
          rawVelocity.y
        );

      /*
       * Fast movement:
       *
       * progressively trust the raw
       * fingertip instead of the delayed
       * filtered position.
       */
      const rawAmount =
        movementBlend(
          speed
        );

      let cursor: Point2D = {
        x: lerp(
          filtered.x,
          rawCursor.x,
          rawAmount
        ),

        y: lerp(
          filtered.y,
          rawCursor.y,
          rawAmount
        )
      };

      /*
       * At extremely tiny movement,
       * retain the previous cursor
       * to stop microscopic MediaPipe
       * jitter.
       *
       * This does NOT apply during
       * actual movement.
       */
      const movement =
        Math.hypot(
          rawCursor.x -
            state.previousRaw.x,

          rawCursor.y -
            state.previousRaw.y
        );

      if (
        movement <
          MICRO_MOVEMENT &&
        speed <
          RAW_BLEND_START
      ) {
        cursor =
          state.previousCursor;
      }

      /*
       * Never allow the visual cursor
       * to leave the normalized screen.
       */
      cursor = {
        x: clamp01(
          cursor.x
        ),

        y: clamp01(
          cursor.y
        )
      };

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
          index,
          speed
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

        velocity:
          rawVelocity,

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

      state.previousCursor =
        cursor;

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
    index: number,
    speed: number
  ): number {
    const category =
      result.handednesses?.[
        index
      ]?.[0];

    const score =
      category?.score ??
      0.8;

    /*
     * Do not punish ordinary
     * fast movement heavily.
     */
    const motionPenalty =
      Math.min(
        speed / 20,
        0.12
      );

    return clamp01(
      score -
        motionPenalty
    );
  }

  private selectPrimaryHand(
    hands: GestureHand[]
  ): GestureHand | null {
    if (
      hands.length === 0
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
  }
    }
