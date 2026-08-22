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

type CursorPoint = {
  x: number;
  y: number;
};

type HandRuntime = {
  filter: OneEuroPointFilter;
  classifier: GestureClassifier;

  previousCursor: CursorPoint;

  previousTimestamp: number;

  velocity: CursorPoint;

  lastSeen: number;

  initialized: boolean;
};

const INDEX_TIP = 8;

const MIN_DT_SECONDS = 0.001;

const MAX_DT_SECONDS = 0.1;

/*
 * Prediction is intentionally tiny.
 *
 * Too much prediction makes a cursor
 * overshoot the actual fingertip.
 *
 * The goal is to compensate for the
 * small latency introduced by camera
 * capture + MediaPipe + filtering.
 */
const MIN_PREDICTION_SECONDS = 0.006;

const MAX_PREDICTION_SECONDS = 0.024;

/*
 * Prevent one noisy landmark frame
 * from producing a huge cursor jump.
 */
const MAX_VELOCITY = 8;

function clamp(
  value: number,
  min: number,
  max: number
): number {
  return Math.max(
    min,
    Math.min(max, value)
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

function limitVelocity(
  velocity: CursorPoint
): CursorPoint {
  const speed =
    Math.hypot(
      velocity.x,
      velocity.y
    );

  if (
    !Number.isFinite(speed) ||
    speed <= MAX_VELOCITY
  ) {
    return velocity;
  }

  const scale =
    MAX_VELOCITY /
    speed;

  return {
    x: velocity.x * scale,
    y: velocity.y * scale
  };
}

function calculateAdaptivePrediction(
  speed: number
): number {
  /*
   * Slow movement:
   * almost no prediction.
   *
   * Fast movement:
   * slightly more prediction.
   */
  const normalizedSpeed =
    clamp01(
      speed / 3
    );

  return (
    MIN_PREDICTION_SECONDS +
    (
      MAX_PREDICTION_SECONDS -
      MIN_PREDICTION_SECONDS
    ) *
      normalizedSpeed
  );
}

export class YuakeGestureTracker {
  private detector:
    HandLandmarker | null = null;

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
    if (this.initialized) {
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

            delegate: "GPU"
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
    if (!this.detector) {
      throw new Error(
        "Gesture tracker has not been initialized."
      );
    }

    const start =
      performance.now();

    /*
     * MediaPipe VIDEO mode expects
     * a monotonically increasing
     * timestamp in milliseconds.
     */
    const safeTimestamp =
      Math.max(
        timestamp,
        this.lastFrameTime + 0.001
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
        this.lastFrameTime,

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
        result.landmarks[index];

      if (
        !raw ||
        raw.length < 21
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

      /*
       * Handedness gives us a stable
       * identity when available.
       *
       * Unknown hands use the result
       * index as a temporary identity.
       */
      const id =
        handedness === "Unknown"
          ? `hand-${index}`
          : handedness;

      let state =
        this.runtime.get(id);

      if (!state) {
        state = {
          /*
           * Slightly more responsive
           * One-Euro configuration.
           *
           * The adaptive beta allows
           * fast motion to pass through
           * without making slow motion
           * noisy.
           */
          filter:
            new OneEuroPointFilter({
              minCutoff:
                FILTER_MIN_CUTOFF,

              beta:
                Math.max(
                  FILTER_BETA,
                  0.055
                ),

              dCutoff:
                FILTER_D_CUTOFF
            }),

          classifier:
            new GestureClassifier(),

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

          initialized: false
        };

        this.runtime.set(
          id,
          state
        );
      }

      state.lastSeen =
        timestamp;

      /*
       * INDEX FINGERTIP
       *
       * Landmark 8 is the actual
       * index fingertip.
       *
       * The old implementation used
       * the midpoint between landmarks
       * 4 and 8, which meant the cursor
       * was never actually attached to
       * the fingertip.
       */
      const indexTip =
        landmarks[INDEX_TIP];

      /*
       * Rear-facing camera is NOT mirrored.
       *
       * Therefore we keep the landmark
       * X coordinate exactly as MediaPipe
       * reports it.
       */
      const rawCursor: CursorPoint = {
        x: clamp01(
          indexTip.x
        ),

        y: clamp01(
          indexTip.y
        )
      };

      /*
       * First frame should snap directly
       * to the fingertip.
       *
       * This prevents the cursor from
       * appearing at the center and then
       * sliding toward the hand.
       */
      if (!state.initialized) {
        state.filter.reset();

        state.previousCursor =
          rawCursor;

        state.previousTimestamp =
          timestamp;

        state.velocity = {
          x: 0,
          y: 0
        };

        state.initialized = true;
      }

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

          MIN_DT_SECONDS,
          MAX_DT_SECONDS
        );

      /*
       * Estimate velocity from the
       * filtered fingertip trajectory.
       */
      const measuredVelocity = {
        x:
          (
            filtered.x -
            state.previousCursor.x
          ) /
          deltaSeconds,

        y:
          (
            filtered.y -
            state.previousCursor.y
          ) /
          deltaSeconds
      };

      const limitedVelocity =
        limitVelocity(
          measuredVelocity
        );

      /*
       * Smooth the velocity separately.
       *
       * This avoids using one noisy
       * frame to create a giant
       * prediction jump.
       */
      const velocityBlend =
        0.35;

      state.velocity = {
        x:
          state.velocity.x *
            (
              1 -
              velocityBlend
            ) +
          limitedVelocity.x *
            velocityBlend,

        y:
          state.velocity.y *
            (
              1 -
              velocityBlend
            ) +
          limitedVelocity.y *
            velocityBlend
      };

      const speed =
        Math.hypot(
          state.velocity.x,
          state.velocity.y
        );

      /*
       * Short-horizon prediction.
       *
       * This compensates for the tiny
       * delay between the real fingertip
       * and the rendered cursor.
       */
      const predictionSeconds =
        calculateAdaptivePrediction(
          speed
        );

      const predictedCursor: CursorPoint = {
        x: clamp01(
          filtered.x +
            state.velocity.x *
              predictionSeconds
        ),

        y: clamp01(
          filtered.y +
            state.velocity.y *
              predictionSeconds
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

      const hand: GestureHand = {
        id,

        handedness,

        landmarks,

        worldLandmarks,

        wrist:
          worldLandmarks[0],

        palm,

        cursor:
          predictedCursor,

        velocity:
          state.velocity,

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

        visible: true,

        lastSeen:
          timestamp
      };

      /*
       * Important:
       *
       * The previous position for the
       * next velocity calculation is
       * the FILTERED position, not the
       * predicted position.
       *
       * Otherwise prediction would feed
       * itself and create runaway drift.
       */
      state.previousCursor =
        filtered;

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
     * Movement should not make the
     * hand disappear. We only expose
     * a slightly lower confidence
     * during extreme motion.
     */
    const motionPenalty =
      Math.min(
        speed / 16,
        0.18
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

    /*
     * Keep pinch priority so a hand
     * actively interacting remains
     * the primary hand.
     */
    const pinching =
      hands.find(
        hand =>
          hand.pinch
      );

    if (pinching) {
      return pinching;
    }

    /*
     * Otherwise prefer the hand with
     * the strongest confidence.
     */
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
