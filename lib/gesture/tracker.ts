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
  calculatePalmNormal,
  midpoint2D
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

type HandRuntime = {
  filter: OneEuroPointFilter;
  classifier: GestureClassifier;

  previousCursor: {
    x: number;
    y: number;
  };

  previousTimestamp: number;

  lastSeen: number;
};

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

    const result =
      this.detector.detectForVideo(
        video,
        timestamp
      );

    const hands =
      this.convertResult(
        result,
        timestamp
      );

    this.lastFrameTime =
      timestamp;

    this.processingTime =
      performance.now() -
      start;

    return {
      timestamp,

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
        handedness === "Unknown"
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

          previousCursor: {
            x: 0.5,
            y: 0.5
          },

          previousTimestamp:
            timestamp,

          lastSeen:
            timestamp
        };

        this.runtime.set(
          id,
          state
        );
      }

      state.lastSeen =
        timestamp;

      /*
       * Cursor anchor:
       *
       * midpoint between thumb
       * and index.
       *
       * This gives a more stable
       * interaction point during pinch.
       */
      const rawCursor =
        midpoint2D(
          {
            x: landmarks[4].x,
            y: landmarks[4].y
          },
          {
            x: landmarks[8].x,
            y: landmarks[8].y
          }
        );

      /*
       * Mirror the X coordinate so
       * the interaction feels natural
       * with a front-facing camera.
       */
      const mirroredCursor = {
        x:
          1 -
          rawCursor.x,

        y:
          rawCursor.y
      };

      const cursor =
        state.filter.filter(
          mirroredCursor,
          timestamp /
            1000
        );

      const deltaSeconds =
        Math.max(
          (
            timestamp -
            state.previousTimestamp
          ) /
            1000,
          0.001
        );

      const velocity = {
        x:
          (
            cursor.x -
            state.previousCursor.x
          ) /
          deltaSeconds,

        y:
          (
            cursor.y -
            state.previousCursor.y
          ) /
          deltaSeconds
      };

      const speed =
        Math.hypot(
          velocity.x,
          velocity.y
        );

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

        visible: true,

        lastSeen:
          timestamp
      };

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
     * Extremely rapid movement is
     * naturally harder to track.
     *
     * We don't reject the hand;
     * we simply reduce the confidence
     * signal exposed to consumers.
     */
    const motionPenalty =
      Math.min(
        speed / 12,
        0.25
      );

    return Math.max(
      0,
      Math.min(
        1,
        score -
          motionPenalty
      )
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
     * Prefer a currently pinching hand.
     * Otherwise use highest confidence.
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
  }
  }
