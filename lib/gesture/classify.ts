import {
  calculateFingerBend,
  calculateHandScale,
  clamp,
  distance3D,
  normalizedFingerDistance
} from "./geometry";

import {
  FINGER_TOUCH_END,
  FINGER_TOUCH_START,
  OPEN_FINGER_COUNT,
  PINCH_END,
  PINCH_START
} from "./constants";

import type {
  FingerName,
  FingerStates,
  Point3D,
  Pose
} from "./types";

type HysteresisState = {
  active: boolean;
};

/*
 * YUAKE Advanced Pinch Engine
 *
 * The cursor tracker is already a continuous, velocity-aware system.
 * Pinch now gets the same treatment without changing the public
 * GestureClassifier API used by tracker.ts.
 *
 * Layers:
 *   1. scale-normalized thumb/index geometry
 *   2. short-horizon temporal velocity
 *   3. adaptive velocity filtering
 *   4. approach/release awareness
 *   5. asymmetric engage/release hysteresis
 *   6. temporal confidence / stability
 *   7. continuous pinch strength
 *   8. fast release, stable engagement
 *
 * No artificial long debounce is used: the goal is responsiveness,
 * not hiding bad landmark data behind delay.
 */

const PINCH_VELOCITY_CUTOFF = 18;
const PINCH_VELOCITY_SCALE = 5.5;

const PINCH_ARM_DISTANCE =
  PINCH_END;

const PINCH_COMMIT_DISTANCE =
  PINCH_START;

const PINCH_REARM_DISTANCE =
  PINCH_START * 1.10;

const PINCH_RELEASE_DISTANCE =
  PINCH_END;

const MIN_ENGAGE_CONFIDENCE = 0.42;
const MIN_RELEASE_CONFIDENCE = 0.18;

const ENGAGE_STABILITY_MS = 14;
const RELEASE_STABILITY_MS = 8;

const STRENGTH_RISE_RATE = 26;
const STRENGTH_FALL_RATE = 34;

const MAX_DT_MS = 100;

function expSmoothing(
  current: number,
  target: number,
  rate: number,
  dtSeconds: number
) {
  const alpha =
    1 -
    Math.exp(
      -rate *
        Math.max(
          0,
          dtSeconds
        )
    );

  return (
    current +
    (
      target -
      current
    ) *
      alpha
  );
}

export class GestureClassifier {
  private pinchState: HysteresisState = {
    active: false
  };

  private readonly fingerStates:
    Record<
      FingerName,
      HysteresisState
    > = {
      thumb: {
        active: false
      },
      index: {
        active: false
      },
      middle: {
        active: false
      },
      ring: {
        active: false
      },
      pinky: {
        active: false
      }
    };

  /*
   * Pinch temporal model.
   */
  private previousPinchDistance = 1;

  private pinchVelocity = 0;

  private pinchConfidence = 0;

  private pinchStrength = 0;

  private lastTimestamp = 0;

  private armStartedAt = 0;

  private releaseStartedAt = 0;

  private pinchStartedAt = 0;

  /*
   * A short-lived confidence memory prevents a single noisy
   * landmark sample from destroying an otherwise stable pinch.
   */
  private confidenceMemory = 0;

  classify(
    landmarks: Point3D[]
  ) {
    const now =
      performance.now();

    const handScale =
      calculateHandScale(
        landmarks
      );

    const thumb =
      landmarks[4];

    const index =
      landmarks[8];

    const middle =
      landmarks[12];

    const ring =
      landmarks[16];

    const pinky =
      landmarks[20];

    const pinchDistance =
      normalizedFingerDistance(
        distance3D(
          thumb,
          index
        ),
        handScale
      );

    /*
     * Calculate closing/opening velocity from the normalized
     * thumb-index distance.
     *
     * Negative = fingers closing.
     * Positive = fingers opening.
     */
    let dt =
      this.lastTimestamp > 0
        ? (
            now -
            this.lastTimestamp
          ) / 1000
        : 1 / 60;

    dt = Math.min(
      Math.max(
        dt,
        1 / 240
      ),
      MAX_DT_MS / 1000
    );

    const rawVelocity =
      (
        pinchDistance -
        this.previousPinchDistance
      ) /
      dt;

    /*
     * Fast enough to preserve responsiveness, but not so fast
     * that one noisy MediaPipe landmark becomes a fake pinch impulse.
     */
    const velocityAlpha =
      1 -
      Math.exp(
        -PINCH_VELOCITY_CUTOFF *
          dt
      );

    this.pinchVelocity +=
      (
        rawVelocity -
        this.pinchVelocity
      ) *
        velocityAlpha;

    this.previousPinchDistance =
      pinchDistance;

    this.lastTimestamp =
      now;

    /*
     * Distance confidence.
     *
     * 0 = clearly open.
     * 1 = clearly inside the pinch zone.
     */
    const distanceConfidence =
      clamp(
        (
          PINCH_END -
          pinchDistance
        ) /
        Math.max(
          0.0001,
          PINCH_END -
            PINCH_START
        ),
        0,
        1
      );

    /*
     * Closing motion is useful evidence when entering a pinch.
     * Opening motion is useful evidence when releasing.
     */
    const closingEvidence =
      clamp(
        -this.pinchVelocity /
          PINCH_VELOCITY_SCALE,
        0,
        1
      );

    const openingEvidence =
      clamp(
        this.pinchVelocity /
          PINCH_VELOCITY_SCALE,
        0,
        1
      );

    /*
     * Stability is highest when the fingers are close and the
     * distance is not violently oscillating.
     */
    const motionMagnitude =
      Math.abs(
        this.pinchVelocity
      );

    const stability =
      clamp(
        1 -
          motionMagnitude /
            2.5,
        0,
        1
      );

    const approachConfidence =
      clamp(
        distanceConfidence *
          0.68 +
          closingEvidence *
          0.17 +
          stability *
          0.15,
        0,
        1
      );

    const releaseConfidence =
      clamp(
        (
          1 -
          distanceConfidence
        ) *
          0.68 +
          openingEvidence *
          0.22 +
          stability *
          0.10,
        0,
        1
      );

    /*
     * Keep a very small confidence memory. This is deliberately
     * short so it cannot create a noticeable interaction delay.
     */
    this.confidenceMemory =
      expSmoothing(
        this.confidenceMemory,
        approachConfidence,
        32,
        dt
      );

    this.updatePinchState(
      pinchDistance,
      approachConfidence,
      releaseConfidence,
      now
    );

    /*
     * Continuous strength.
     *
     * Strength follows the actual distance, while velocity slightly
     * biases it during approach/release. The output remains 0..1.
     */
    let targetStrength =
      clamp(
        (
          PINCH_END -
          pinchDistance
        ) /
        Math.max(
          0.0001,
          PINCH_END -
            PINCH_START
        ),
        0,
        1
      );

    targetStrength = clamp(
      targetStrength +
        closingEvidence *
          0.08 -
        openingEvidence *
          0.10,
      0,
      1
    );

    /*
     * Once a pinch is committed, don't allow tiny distance noise
     * to make its strength visibly flicker.
     */
    if (
      this.pinchState.active &&
      targetStrength < 0.12 &&
      pinchDistance <
        PINCH_RELEASE_DISTANCE
    ) {
      targetStrength =
        0.12;
    }

    this.pinchStrength =
      expSmoothing(
        this.pinchStrength,
        targetStrength,
        targetStrength >
          this.pinchStrength
          ? STRENGTH_RISE_RATE
          : STRENGTH_FALL_RATE,
        dt
      );

    /*
     * Public confidence favors the state currently being evaluated.
     */
    this.pinchConfidence =
      this.pinchState.active
        ? clamp(
            0.62 *
              (
                1 -
                releaseConfidence
              ) +
              0.38 *
                distanceConfidence,
            0,
            1
          )
        : clamp(
            0.72 *
              approachConfidence +
              0.28 *
                this.confidenceMemory,
            0,
            1
          );

    const fingers =
      this.classifyFingers(
        landmarks,
        handScale
      );

    const extendedCount =
      Number(
        fingers.index.extended
      ) +
      Number(
        fingers.middle.extended
      ) +
      Number(
        fingers.ring.extended
      ) +
      Number(
        fingers.pinky.extended
      );

    let pose: Pose =
      "none";

    if (
      this.pinchState.active
    ) {
      pose = "pinch";
    } else if (
      extendedCount >=
      OPEN_FINGER_COUNT
    ) {
      pose = "open";
    } else if (
      fingers.index.extended &&
      !fingers.middle.extended &&
      !fingers.ring.extended &&
      !fingers.pinky.extended
    ) {
      pose = "point";
    } else if (
      extendedCount === 0
    ) {
      pose = "fist";
    }

    return {
      pinch:
        this.pinchState.active,

      pinchStrength:
        clamp(
          this.pinchStrength,
          0,
          1
        ),

      pose,

      fingers,

      handScale,

      pinchDistance
    };
  }

  private updatePinchState(
    distance: number,
    engageConfidence: number,
    releaseConfidence: number,
    now: number
  ) {
    /*
     * OPEN
     *
     * Do not enter the active state merely because one noisy frame
     * crosses the start threshold. ARMING is intentionally tiny.
     */
    if (
      !this.pinchState.active
    ) {
      this.releaseStartedAt =
        0;

      if (
        distance <=
          PINCH_ARM_DISTANCE &&
        (
          engageConfidence >=
            MIN_ENGAGE_CONFIDENCE ||
          this.pinchVelocity <
            -0.35
        )
      ) {
        if (
          this.armStartedAt === 0
        ) {
          this.armStartedAt =
            now;
        }

        const armedFor =
          now -
          this.armStartedAt;

        /*
         * A clearly fast closing motion can commit immediately once
         * the distance itself is deep enough. This prevents the
         * classifier from adding noticeable latency to deliberate
         * pinches.
         */
        const fastClose =
          this.pinchVelocity <
          -1.15;

        if (
          distance <=
            PINCH_COMMIT_DISTANCE &&
          (
            armedFor >=
              ENGAGE_STABILITY_MS ||
            fastClose
          )
        ) {
          this.pinchState.active =
            true;

          this.pinchStartedAt =
            now;

          this.armStartedAt =
            0;
        }
      } else if (
        distance >
          PINCH_REARM_DISTANCE
      ) {
        this.armStartedAt =
          0;
      }

      return;
    }

    /*
     * ACTIVE
     *
     * Release is intentionally asymmetric. Once the fingers clearly
     * separate, we release rapidly instead of adding a long debounce.
     */
    if (
      distance >=
        PINCH_RELEASE_DISTANCE &&
      (
        releaseConfidence >=
          MIN_RELEASE_CONFIDENCE ||
        this.pinchVelocity >
          0.55
      )
    ) {
      if (
        this.releaseStartedAt ===
        0
      ) {
        this.releaseStartedAt =
          now;
      }

      const releasingFor =
        now -
        this.releaseStartedAt;

      const fastRelease =
        this.pinchVelocity >
        1.4;

      if (
        releasingFor >=
          RELEASE_STABILITY_MS ||
        fastRelease
      ) {
        this.pinchState.active =
          false;

        this.pinchStartedAt =
          0;

        this.releaseStartedAt =
          0;
      }

      return;
    }

    /*
     * If the fingers return to the pinch zone while a release was
     * being evaluated, cancel the release immediately.
     */
    if (
      distance <
        PINCH_REARM_DISTANCE
    ) {
      this.releaseStartedAt =
        0;
    }
  }

  private classifyFingers(
    landmarks: Point3D[],
    handScale: number
  ): FingerStates {
    const definitions = {
      index: [5, 6, 7, 8],
      middle: [9, 10, 11, 12],
      ring: [13, 14, 15, 16],
      pinky: [17, 18, 19, 20]
    } as const;

    const output =
      {} as FingerStates;

    const names:
      FingerName[] = [
        "thumb",
        "index",
        "middle",
        "ring",
        "pinky"
      ];

    for (
      const name of names
    ) {
      let extended = false;
      let distance = 0;

      if (
        name ===
        "thumb"
      ) {
        distance =
          normalizedFingerDistance(
            distance3D(
              landmarks[4],
              landmarks[2]
            ),
            handScale
          );

        extended =
          distance > 0.32;
      } else {
        const [
          mcp,
          pip,
          dip,
          tip
        ] =
          definitions[
            name
          ];

        const bend =
          calculateFingerBend(
            landmarks,
            mcp,
            pip,
            dip,
            tip
          );

        extended =
          bend < 0.30;

        distance =
          normalizedFingerDistance(
            distance3D(
              landmarks[tip],
              landmarks[0]
            ),
            handScale
          );
      }

      const touch =
        this.updateFingerTouch(
          name,
          landmarks,
          handScale
        );

      const strength =
        clamp(
          1 -
            (
              this.getTouchDistance(
                name,
                landmarks,
                handScale
              ) -
              FINGER_TOUCH_START
            ) /
            (
              FINGER_TOUCH_END -
              FINGER_TOUCH_START
            ),
          0,
          1
        );

      output[name] = {
        extended,
        touch,
        distance,
        strength
      };
    }

    return output;
  }

  private updateFingerTouch(
    name: FingerName,
    landmarks: Point3D[],
    handScale: number
  ): boolean {
    const distance =
      this.getTouchDistance(
        name,
        landmarks,
        handScale
      );

    return this.updateHysteresis(
      this.fingerStates[name],
      distance,
      FINGER_TOUCH_START,
      FINGER_TOUCH_END
    );
  }

  private getTouchDistance(
    name: FingerName,
    landmarks: Point3D[],
    handScale: number
  ): number {
    const tipIndex =
      name === "thumb"
        ? 4
        : name === "index"
          ? 8
          : name === "middle"
            ? 12
            : name === "ring"
              ? 16
              : 20;

    return normalizedFingerDistance(
      distance3D(
        landmarks[4],
        landmarks[tipIndex]
      ),
      handScale
    );
  }

  private updateHysteresis(
    state: HysteresisState,
    value: number,
    startThreshold: number,
    endThreshold: number
  ): boolean {
    if (
      !state.active
    ) {
      if (
        value <
        startThreshold
      ) {
        state.active =
          true;
      }
    } else if (
      value >
      endThreshold
    ) {
      state.active =
        false;
    }

    return state.active;
  }

  reset() {
    this.pinchState.active =
      false;

    for (
      const state of Object.values(
        this.fingerStates
      )
    ) {
      state.active =
        false;
    }

    this.previousPinchDistance =
      1;

    this.pinchVelocity =
      0;

    this.pinchConfidence =
      0;

    this.pinchStrength =
      0;

    this.lastTimestamp =
      0;

    this.armStartedAt =
      0;

    this.releaseStartedAt =
      0;

    this.pinchStartedAt =
      0;

    this.confidenceMemory =
      0;
  }
}
