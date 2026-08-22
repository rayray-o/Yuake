import {
  calculateFingerBend,
  calculateHandScale,
  clamp,
  distance3D,
  midpoint2D,
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
  Point2D,
  Point3D,
  Pose
} from "./types";

type HysteresisState = {
  active: boolean;
};

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

  classify(
    landmarks: Point3D[]
  ) {
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

    const pinch =
      this.updateHysteresis(
        this.pinchState,
        pinchDistance,
        PINCH_START,
        PINCH_END
      );

    const pinchStrength =
      clamp(
        1 -
          (
            pinchDistance -
            PINCH_START
          ) /
          (
            PINCH_END -
            PINCH_START
          ),
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

    if (pinch) {
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
      pinch,
      pinchStrength,
      pose,
      fingers,
      handScale,
      pinchDistance
    };
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
    if (!state.active) {
      if (
        value <
        startThreshold
      ) {
        state.active = true;
      }
    } else if (
      value >
      endThreshold
    ) {
      state.active = false;
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
      state.active = false;
    }
  }
      }
