import { CursorPredictor } from "./predictor";
import type { GestureHand } from "./types";

export type GrabbableObject = {
  id: string;
  x: number;
  y: number;
  radius: number;
};

export type GrabState = {
  grabbedId: string | null;
  x: number;
  y: number;
};

const GRAB_THRESHOLD = 0.62;
const RELEASE_THRESHOLD = 0.20;

/*
 * The fingertip tracker is already filtered/predicted.
 *
 * Do NOT add another One-Euro filter here.
 * A second filter makes the held object lag behind the
 * fingertip that is already being predicted.
 *
 * Instead:
 *   hand.cursor + hand.velocity
 *          ↓
 *   object offset
 *          ↓
 *   CursorPredictor
 *          ↓
 *   every animation frame
 */

export class GrabManager {
  private objects: GrabbableObject[] = [];

  private grabbedId: string | null = null;

  private grabOffsetX = 0;
  private grabOffsetY = 0;

  private wasPinching = false;

  private objectPredictor =
    new CursorPredictor();

  /*
   * Keep the most recent predicted target available even when
   * MediaPipe has not produced another detection yet.
   */
  private lastTarget = {
    x: 0,
    y: 0
  };

  setObjects(
    objects: GrabbableObject[]
  ) {
    this.objects = objects;
  }

  getObjects() {
    return this.objects;
  }

  getGrabbedId() {
    return this.grabbedId;
  }

  update(
    hand: GestureHand | null,
    timestampMs: number
  ): GrabState {
    if (!hand) {
      this.release();
      return this.currentState();
    }

    const cursorPx = {
      x:
        hand.cursor.x *
        window.innerWidth,

      y:
        hand.cursor.y *
        window.innerHeight
    };

    /*
     * pinchStrength is now the continuous output of the upgraded
     * pinch classifier. Use it as an input signal rather than
     * treating every individual landmark frame as a binary switch.
     */
    const isPinching =
      hand.pinchStrength >=
      GRAB_THRESHOLD;

    const shouldRelease =
      !hand.pinch ||
      hand.pinchStrength <=
        RELEASE_THRESHOLD;

    /*
     * Rising edge:
     * grab only once when a deliberate pinch enters the active zone.
     */
    if (
      !this.wasPinching &&
      isPinching &&
      !this.grabbedId
    ) {
      const target =
        this.findNearest(
          cursorPx.x,
          cursorPx.y
        );

      if (target) {
        this.grabbedId =
          target.id;

        /*
         * Preserve the exact point where the object was grabbed.
         * This prevents the object from snapping its center to the
         * fingertip.
         */
        this.grabOffsetX =
          target.x -
          cursorPx.x;

        this.grabOffsetY =
          target.y -
          cursorPx.y;

        this.lastTarget = {
          x: target.x,
          y: target.y
        };

        /*
         * Start prediction directly from the already-processed
         * fingertip signal.
         */
        this.objectPredictor.clear();

        this.objectPredictor.update(
          {
            x: target.x,
            y: target.y
          },
          {
            x:
              hand.velocity.x,
            y:
              hand.velocity.y
          },
          timestampMs
        );
      }
    }

    /*
     * Release is intentionally fast once the upgraded pinch signal
     * clearly says the fingers are open.
     */
    if (
      this.grabbedId &&
      shouldRelease
    ) {
      this.release();
    }

    this.wasPinching =
      isPinching;

    if (
      this.grabbedId
    ) {
      const targetX =
        cursorPx.x +
        this.grabOffsetX;

      const targetY =
        cursorPx.y +
        this.grabOffsetY;

      /*
       * No secondary smoothing.
       *
       * The tracker already gives us a filtered cursor and a clean
       * velocity. Feeding both straight into the predictor keeps
       * the object aligned with the same motion as the fingertip.
       */
      this.lastTarget = {
        x: targetX,
        y: targetY
      };

      this.objectPredictor.update(
        this.lastTarget,
        {
          x:
            hand.velocity.x,
          y:
            hand.velocity.y
        },
        timestampMs
      );

      /*
       * Update the backing object immediately too.
       * This keeps state correct even before the next RAF render.
       */
      const object =
        this.objects.find(
          candidate =>
            candidate.id ===
            this.grabbedId
        );

      if (object) {
        object.x =
          targetX;

        object.y =
          targetY;
      }
    }

    return this.currentState();
  }

  private findNearest(
    px: number,
    py: number
  ): GrabbableObject | null {
    let best:
      GrabbableObject | null =
      null;

    let bestDistance =
      Infinity;

    for (
      const object of
        this.objects
    ) {
      const d =
        Math.hypot(
          px -
            object.x,
          py -
            object.y
        );

      if (
        d <=
          object.radius &&
        d <
          bestDistance
      ) {
        best =
          object;

        bestDistance =
          d;
      }
    }

    return best;
  }

  private release() {
    if (
      this.grabbedId
    ) {
      this.grabbedId =
        null;

      this.objectPredictor.clear();
    }

    this.wasPinching =
      false;
  }

  /*
   * Called from the animation/render loop.
   *
   * This is the important part for the "glide":
   * the object is predicted at display refresh rate rather than
   * waiting for the next MediaPipe detection.
   */
  predict(
    nowMs: number
  ) {
    if (
      !this.grabbedId
    ) {
      return null;
    }

    const predicted =
      this.objectPredictor.predict(
        nowMs
      );

    if (
      !predicted
    ) {
      return null;
    }

    const object =
      this.objects.find(
        candidate =>
          candidate.id ===
          this.grabbedId
      );

    if (object) {
      object.x =
        predicted.x;

      object.y =
        predicted.y;
    }

    return {
      id:
        this.grabbedId,

      x:
        predicted.x,

      y:
        predicted.y
    };
  }

  private currentState(): GrabState {
    if (
      !this.grabbedId
    ) {
      return {
        grabbedId:
          null,
        x: 0,
        y: 0
      };
    }

    const object =
      this.objects.find(
        candidate =>
          candidate.id ===
          this.grabbedId
      );

    return {
      grabbedId:
        this.grabbedId,

      x:
        object?.x ??
        this.lastTarget.x,

      y:
        object?.y ??
        this.lastTarget.y
    };
  }
}
