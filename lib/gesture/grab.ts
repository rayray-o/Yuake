import { OneEuroPointFilter } from "./oneEuro";
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

const GRAB_THRESHOLD = 0.7;
const RELEASE_THRESHOLD = 0.4;

const OBJECT_FILTER_MIN_CUTOFF = 0.6;
const OBJECT_FILTER_BETA = 0.02;
const OBJECT_FILTER_D_CUTOFF = 1.0;

function distance(ax: number, ay: number, bx: number, by: number) {
  return Math.hypot(ax - bx, ay - by);
}

export class GrabManager {
  private objects: GrabbableObject[] = [];

  private grabbedId: string | null = null;

  private grabOffsetX = 0;
  private grabOffsetY = 0;

  private wasPinching = false;

  private objectFilter = new OneEuroPointFilter({
    minCutoff: OBJECT_FILTER_MIN_CUTOFF,
    beta: OBJECT_FILTER_BETA,
    dCutoff: OBJECT_FILTER_D_CUTOFF
  });

  setObjects(objects: GrabbableObject[]) {
    this.objects = objects;
  }

  getObjects() {
    return this.objects;
  }

  getGrabbedId() {
    return this.grabbedId;
  }

  update(hand: GestureHand | null, timestampMs: number): GrabState {
    if (!hand) {
      this.release();
      return this.currentState();
    }

    const cursorPx = {
      x: hand.cursor.x * window.innerWidth,
      y: hand.cursor.y * window.innerHeight
    };

    const isPinching = hand.pinchStrength >= GRAB_THRESHOLD;
    const shouldRelease = hand.pinchStrength <= RELEASE_THRESHOLD;

    /*
     * Grab: only trigger on the RISING EDGE of a
     * strong pinch (not every frame while pinching),
     * and only if a grabbable object is near the
     * cursor at that exact moment.
     */
    if (!this.wasPinching && isPinching && !this.grabbedId) {
      const target = this.findNearest(cursorPx.x, cursorPx.y);

      if (target) {
        this.grabbedId = target.id;

        this.grabOffsetX = target.x - cursorPx.x;
        this.grabOffsetY = target.y - cursorPx.y;

        this.objectFilter.reset();
      }
    }

    /*
     * Release: use a LOWER threshold than the grab
     * threshold (hysteresis). Without this gap, a
     * pinch strength hovering right at the boundary
     * would grab/release rapidly every frame.
     */
    if (this.grabbedId && shouldRelease) {
      this.release();
    }

    this.wasPinching = isPinching;

    if (this.grabbedId) {
      const rawTargetX = cursorPx.x + this.grabOffsetX;
      const rawTargetY = cursorPx.y + this.grabOffsetY;

      /*
       * Smooth the CARRIED object separately from the
       * fingertip cursor. Hand jitter while pinching is
       * usually worse than while pointing, and an object
       * being "held" should feel weightier / less twitchy
       * than the bare cursor dot.
       */
      const filtered = this.objectFilter.filter(
        { x: rawTargetX, y: rawTargetY },
        timestampMs / 1000
      );

      const object = this.objects.find(o => o.id === this.grabbedId);

      if (object) {
        object.x = filtered.x;
        object.y = filtered.y;
      }
    }

    return this.currentState();
  }

  private findNearest(px: number, py: number): GrabbableObject | null {
    let best: GrabbableObject | null = null;
    let bestDistance = Infinity;

    for (const object of this.objects) {
      const d = distance(px, py, object.x, object.y);

      if (d <= object.radius && d < bestDistance) {
        best = object;
        bestDistance = d;
      }
    }

    return best;
  }

  private release() {
    if (this.grabbedId) {
      this.grabbedId = null;
      this.objectFilter.reset();
    }

    this.wasPinching = false;
  }

  private currentState(): GrabState {
    if (!this.grabbedId) {
      return { grabbedId: null, x: 0, y: 0 };
    }

    const object = this.objects.find(o => o.id === this.grabbedId);

    return {
      grabbedId: this.grabbedId,
      x: object?.x ?? 0,
      y: object?.y ?? 0
    };
  }
}
