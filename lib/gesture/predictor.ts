/*
 * --------------------------------------------------
 * CURSOR PREDICTOR
 * --------------------------------------------------
 *
 * MediaPipe only produces a new fingertip position
 * whenever inference finishes — that might be far
 * slower than the screen's refresh rate (e.g. 15fps
 * inference vs a 60fps+ display).
 *
 * If we only redraw the cursor when a new detection
 * arrives, the dot visibly "teleports" between sparse
 * positions instead of gliding.
 *
 * This class fills the gap: every time a real
 * detection arrives, it records position + velocity.
 * On every animation frame (call predict()), it
 * extrapolates forward using that velocity, so the
 * dot keeps moving smoothly even between detections.
 *
 * When the NEXT real detection arrives, it simply
 * becomes the new anchor point — since the
 * extrapolation was tracking real velocity, the jump
 * is small or invisible, not a teleport.
 */

type Point2D = {
  x: number;
  y: number;
};

const MAX_EXTRAPOLATION_MS = 220;

export class CursorPredictor {
  private anchor: Point2D | null = null;
  private velocity: Point2D = { x: 0, y: 0 };
  private anchorTimeMs = 0;

  /*
   * Call this every time the tracker produces a
   * real detection (i.e. inside processFrame, right
   * after tracker.process()).
   */
  update(cursor: Point2D, velocityNormalizedPerSecond: Point2D, nowMs: number) {
    this.anchor = { x: cursor.x, y: cursor.y };
    this.velocity = velocityNormalizedPerSecond;
    this.anchorTimeMs = nowMs;
  }

  /*
   * Call this on EVERY animation frame, regardless of
   * whether a new detection happened this frame.
   */
  predict(nowMs: number): Point2D | null {
    if (!this.anchor) {
      return null;
    }

    const elapsedMs = Math.min(
      Math.max(nowMs - this.anchorTimeMs, 0),
      MAX_EXTRAPOLATION_MS
    );

    const elapsedSeconds = elapsedMs / 1000;

    return {
      x: this.anchor.x + this.velocity.x * elapsedSeconds,
      y: this.anchor.y + this.velocity.y * elapsedSeconds
    };
  }

  clear() {
    this.anchor = null;
    this.velocity = { x: 0, y: 0 };
    this.anchorTimeMs = 0;
  }
}
