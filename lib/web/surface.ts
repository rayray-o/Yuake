import type { Point2D, Point3D } from "../gesture/types";
import type { WebTarget } from "./types";

const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));

/**
 * RGB-only environment estimator.
 * It deliberately reports an ESTIMATE, never fake metric depth.
 * A future depth/segmentation provider can replace this class.
 */
export class SurfaceEstimator {
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;
  private lastSample = 0;

  private ensureCanvas() {
    if (!this.canvas) {
      this.canvas = document.createElement("canvas");
      this.canvas.width = 96;
      this.canvas.height = 54;
      this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
    }
  }

  estimate(video: HTMLVideoElement, aim: Point2D, normal: Point3D, now: number): WebTarget {
    const fallback: WebTarget = {
      point: { x: clamp(aim.x, 0.04, 0.96), y: clamp(aim.y, 0.04, 0.96) },
      depth: 0.5,
      confidence: 0.32,
      normal
    };

    if (now - this.lastSample < 55 || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return fallback;
    this.lastSample = now;
    this.ensureCanvas();
    if (!this.ctx || !this.canvas || !video.videoWidth) return fallback;

    try {
      this.ctx.drawImage(video, 0, 0, this.canvas.width, this.canvas.height);
      const image = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
      const ax = Math.round(aim.x * (this.canvas.width - 1));
      const ay = Math.round(aim.y * (this.canvas.height - 1));

      let bestScore = -Infinity;
      let bestX = ax;
      let bestY = ay;
      const radius = 18;

      for (let oy = -radius; oy <= radius; oy += 3) {
        for (let ox = -radius; ox <= radius; ox += 3) {
          const x = clamp(ax + ox, 1, this.canvas.width - 2);
          const y = clamp(ay + oy, 1, this.canvas.height - 2);
          const i = (y * this.canvas.width + x) * 4;
          const left = ((y * this.canvas.width + (x - 1)) * 4);
          const right = ((y * this.canvas.width + (x + 1)) * 4);
          const up = (((y - 1) * this.canvas.width + x) * 4);
          const down = (((y + 1) * this.canvas.width + x) * 4);
          const gx = Math.abs(image.data[right] - image.data[left]);
          const gy = Math.abs(image.data[down] - image.data[up]);
          const edge = (gx + gy) / 510;
          const distance = Math.hypot(ox, oy) / radius;
          const score = edge * 0.35 - distance * 0.18;
          if (score > bestScore) {
            bestScore = score;
            bestX = x;
            bestY = y;
          }
        }
      }

      return {
        point: { x: bestX / (this.canvas.width - 1), y: bestY / (this.canvas.height - 1) },
        depth: clamp(0.30 + (1 - clamp(bestScore, -0.2, 0.8)) * 0.5, 0.18, 0.82),
        confidence: clamp(0.42 + Math.max(0, bestScore) * 0.7, 0.32, 0.82),
        normal
      };
    } catch {
      return fallback;
    }
  }
}
