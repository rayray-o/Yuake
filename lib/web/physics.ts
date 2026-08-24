import type { Point2D } from "../gesture/types";
import type { WebConfig, WebSample, WebState, WebTarget } from "./types";

const DEFAULTS: WebConfig = {
  maxRangePx: 1400,
  launchSpeedPxPerSecond: 3200,
  retractSpeedPxPerSecond: 4200,
  drag: 2.8,
  spring: 34,
  damping: 8.5
};

const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

export class WebPhysics {
  readonly config: WebConfig;
  state: WebState = "idle";
  origin: Point2D = { x: 0, y: 0 };
  handVelocity: Point2D = { x: 0, y: 0 };
  target: WebTarget | null = null;
  tip: Point2D = { x: 0, y: 0 };
  velocity: Point2D = { x: 0, y: 0 };
  length = 0;
  restLength = 0;
  tension = 0;
  age = 0;

  constructor(config: Partial<WebConfig> = {}) {
    this.config = { ...DEFAULTS, ...config };
  }

  fire(origin: Point2D, velocity: Point2D, target: WebTarget) {
    this.state = "firing";
    this.origin = { ...origin };
    this.handVelocity = { ...velocity };
    this.target = target;
    this.tip = { ...origin };
    this.velocity = {
      x: velocity.x * 0.35,
      y: velocity.y * 0.35
    };
    this.length = 0;
    this.restLength = Math.hypot(target.point.x * this.config.maxRangePx - origin.x, target.point.y * this.config.maxRangePx - origin.y);
    this.tension = 0;
    this.age = 0;
  }

  update(origin: Point2D, handVelocity: Point2D, dt: number) {
    const safeDt = clamp(dt, 0, 0.033);
    this.origin = { ...origin };
    this.handVelocity = { ...handVelocity };
    this.age += safeDt;

    if (!this.target || this.state === "idle") return;

    const tx = this.target.point.x;
    const ty = this.target.point.y;

    if (this.state === "firing") {
      const dx = tx - this.tip.x;
      const dy = ty - this.tip.y;
      const dist = Math.hypot(dx, dy);
      const inv = dist > 0.001 ? 1 / dist : 0;
      const speed = this.config.launchSpeedPxPerSecond * (0.92 + this.target.confidence * 0.16);
      this.velocity.x += (dx * inv * speed - this.velocity.x) * (1 - Math.exp(-18 * safeDt));
      this.velocity.y += (dy * inv * speed - this.velocity.y) * (1 - Math.exp(-18 * safeDt));
      const drag = Math.exp(-this.config.drag * safeDt);
      this.velocity.x *= drag;
      this.velocity.y *= drag;
      this.tip.x += this.velocity.x * safeDt;
      this.tip.y += this.velocity.y * safeDt;
      this.length = Math.hypot(this.tip.x - this.origin.x, this.tip.y - this.origin.y);

      if (dist < 28 || this.length > this.config.maxRangePx || this.age > 0.8) {
        this.tip = { x: tx, y: ty };
        this.restLength = Math.max(1, Math.hypot(tx - this.origin.x, ty - this.origin.y));
        this.state = "attached";
        this.velocity = { x: 0, y: 0 };
      }
      return;
    }

    if (this.state === "attached") {
      const dx = tx - origin.x;
      const dy = ty - origin.y;
      const distance = Math.hypot(dx, dy);
      const stretch = Math.max(0, distance - this.restLength);
      this.tension = clamp(stretch / Math.max(this.restLength * 0.18, 30), 0, 1);

      const springForce = stretch * this.config.spring;
      const nx = distance > 0.001 ? dx / distance : 0;
      const ny = distance > 0.001 ? dy / distance : 0;
      this.velocity.x += nx * springForce * safeDt;
      this.velocity.y += ny * springForce * safeDt;
      this.velocity.x *= Math.exp(-this.config.damping * safeDt);
      this.velocity.y *= Math.exp(-this.config.damping * safeDt);
      this.tip.x += this.velocity.x * safeDt;
      this.tip.y += this.velocity.y * safeDt;
      this.tip.x = lerp(this.tip.x, tx, 1 - Math.exp(-12 * safeDt));
      this.tip.y = lerp(this.tip.y, ty, 1 - Math.exp(-12 * safeDt));
      this.length = Math.hypot(this.tip.x - origin.x, this.tip.y - origin.y);
      return;
    }

    if (this.state === "retracting") {
      const dx = origin.x - this.tip.x;
      const dy = origin.y - this.tip.y;
      const dist = Math.hypot(dx, dy);
      if (dist < 20) {
        this.state = "idle";
        this.target = null;
        this.tension = 0;
        return;
      }
      const inv = dist > 0.001 ? 1 / dist : 0;
      this.tip.x += dx * inv * this.config.retractSpeedPxPerSecond * safeDt;
      this.tip.y += dy * inv * this.config.retractSpeedPxPerSecond * safeDt;
      this.length = dist;
      this.tension = Math.max(0, this.tension - safeDt * 4);
    }
  }

  release() {
    if (this.state !== "idle") this.state = "retracting";
  }

  reset() {
    this.state = "idle";
    this.target = null;
    this.tension = 0;
    this.length = 0;
  }

  samples(count = 22): WebSample[] {
    const points: WebSample[] = [];
    const dx = this.tip.x - this.origin.x;
    const dy = this.tip.y - this.origin.y;
    const distance = Math.hypot(dx, dy);
    const nx = distance > 0.001 ? -dy / distance : 0;
    const ny = distance > 0.001 ? dx / distance : 0;
    const sag = this.state === "attached" ? Math.min(48, distance * 0.035) * (1 - this.tension * 0.85) : 0;

    for (let i = 0; i < count; i++) {
      const t = i / (count - 1);
      const arc = Math.sin(Math.PI * t) * sag;
      points.push({ x: this.origin.x + dx * t + nx * arc, y: this.origin.y + dy * t + ny * arc, z: this.target?.depth ?? 0.5 });
    }
    return points;
  }
}
