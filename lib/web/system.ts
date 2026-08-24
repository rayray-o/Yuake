import type { GestureHand } from "../gesture/types";
import { SpiderPoseDetector, type SpiderPoseResult } from "./pose";
import { SurfaceEstimator } from "./surface";
import { WebPhysics } from "./physics";
import { WebRenderer } from "./renderer";
import type { WebDiagnostics, WebUpdateInput } from "./types";

export class YuakeWebSystem {
  readonly pose = new SpiderPoseDetector();
  readonly physics = new WebPhysics();
  readonly surface = new SurfaceEstimator();
  readonly renderer = new WebRenderer();

  private lastNow = 0;
  private lastPose: SpiderPoseResult = { active: false, confidence: 0, indexExtended: 0, pinkyExtended: 0, middleFolded: 0, ringFolded: 0 };

  mount(parent: HTMLElement) {
    this.renderer.mount(parent);
    window.addEventListener("resize", this.resize);
  }

  private resize = () => this.renderer.resize();

  update(input: WebUpdateInput) {
    const { hand, video, now, viewportWidth, viewportHeight } = input;
    const dt = this.lastNow ? Math.min((now - this.lastNow) / 1000, 0.033) : 1 / 60;
    this.lastNow = now;

    this.lastPose = this.pose.update(hand, now);

    if (this.lastPose.active && this.physics.state === "idle" && hand) {
      const origin = {
        x: hand.cursor.x * viewportWidth,
        y: hand.cursor.y * viewportHeight
      };

      const normal = hand.palmNormal;
      const aimLength = 0.36;
      const aim = {
        x: Math.max(0.02, Math.min(0.98, hand.cursor.x + normal.x * aimLength)),
        y: Math.max(0.02, Math.min(0.98, hand.cursor.y - normal.y * aimLength))
      };

      const target = video
        ? this.surface.estimate(video, aim, normal, now)
        : { point: aim, depth: 0.5, confidence: 0.3, normal };

      this.physics.fire(origin, {
        x: hand.velocity.x * viewportWidth,
        y: hand.velocity.y * viewportHeight
      }, {
        ...target,
        point: {
          x: target.point.x * viewportWidth,
          y: target.point.y * viewportHeight
        }
      });
    }

    if (!this.lastPose.active && (this.physics.state === "attached" || this.physics.state === "firing")) {
      this.physics.release();
    }

    if (hand) {
      this.physics.update(
        { x: hand.cursor.x * viewportWidth, y: hand.cursor.y * viewportHeight },
        { x: hand.velocity.x * viewportWidth, y: hand.velocity.y * viewportHeight },
        dt
      );
    }

    this.renderer.draw(this.physics);
  }

  getDiagnostics(): WebDiagnostics {
    return {
      poseConfidence: this.lastPose.confidence,
      targetConfidence: this.physics.target?.confidence ?? 0,
      speed: Math.hypot(this.physics.velocity.x, this.physics.velocity.y),
      distance: this.physics.length,
      tension: this.physics.tension,
      state: this.physics.state
    };
  }

  reset() {
    this.pose.reset();
    this.physics.reset();
    this.renderer.clear();
    this.lastNow = 0;
  }

  destroy() {
    window.removeEventListener("resize", this.resize);
    this.reset();
    this.renderer.destroy();
  }
}
