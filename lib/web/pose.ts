import type { GestureHand } from "../gesture/types";

export type SpiderPoseResult = {
  active: boolean;
  confidence: number;
  indexExtended: number;
  pinkyExtended: number;
  middleFolded: number;
  ringFolded: number;
};

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

function extendedScore(hand: GestureHand, name: "index" | "middle" | "ring" | "pinky") {
  const finger = hand.fingers[name];
  return finger.extended ? clamp01(0.72 + finger.strength * 0.28) : clamp01(finger.strength * 0.25);
}

function foldedScore(hand: GestureHand, name: "middle" | "ring") {
  const finger = hand.fingers[name];
  return finger.extended ? clamp01(1 - finger.strength) * 0.35 : clamp01(0.72 + (1 - finger.strength) * 0.28);
}

export class SpiderPoseDetector {
  private confidence = 0;
  private active = false;
  private candidateSince = 0;
  private releaseSince = 0;

  update(hand: GestureHand | null, now: number): SpiderPoseResult {
    if (!hand || !hand.visible) {
      this.confidence *= 0.72;
      this.active = false;
      this.candidateSince = 0;
      this.releaseSince = 0;
      return { active: false, confidence: this.confidence, indexExtended: 0, pinkyExtended: 0, middleFolded: 0, ringFolded: 0 };
    }

    const indexExtended = extendedScore(hand, "index");
    const pinkyExtended = extendedScore(hand, "pinky");
    const middleFolded = foldedScore(hand, "middle");
    const ringFolded = foldedScore(hand, "ring");

    const geometryScore =
      indexExtended * 0.29 +
      pinkyExtended * 0.29 +
      middleFolded * 0.21 +
      ringFolded * 0.21;

    const handConfidence = clamp01(hand.confidence);
    const target = geometryScore * (0.72 + handConfidence * 0.28);
    const alpha = 1 - Math.exp(-24 / 60);
    this.confidence += (target - this.confidence) * alpha;

    const engage =
      indexExtended > 0.66 &&
      pinkyExtended > 0.66 &&
      middleFolded > 0.62 &&
      ringFolded > 0.62 &&
      this.confidence > 0.72;

    if (!this.active) {
      if (engage) {
        if (!this.candidateSince) this.candidateSince = now;
        if (now - this.candidateSince >= 24) {
          this.active = true;
          this.releaseSince = 0;
        }
      } else {
        this.candidateSince = 0;
      }
    } else {
      const release = this.confidence < 0.50 || middleFolded < 0.42 || ringFolded < 0.42;
      if (release) {
        if (!this.releaseSince) this.releaseSince = now;
        if (now - this.releaseSince >= 28) {
          this.active = false;
          this.candidateSince = 0;
          this.releaseSince = 0;
        }
      } else {
        this.releaseSince = 0;
      }
    }

    return { active: this.active, confidence: this.confidence, indexExtended, pinkyExtended, middleFolded, ringFolded };
  }

  reset() {
    this.confidence = 0;
    this.active = false;
    this.candidateSince = 0;
    this.releaseSince = 0;
  }
}
