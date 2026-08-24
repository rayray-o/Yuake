import type { WebPhysics } from "./physics";

export class WebRenderer {
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;
  private dpr = 1;

  mount(parent: HTMLElement) {
    if (this.canvas) return;
    const canvas = document.createElement("canvas");
    canvas.className = "yuakeWebCanvas";
    Object.assign(canvas.style, {
      position: "fixed",
      inset: "0",
      width: "100%",
      height: "100%",
      pointerEvents: "none",
      zIndex: "20"
    });
    parent.appendChild(canvas);
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.resize();
  }

  resize() {
    if (!this.canvas) return;
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = Math.floor(innerWidth * this.dpr);
    this.canvas.height = Math.floor(innerHeight * this.dpr);
  }

  draw(physics: WebPhysics) {
    if (!this.ctx || !this.canvas || physics.state === "idle") return;
    const ctx = this.ctx;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, innerWidth, innerHeight);

    const points = physics.samples(28);
    if (points.length < 2) return;

    const project = (p: { x: number; y: number }) => ({ x: p.x, y: p.y });

    const end = project(points[points.length - 1]);

    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.globalCompositeOperation = "lighter";

    ctx.beginPath();
    points.forEach((p, i) => {
      const q = project(p);
      if (i === 0) ctx.moveTo(q.x, q.y); else ctx.lineTo(q.x, q.y);
    });
    ctx.strokeStyle = "rgba(235,240,255,0.16)";
    ctx.lineWidth = 7;
    ctx.stroke();

    ctx.beginPath();
    points.forEach((p, i) => {
      const q = project(p);
      if (i === 0) ctx.moveTo(q.x, q.y); else ctx.lineTo(q.x, q.y);
    });
    ctx.strokeStyle = "rgba(255,255,255,0.88)";
    ctx.lineWidth = 1.8 + physics.tension * 0.9;
    ctx.stroke();

    ctx.beginPath();
    points.forEach((p, i) => {
      if (i % 2) return;
      const q = project(p);
      const wobble = Math.sin(performance.now() * 0.018 + i * 1.7) * (1 - physics.tension) * 0.7;
      if (i === 0) ctx.moveTo(q.x, q.y + wobble); else ctx.lineTo(q.x, q.y + wobble);
    });
    ctx.strokeStyle = "rgba(255,255,255,0.34)";
    ctx.lineWidth = 0.7;
    ctx.stroke();

    if (physics.state === "attached") {
      ctx.beginPath();
      ctx.arc(end.x, end.y, 5 + physics.tension * 5, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(255,255,255,0.6)";
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    ctx.restore();
  }

  clear() {
    if (!this.ctx || !this.canvas) return;
    this.ctx.setTransform(1, 0, 0, 1, 0, 0);
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  destroy() {
    this.canvas?.remove();
    this.canvas = null;
    this.ctx = null;
  }
}
