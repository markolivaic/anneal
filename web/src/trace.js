/*
 * The force-convergence trace: the baseline rule running under both columns,
 * and the only thing on the page that moves by itself.
 *
 * Max force on a log axis against step number, drawn in the accent because it
 * is model output. The convergence threshold is a dotted ink line; that one is
 * a setting, not a prediction, so it stays in ink.
 *
 * Log scale is not decoration. Forces fall by two or three orders of magnitude
 * over a relaxation, and on a linear axis everything after the first few steps
 * is flat against zero, which is exactly the part worth watching.
 */

const PADDING = { left: 44, right: 10, top: 10, bottom: 18 };

export class ForceTrace {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.steps = [];
    this.fmax = 0.1;
    // Same reason as the viewer: the first draw happens before layout, so the
    // canvas must be told when its box actually gets a size.
    new ResizeObserver(() => this.draw()).observe(canvas.parentElement);
  }

  reset(fmax = 0.1) {
    this.steps = [];
    this.fmax = fmax;
    this.draw();
  }

  push(step) {
    this.steps.push(step);
    this.draw();
  }

  draw() {
    const canvas = this.canvas;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (!width || !height) return;

    canvas.width = width * ratio;
    canvas.height = height * ratio;
    const ctx = this.ctx;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const styles = getComputedStyle(document.documentElement);
    const accent = styles.getPropertyValue("--accent").trim() || "#c2410c";
    const rule = styles.getPropertyValue("--rule").trim() || "#c9c4b7";
    const faint = styles.getPropertyValue("--ink-faint").trim() || "#8b8778";

    const plotWidth = width - PADDING.left - PADDING.right;
    const plotHeight = height - PADDING.top - PADDING.bottom;

    ctx.font = "9px ui-monospace, monospace";
    ctx.fillStyle = faint;
    ctx.strokeStyle = rule;
    ctx.lineWidth = 1;

    if (this.steps.length === 0) {
      ctx.fillText("force convergence, press relax", PADDING.left, height / 2);
      this._axisFrame(ctx, width, height, rule);
      return;
    }

    const forces = this.steps.map((s) => Math.max(s.max_force_ev_per_a, 1e-4));
    const high = Math.max(...forces, this.fmax * 2);
    const low = Math.min(...forces, this.fmax * 0.5);
    const logHigh = Math.log10(high);
    const logLow = Math.log10(low);
    const span = Math.max(logHigh - logLow, 0.4);

    const xOf = (i) =>
      PADDING.left +
      (this.steps.length === 1 ? 0 : (i / (this.steps.length - 1)) * plotWidth);
    const yOf = (f) =>
      PADDING.top + plotHeight - ((Math.log10(f) - logLow) / span) * plotHeight;

    // Decade gridlines, so the reader can see it really is orders of magnitude.
    ctx.strokeStyle = rule;
    ctx.setLineDash([]);
    for (let d = Math.ceil(logLow); d <= Math.floor(logHigh); d++) {
      const y = yOf(Math.pow(10, d));
      ctx.globalAlpha = 0.5;
      ctx.beginPath();
      ctx.moveTo(PADDING.left, y);
      ctx.lineTo(width - PADDING.right, y);
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = faint;
      ctx.fillText(`1e${d}`, 4, y + 3);
    }

    // The convergence threshold: a setting, so it stays in ink.
    const thresholdY = yOf(this.fmax);
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = faint;
    ctx.beginPath();
    ctx.moveTo(PADDING.left, thresholdY);
    ctx.lineTo(width - PADDING.right, thresholdY);
    ctx.stroke();
    ctx.setLineDash([]);
    // Anchored left, not right: the trace descends toward this line and ends
    // near the right edge, so a label there sits on top of the data.
    ctx.fillText(`fmax ${this.fmax}`, PADDING.left + 4, thresholdY - 3);

    // The trace itself: model output, so accent.
    ctx.strokeStyle = accent;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    forces.forEach((f, i) => {
      const x = xOf(i);
      const y = yOf(f);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Leading dot: where the relaxation is right now.
    const lastX = xOf(forces.length - 1);
    const lastY = yOf(forces[forces.length - 1]);
    ctx.fillStyle = accent;
    ctx.beginPath();
    ctx.arc(lastX, lastY, 2.5, 0, Math.PI * 2);
    ctx.fill();

    this._axisFrame(ctx, width, height, rule);

    ctx.fillStyle = faint;
    ctx.fillText(`step ${this.steps.length - 1}`, PADDING.left, height - 5);
  }

  _axisFrame(ctx, width, height, rule) {
    ctx.strokeStyle = rule;
    ctx.lineWidth = 1;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(PADDING.left, PADDING.top);
    ctx.lineTo(PADDING.left, height - PADDING.bottom);
    ctx.lineTo(width - PADDING.right, height - PADDING.bottom);
    ctx.stroke();
  }
}
