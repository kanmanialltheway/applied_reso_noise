// Audio transport with an instant A/B switch between original and processed audio.
import { icon } from "./icons.js";
import { fmtTime } from "./ui.js";

export class Player {
  constructor(root, { onTime, onMode } = {}) {
    this.root = root; this.onTime = onTime; this.onMode = onMode;
    this.mode = "original"; this.tracks = {}; this.duration = 0;
    root.innerHTML = `
      <div class="player">
        <div class="player-bar">
          <button class="skip" data-act="back" aria-label="Back 5 seconds" title="Back 5 s">${icon("back")}</button>
          <button class="play-btn" data-act="toggle" aria-label="Play">${icon("play")}</button>
          <button class="skip" data-act="fwd" aria-label="Forward 5 seconds" title="Forward 5 s">${icon("fwd")}</button>
          <input class="seek" type="range" min="0" max="1000" value="0" aria-label="Seek" />
          <span class="time" aria-live="off">0:00 / 0:00</span>
          <label class="vol" title="Volume">${icon("volume")}<input type="range" min="0" max="100" value="85" aria-label="Volume" /></label>
        </div>
        <div class="row">
          <div class="ab" role="group" aria-label="Listen to">
            <button class="orig" data-mode="original" aria-pressed="true">${icon("headphones")}Original</button>
            <button data-mode="processed" aria-pressed="false" disabled>${icon("sparkle")}Processed</button>
          </div>
          <span class="small muted" data-role="status"></span>
        </div>
      </div>`;
    this.$toggle = root.querySelector('[data-act="toggle"]');
    this.$seek = root.querySelector(".seek");
    this.$time = root.querySelector(".time");
    this.$status = root.querySelector('[data-role="status"]');
    root.addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      if (b.dataset.act === "toggle") this.toggle();
      if (b.dataset.act === "back") this.seek(this.time - 5);
      if (b.dataset.act === "fwd") this.seek(this.time + 5);
      if (b.dataset.mode) this.setMode(b.dataset.mode);
    });
    this.$seek.addEventListener("input", () => this.seek((this.$seek.value / 1000) * this.duration));
    root.querySelector(".vol input").addEventListener("input", (e) => this.setVolume(e.target.value / 100));
    this.volume = 0.85;
  }

  load({ original, processed }) {
    this.dispose();
    this.tracks = {};
    for (const [k, url] of Object.entries({ original, processed })) {
      if (!url) continue;
      const a = new Audio();
      a.preload = "auto"; a.src = url; a.volume = this.volume;
      a.addEventListener("loadedmetadata", () => { if (k === "original" || !this.duration) { this.duration = a.duration; this.paint(); } });
      a.addEventListener("ended", () => { this.setPlaying(false); this.seek(0); });
      a.addEventListener("error", () => { this.$status.textContent = "This audio could not be played in the browser."; });
      this.tracks[k] = a;
    }
    this.root.querySelector('[data-mode="processed"]').disabled = !processed;
    this.mode = "original";
    this.syncAB();
    this.paint();
  }

  get cur() { return this.tracks[this.mode]; }
  get time() { return this.cur ? this.cur.currentTime : 0; }
  get playing() { return !!this.cur && !this.cur.paused; }

  async toggle() {
    if (!this.cur) return;
    if (this.cur.paused) {
      try { await this.cur.play(); } catch (e) { this.$status.textContent = "Playback was blocked by the browser."; return; }
      this.setPlaying(true); this.loop();
    } else { this.cur.pause(); this.setPlaying(false); }
  }

  setPlaying(p) {
    this.$toggle.innerHTML = icon(p ? "pause" : "play");
    this.$toggle.setAttribute("aria-label", p ? "Pause" : "Play");
    if (!p) cancelAnimationFrame(this.raf);
  }

  loop() {
    const tick = () => { this.paint(); if (this.playing) this.raf = requestAnimationFrame(tick); };
    tick();
  }

  seek(t) {
    if (!this.cur) return;
    t = Math.max(0, Math.min(this.duration || 0, t));
    for (const a of Object.values(this.tracks)) a.currentTime = t;
    this.paint();
  }

  setMode(mode) {
    if (!this.tracks[mode] || mode === this.mode) return;
    const t = this.time, was = this.playing;
    this.cur.pause();
    this.mode = mode;
    this.cur.currentTime = t;
    if (was) this.cur.play().catch(() => {});
    this.syncAB();
    this.onMode && this.onMode(mode);
  }

  syncAB() {
    this.root.querySelectorAll("[data-mode]").forEach((b) => b.setAttribute("aria-pressed", b.dataset.mode === this.mode));
    this.$status.textContent = this.mode === "processed" ? "Playing the reconstructed signal" : "Playing the original recording";
  }

  setVolume(v) { this.volume = v; Object.values(this.tracks).forEach((a) => (a.volume = v)); }

  paint() {
    const t = this.time;
    this.$time.textContent = `${fmtTime(t)} / ${fmtTime(this.duration)}`;
    if (this.duration) this.$seek.value = Math.round((t / this.duration) * 1000);
    this.onTime && this.onTime(t);
  }

  dispose() {
    cancelAnimationFrame(this.raf);
    Object.values(this.tracks).forEach((a) => { a.pause(); a.removeAttribute("src"); a.load(); });
    this.setPlaying(false);
  }
}
