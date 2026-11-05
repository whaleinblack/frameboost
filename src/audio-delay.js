// FrameBoost audio delay: holds the page's sound back so it stays with a delayed picture.
// An element can only ever have one audio source node, so the route stays for the element's life
// and is set to zero when not needed. A context blocked by autoplay policy would silence the video,
// so nothing is rerouted until the context is really running.
(() => {
  "use strict";

  const MAX_DELAY = 2;
  const routes = new WeakMap();

  class AudioDelay {
    constructor(video) {
      this.video = video;
      this.failed = false;
      this.target = 0;
    }

    /** True once the element's sound runs through the delay line. Safe to call every frame. */
    ensure() {
      if (this.failed) return false;
      if (this.delayNode && this.context.state === "running") return true;
      try {
        this.context ??= new AudioContext({ latencyHint: "interactive" });
        if (this.context.state !== "running") {
          if (this.context.state === "suspended") this.context.resume().catch(() => {});
          return false;
        }
        if (!this.delayNode) {
          const source = this.context.createMediaElementSource(this.video);
          this.delayNode = this.context.createDelay(MAX_DELAY);
          this.delayNode.delayTime.value = this.target;
          source.connect(this.delayNode).connect(this.context.destination);
        }
        return true;
      } catch {
        // The page routed this element through its own WebAudio graph first.
        this.failed = true;
        return false;
      }
    }

    set(seconds) {
      if (seconds === this.target) return;
      this.target = seconds;
      this.delayNode?.delayTime.setTargetAtTime(seconds, this.context.currentTime, 0.02);
    }
  }

  globalThis.FrameBoostAudio = {
    forVideo(video) {
      if (!routes.has(video)) routes.set(video, new AudioDelay(video));
      return routes.get(video);
    },
  };
})();
