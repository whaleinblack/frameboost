// FrameBoost captions: redraws native <track> cues above the canvas, which would cover them.
(() => {
  "use strict";

  const HIDE_ATTRIBUTE = "data-frameboost-hide-native-captions";
  const LINE_HEIGHT = 1.3;
  const HIDE_RULES = `
video[${HIDE_ATTRIBUTE}]::-webkit-media-text-track-container { display: none !important; }
video[${HIDE_ATTRIBUTE}]::cue { color: transparent !important; background: transparent !important; text-shadow: none !important; }`;

  let hideSheet = null;

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  // Horizontal placement from the WebVTT cue settings, as percentages of the picture.
  function cueBox(cue) {
    const size = typeof cue.size === "number" ? cue.size : 100;
    const align = cue.align === "start" || cue.align === "left" ? "left" : cue.align === "end" || cue.align === "right" ? "right" : "center";
    let left = (100 - size) / 2;
    if (typeof cue.position === "number") {
      left = align === "left" ? cue.position : align === "right" ? cue.position - size : cue.position - size / 2;
    } else if (align === "left") {
      left = 0;
    } else if (align === "right") {
      left = 100 - size;
    }
    return { left: clamp(left, 0, 100 - size), size, align };
  }

  function cueElement(cue) {
    const { left, size, align } = cueBox(cue);
    const element = document.createElement("div");
    element.style.cssText = `margin-left:${left}%;width:${size}%;text-align:${align};white-space:pre-line`;
    const text = document.createElement("span");
    text.style.cssText = "background:rgba(0,0,0,.8);color:#fff;padding:0 .25em;-webkit-box-decoration-break:clone;box-decoration-break:clone";
    // getCueAsHTML returns inert nodes built by the browser's WebVTT parser, never page markup.
    if (cue.getCueAsHTML) text.append(cue.getCueAsHTML());
    else text.textContent = cue.text;
    element.append(text);
    return element;
  }

  function placeFixedLine(element, cue) {
    element.style.position = "absolute";
    element.style.left = "0";
    element.style.right = "0";
    if (!cue.snapToLines) {
      element.style.top = `${cue.line}%`;
      if (cue.lineAlign === "center") element.style.transform = "translateY(-50%)";
      if (cue.lineAlign === "end") element.style.transform = "translateY(-100%)";
    } else if (cue.line >= 0) {
      element.style.top = `${cue.line * LINE_HEIGHT}em`;
    } else {
      element.style.bottom = `${(-cue.line - 1) * LINE_HEIGHT}em`;
    }
  }

  class CaptionOverlay {
    constructor(video) {
      this.video = video;
      this.key = null;
      this.box = document.createElement("div");
      this.box.dataset.frameboostCaptions = "";
      this.box.setAttribute("aria-hidden", "true");
      this.box.style.cssText =
        `display:none;position:absolute;pointer-events:none;overflow:hidden;line-height:${LINE_HEIGHT};` +
        "font-family:'Segoe UI','PingFang SC','Microsoft YaHei','Noto Sans SC',system-ui,sans-serif";
      this.autoLines = document.createElement("div");
      this.autoLines.style.cssText = "position:absolute;left:0;right:0;bottom:0";
    }

    /**
     * Native cues the browser would be drawing, at the picture's time when it runs delayed.
     * Null means no captions are on.
     */
    activeCues(time) {
      const tracks = [...this.video.textTracks].filter(
        (track) => track.mode === "showing" && (track.kind === "subtitles" || track.kind === "captions"),
      );
      if (!tracks.length) return null;
      if (time === undefined) return tracks.flatMap((track) => [...(track.activeCues ?? [])]);
      return tracks.flatMap((track) => [...(track.cues ?? [])].filter((cue) => cue.startTime <= time && time < cue.endTime));
    }

    /** Call while the canvas covers the video. */
    show(anchor, time) {
      const cues = this.activeCues(time);
      if (!cues) {
        this.hide();
        return;
      }
      this.#hideNative();
      if (this.box.previousElementSibling !== anchor) anchor.after(this.box);
      this.box.style.display = "block";
      const key = cues.map((cue) => `${cue.startTime}|${cue.endTime}|${cue.line}|${cue.position}|${cue.text}`).join("\n");
      if (key !== this.key) this.#render(cues);
      this.key = key;
    }

    /** Call when the canvas is gone, so the browser's own captions take over again. */
    hide() {
      this.box.style.display = "none";
      this.video.removeAttribute(HIDE_ATTRIBUTE);
    }

    /**
     * Copies the canvas geometry. Like the browser's own captions, cues use the whole video box
     * including letterbox bars, so they stay put when the canvas hides and native ones return.
     */
    sync(canvas, height) {
      const from = canvas.style;
      for (const name of ["position", "left", "top", "width", "height", "transform", "transform-origin", "border-radius", "z-index"]) {
        this.box.style.setProperty(name, from.getPropertyValue(name));
      }
      this.box.style.fontSize = `${Math.max(14, height * 0.05)}px`;
    }

    destroy() {
      this.hide();
      this.box.remove();
    }

    #render(cues) {
      this.autoLines.replaceChildren();
      const fixed = [];
      for (const cue of cues) {
        const element = cueElement(cue);
        if (cue.line === "auto") {
          this.autoLines.append(element);
        } else {
          placeFixedLine(element, cue);
          fixed.push(element);
        }
      }
      this.box.replaceChildren(this.autoLines, ...fixed);
    }

    #hideNative() {
      // A constructed sheet, because a strict style-src blocks a <style> element added to the page.
      hideSheet ??= new CSSStyleSheet();
      if (!hideSheet.cssRules.length) hideSheet.replaceSync(HIDE_RULES);
      const root = this.video.getRootNode();
      if (!root.adoptedStyleSheets.includes(hideSheet)) root.adoptedStyleSheets.push(hideSheet);
      this.video.setAttribute(HIDE_ATTRIBUTE, "");
    }
  }

  globalThis.FrameBoostCaptions = { CaptionOverlay };
})();
