"""AI media generation: a described visual -> a self-contained HTML5 Canvas animation.

One LLM call (Claude Sonnet 4.6 via the Anthropic SDK) writes a complete HTML
document that draws the requested animation on a single <canvas>. The document is
never shown to the user and never stored: the Node side loads it in a headless
browser, drives ``window.renderFrame(t)`` frame by frame, encodes the frames to
video with ffmpeg, and imports the video into the project's media library. From
the user's perspective they described a visual and a video appeared.

The contract with the renderer (enforced by the system prompt and validated
loosely here) is:

  - one <canvas id="stage" width=W height=H>, everything inline, no external assets
  - window.renderFrame(t) draws the complete, deterministic frame for time t
  - the live-preview rAF loop is gated behind !window.__PHANSORA_CAPTURE__ so the
    capture harness can drive time itself
"""
from __future__ import annotations

import os
import re

import anthropic

# The user picked this model for media generation explicitly. Env-overridable so a
# newer model can be tried without a deploy.
_DEFAULT_MODEL = "claude-sonnet-4-6"


def model_name() -> str:
    return os.environ.get("NARRAVA_ANIMATION_MODEL", _DEFAULT_MODEL) or _DEFAULT_MODEL


def provider_configured() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def required_key_name() -> str:
    return "ANTHROPIC_API_KEY"


# How each Animation Style choice is described to the model. Unknown/missing styles
# fall back to 'minimal' rather than erroring over a creative preference.
_STYLES = {
    "minimal": (
        "Minimal: clean flat shapes, a restrained palette of 2-4 colors, generous "
        "empty space, precise easing. Nothing decorative that does not serve the idea."
    ),
    "educational": (
        "Educational: clear labels, step-by-step reveals, diagrammatic clarity. "
        "Friendly readable palette, arrows and callouts where they aid understanding."
    ),
    "motion-graphics": (
        "Motion graphics: bold kinetic shapes, snappy overshoot easing, dynamic "
        "composition, high contrast. Confident, energetic movement."
    ),
    "whiteboard": (
        "Whiteboard: a hand-drawn feel — sketch-like strokes progressively drawn onto "
        "a white or off-white board, imperfect lines, handwriting-adjacent labels."
    ),
    "playful": (
        "Playful: bouncy squash-and-stretch easing, rounded shapes, bright cheerful "
        "palette, small delightful details."
    ),
}


_SYSTEM = """You are an expert motion designer who writes self-contained HTML5 Canvas \
animations. You output ONE complete HTML document and nothing else - no markdown fences, \
no commentary before or after.

Hard requirements:
1. SELF-CONTAINED. All CSS and JavaScript inline in the document. NO external resources \
of any kind: no CDN scripts, no web fonts, no images, no fetch/XHR/WebSocket. Draw \
everything with the Canvas 2D API. Use only generic font stacks (sans-serif, serif, \
monospace).
2. ONE CANVAS. Exactly one <canvas id="stage"> with the width and height attributes set \
to the requested pixel size. body has margin 0 and the canvas is the only visible element.
3. DETERMINISTIC FRAME FUNCTION. Define window.renderFrame(t) where t is seconds from 0 \
to the requested duration. It draws the COMPLETE frame for time t and must be pure: \
calling renderFrame(3.2) twice, or after renderFrame(7.0), draws the identical image. \
Derive ALL motion from t. Never read Date.now, performance.now, or Math.random inside \
renderFrame - if you need random-looking values, precompute them once at load time with \
a seeded pseudo-random function so every page load is identical.
4. REPAINT FULLY. Start every renderFrame call by clearing the canvas (clearRect), then \
paint the frame's background (see the background rule in the request), then the content.
5. GATED PREVIEW LOOP. After defining renderFrame, add a live preview exactly like:
   if (!window.__PHANSORA_CAPTURE__) {
     const start = performance.now();
     (function loop(now) {
       window.renderFrame(((now - start) / 1000) % DURATION);
       requestAnimationFrame(loop);
     })(start);
   }
   The capture harness sets __PHANSORA_CAPTURE__ and drives renderFrame itself.
6. A COMPLETE PIECE. The animation lasts exactly the requested duration and reads as \
finished: ease in, develop, resolve. In the final ~10% settle into a stable end state \
(hold, or land cleanly so a loop back to t=0 feels intentional).
7. ROBUST. No exceptions, no infinite loops, guard divisions, no console spam. Text must \
be comfortably legible at the requested pixel size.

Craft expectations: real easing (never linear position lerps for hero elements), \
composition with a clear focal point, readable pacing (nothing important flashes by in \
under half a second), and colors chosen deliberately for the requested style."""


def _build_user_prompt(
    prompt: str,
    *,
    duration_sec: int,
    width: int,
    height: int,
    transparent: bool,
    style: str | None,
    feedback: str | None,
) -> str:
    style_key = (style or "minimal").strip().lower()
    style_desc = _STYLES.get(style_key, _STYLES["minimal"])
    background_rule = (
        "Background: TRANSPARENT. Do not paint any background fill - after clearRect the "
        "canvas stays transparent, so the animation can be composited over other footage. "
        "Make sure the content itself is opaque enough to read over arbitrary backgrounds."
        if transparent
        else "Background: SOLID. Paint an opaque background over the whole canvas every "
        "frame; choose a color that fits the request and style unless one is specified."
    )
    parts = [
        f"Create this animation: {prompt.strip()}",
        "",
        f"Canvas size: {width}x{height} pixels.",
        f"Duration: exactly {duration_sec} seconds (use {duration_sec} as DURATION in the preview loop).",
        background_rule,
        f"Animation style - {style_desc}",
    ]
    if feedback:
        parts += [
            "",
            "A previous attempt failed validation with this error, fix it in the new "
            f"document: {feedback.strip()}",
        ]
    return "\n".join(parts)


def _strip_fences(text: str) -> str:
    """Tolerate a model that wraps the document in ```html fences despite instructions."""
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*\n", "", s)
        if s.rstrip().endswith("```"):
            s = s.rstrip()[: -3].rstrip()
    # Anything before the doctype/html tag is commentary.
    m = re.search(r"<!(?:doctype|DOCTYPE)|<html", s)
    if m and m.start() > 0:
        s = s[m.start():]
    return s.strip()


def generate_animation_html(
    prompt: str,
    *,
    duration_sec: int,
    width: int,
    height: int,
    transparent: bool = False,
    style: str | None = None,
    feedback: str | None = None,
) -> str:
    """One Claude Sonnet 4.6 call -> the complete animation HTML document.

    Raises ValueError when the response doesn't look like a usable document, so the
    caller can retry once with the error as feedback.
    """
    client = anthropic.Anthropic()
    # Thinking is OFF, on measurement: adaptive thinking spent 8+ minutes at the
    # default effort and ~3 minutes at medium designing an animation — far past
    # what a user waiting in the dialog (or the Node proxy's timeout) tolerates.
    # This is single-shot code generation with a validator and one feedback retry
    # behind it, so the thinking latency buys little. Streamed so the connection
    # stays comfortably clear of HTTP timeouts while ~10KB of HTML is written.
    with client.messages.stream(
        model=model_name(),
        max_tokens=24000,
        thinking={"type": "disabled"},
        output_config={"effort": "medium"},
        system=_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": _build_user_prompt(
                    prompt,
                    duration_sec=duration_sec,
                    width=width,
                    height=height,
                    transparent=transparent,
                    style=style,
                    feedback=feedback,
                ),
            }
        ],
    ) as stream:
        response = stream.get_final_message()
    if response.stop_reason == "refusal":
        raise ValueError("The model declined to generate this animation.")
    if response.stop_reason == "max_tokens":
        raise ValueError("The generated document was cut off before completion.")

    html = _strip_fences(
        "".join(block.text for block in response.content if block.type == "text")
    )
    if "<canvas" not in html or 'id="stage"' not in html.replace("'", '"'):
        raise ValueError('The generated document has no <canvas id="stage">.')
    if "renderFrame" not in html:
        raise ValueError("The generated document does not define window.renderFrame.")
    if re.search(r'\bsrc\s*=\s*["\']https?://', html) or "@import" in html:
        raise ValueError("The generated document references external resources.")
    return html
