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

# SDKs are imported lazily inside each provider branch. At module scope an absent
# `anthropic` package would raise on import, and main.py mounts a product only if it
# imports cleanly — so one missing dependency would silently un-mount the whole of
# Narrava Studio and every /studio route would 404 with nothing explaining why.

# Provider switch, mirroring NARRAVA_LLM_PROVIDER on the script/storyboard path.
_DEFAULT_PROVIDER = "deepseek"

# Only the Anthropic branch carries a literal: it is the one provider whose model is
# not already named in .env. The DeepSeek branch resolves through shared.ai.models,
# which deliberately has no built-in default — see _missing_message() there.
_ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-6"


def _provider() -> str:
    return (os.environ.get("NARRAVA_ANIMATION_PROVIDER") or _DEFAULT_PROVIDER).strip().lower()


def model_name() -> str:
    if _provider() == "anthropic":
        return os.environ.get("NARRAVA_ANIMATION_MODEL") or _ANTHROPIC_DEFAULT_MODEL
    from phansora.shared.ai.models import resolve_model

    # NARRAVA_ANIMATION_MODEL > DEEPSEEK_MODEL. Raises MissingModelConfig when neither
    # is set, which is the house convention: a hardcoded name breaks silently when the
    # provider retires it.
    return resolve_model("NARRAVA_ANIMATION_MODEL", provider="deepseek")


def provider_configured() -> bool:
    if _provider() == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    return bool(os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_CHAT_API_KEY"))


def required_key_name() -> str:
    return "ANTHROPIC_API_KEY" if _provider() == "anthropic" else "DEEPSEEK_API_KEY"


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


def _anthropic_html(user_prompt: str) -> str:
    import anthropic  # lazy — see the note at the top of this module

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
        messages=[{"role": "user", "content": user_prompt}],
    ) as stream:
        response = stream.get_final_message()
    if response.stop_reason == "refusal":
        raise ValueError("The model declined to generate this animation.")
    if response.stop_reason == "max_tokens":
        raise ValueError("The generated document was cut off before completion.")
    return "".join(block.text for block in response.content if block.type == "text")


def _deepseek_html(user_prompt: str) -> str:
    import httpx

    from phansora.shared.ai.deepseek import DeepSeekChatConfig

    cfg = DeepSeekChatConfig.from_env(product_var="NARRAVA_ANIMATION_MODEL")
    resp = httpx.post(
        f"{cfg.base_url}/v1/chat/completions",
        json={
            "model": cfg.model,
            # Low but non-zero: this is code generation, where a deterministic-ish
            # sample beats a creative one, but 0 makes a failed retry reproduce the
            # same broken document the feedback round is meant to fix.
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            # Well above the ~4k tokens a finished document runs to, and inside the
            # per-request output ceiling the chat API enforces.
            "max_tokens": 8192,
            "stream": False,
        },
        headers={"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"},
        timeout=cfg.timeout_s,
    )
    resp.raise_for_status()
    choices = resp.json().get("choices") or []
    if not choices:
        raise ValueError("The model returned no animation.")
    # Same distinction the Anthropic branch draws: a truncated document is a retryable
    # generation problem, not a malformed one, and says so rather than failing later
    # on a missing </html> the validator would blame on the prompt.
    if (choices[0].get("finish_reason") or "") == "length":
        raise ValueError("The generated document was cut off before completion.")
    return ((choices[0].get("message") or {}).get("content") or "").strip()


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
    """One LLM call -> the complete animation HTML document.

    Raises ValueError when the response doesn't look like a usable document, so the
    caller can retry once with the error as feedback.
    """
    user_prompt = _build_user_prompt(
        prompt,
        duration_sec=duration_sec,
        width=width,
        height=height,
        transparent=transparent,
        style=style,
        feedback=feedback,
    )
    raw = (
        _anthropic_html(user_prompt)
        if _provider() == "anthropic"
        else _deepseek_html(user_prompt)
    )
    html = _strip_fences(raw)
    if "<canvas" not in html or 'id="stage"' not in html.replace("'", '"'):
        raise ValueError('The generated document has no <canvas id="stage">.')
    if "renderFrame" not in html:
        raise ValueError("The generated document does not define window.renderFrame.")
    if re.search(r'\bsrc\s*=\s*["\']https?://', html) or "@import" in html:
        raise ValueError("The generated document references external resources.")
    return html
