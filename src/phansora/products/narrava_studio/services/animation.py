"""AI media generation: a described visual -> a self-contained HTML5 Canvas animation.

One LLM call (DeepSeek by default, Anthropic behind a switch) writes a complete HTML
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

import logging
import os
import re

logger = logging.getLogger("narrava-studio")

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


def provider_name() -> str:
    """The active provider, for logs and /health."""
    return _provider()


# Output ceiling for the DeepSeek branch. 8192 was measured too small — real documents
# hit it and came back truncated. Env-tunable because the true per-model ceiling is the
# provider's to change, and a request above it is rejected outright: if this starts
# failing with a 400 from DeepSeek naming max_tokens, lower it rather than editing code.
_DEFAULT_MAX_TOKENS = 16384


def _max_tokens() -> int:
    raw = (os.environ.get("NARRAVA_ANIMATION_MAX_TOKENS") or "").strip()
    try:
        return max(1024, int(raw)) if raw else _DEFAULT_MAX_TOKENS
    except ValueError:
        return _DEFAULT_MAX_TOKENS


# Thinking is OFF for the DeepSeek branch too, and for a harder reason than the
# Anthropic branch's latency: with it on, this call did not merely run slowly, it
# never produced anything at all. Every attempt in production came back
#
#   Animation truncated: max_tokens=16384, completion=16384, reasoning=16384, html_chars=0
#
# — the whole budget spent reasoning, zero characters of document, nine times out
# of nine. Whole-program code generation is exactly the workload this model
# thinks longest about, and DeepSeek counts reasoning against max_tokens, so the
# budget was gone before the first tag. Raising the ceiling only buys a more
# expensive failure; the retry's "write something shorter" feedback cannot help a
# document that was never started.
#
# With reasoning off the same prompt returns a complete, contract-passing
# document in ~24s using ~3k output tokens. Env-tunable rather than hardcoded
# because a future model may need some thinking here — set it to `default` to
# hand the decision back to the provider.
_DEFAULT_REASONING_EFFORT = "none"


def _reasoning_effort() -> str | None:
    """The `reasoning_effort` to send, or None to omit the field entirely."""
    raw = (os.environ.get("NARRAVA_ANIMATION_REASONING_EFFORT") or _DEFAULT_REASONING_EFFORT).strip().lower()
    return None if raw in ("", "default", "auto", "provider") else raw


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
    # Everything below is TREATMENT — how a thing is drawn and moves — because the
    # subject is already carried by the user's prompt. An earlier draft had seventeen
    # entries; most collapsed into each other once that split was clear (alchemical,
    # celestial and art-deco are "engraved" with different iconography, and iconography
    # is the prompt's job). Fewer, more distinct entries also means fewer untuned
    # fragments to notice going wrong.
    "blueprint": (
        "Blueprint: white and pale cyan line-work on deep blueprint blue over a fine "
        "grid. Dimension arrows, leader lines, section marks and technical annotation in "
        "light monospace. Orthographic construction, consistent stroke weights, every "
        "element drawn as if measured."
    ),
    "isometric": (
        "Isometric: 2.5D forms on fixed 30-degree axes with no perspective convergence. "
        "Flat faces in three tonal steps for top, left and right; muted palette with a "
        "single accent. Objects assembled from stacked volumes, movement travelling "
        "along the isometric axes."
    ),
    "neon": (
        "Neon: glowing strokes with soft bloom against near-black, horizon grids "
        "receding to a vanishing point, magenta-to-cyan gradients. Light sources rather "
        "than filled shapes, additive glow where strokes overlap, slow confident drift "
        "with the occasional flicker."
    ),
    # The general ceremonial treatment. Prompt it with the iconography you want —
    # "a constellation chart", "a rose window", "stepped deco fans" — and it covers
    # the ground four separate styles used to.
    "engraved": (
        "Engraved: fine metallic line-work — gold, brass or sepia — on deep indigo, "
        "black or parchment, drawn as if etched into a plate. Visible construction "
        "geometry, compass arcs and ruled lines, strong symmetry, restrained ornament. "
        "Hatching and line density for tone rather than flat fills."
    ),
    # Kept as its own entry rather than folded into 'engraved': it is the specific
    # vocabulary this tool gets asked for by name, and a preset is also documentation
    # that the capability exists.
    #
    # Phrased as a visual LANGUAGE, not a set of objects. A style written as subject
    # matter ("compasses, all-seeing eyes") reads as content to insert, so an animation
    # about quarterly revenue comes back wearing regalia. Palette, line quality and
    # construction rule keep it steering the drawing instead.
    "masonic": (
        "Masonic: engraved geometric symbolism. Construction-line geometry — compass "
        "arcs, set-square angles, radiating light, twin pillars, checkerboard grids, "
        "allegorical emblems — drawn as fine gold or sepia line-work on deep indigo or "
        "parchment, as if etched. Strong symmetry, visible construction geometry, "
        "restrained ornament. Precision over illustration."
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
        raise ValueError(
            "The generated document was cut off before completion. Write a shorter, "
            "more compact document: fewer helper functions, no long comments, and no "
            "repeated drawing code that a loop could express."
        )
    return "".join(block.text for block in response.content if block.type == "text")


def _deepseek_html(user_prompt: str) -> str:
    import httpx

    from phansora.shared.ai.deepseek import DeepSeekChatConfig

    cfg = DeepSeekChatConfig.from_env(product_var="NARRAVA_ANIMATION_MODEL")
    body = {
        "model": cfg.model,
        # Low but non-zero: this is code generation, where a deterministic-ish
        # sample beats a creative one, but 0 makes a failed retry reproduce the
        # same broken document the feedback round is meant to fix.
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": _max_tokens(),
        "stream": False,
    }
    # Omitted rather than sent empty when disabled: DeepSeek ignores parameters it
    # does not recognise instead of rejecting them, so a wrong key would look like
    # it worked. See _reasoning_effort for why the default is off.
    effort = _reasoning_effort()
    if effort:
        body["reasoning_effort"] = effort
    resp = httpx.post(
        f"{cfg.base_url}/v1/chat/completions",
        json=body,
        headers={"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"},
        timeout=cfg.timeout_s,
    )
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("The model returned no animation.")
    message = choices[0].get("message") or {}
    text = (message.get("content") or "").strip()
    # Same distinction the Anthropic branch draws: a truncated document is a retryable
    # generation problem, not a malformed one, and says so rather than failing later
    # on a missing </html> the validator would blame on the prompt.
    if (choices[0].get("finish_reason") or "") == "length":
        _log_truncation(data, message, text)
        raise ValueError(
            "The generated document was cut off before completion. Write a shorter, "
            "more compact document: fewer helper functions, no long comments, and no "
            "repeated drawing code that a loop could express."
        )
    return text


def _log_truncation(data: dict, message: dict, text: str) -> None:
    """Say WHICH budget ran out, because the three causes need opposite fixes.

    `asked` vs `completion` separates them at a glance: equal means the document really
    is that long (write less / raise the cap), completion stuck below asked means the
    provider clamped to its own ceiling (raising ours does nothing), and a large
    `reasoning` share means thinking consumed the budget before any HTML was written.
    """
    usage = data.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    reasoning = details.get("reasoning_tokens")
    # Some providers put thinking in a sibling field rather than counting it separately.
    if reasoning is None and message.get("reasoning_content"):
        reasoning = f"~{len(str(message['reasoning_content']))} chars (uncounted)"
    logger.warning(
        "Animation truncated: asked max_tokens=%s, completion=%s, prompt=%s, "
        "reasoning=%s, html_chars=%s",
        _max_tokens(), usage.get("completion_tokens"), usage.get("prompt_tokens"),
        reasoning if reasoning is not None else "n/a", len(text),
    )


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
