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


# How each Animation Style choice is described to the model. The keys are the values the
# Studio's style <select> submits (components/MediaPanel.js, STYLE_OPTS) — they must match
# exactly, because an unknown style is NOT an error here: it falls back to the default
# rather than failing a render over a creative preference, so a key that only exists on one
# side looks like the style "not working" rather than not existing.
#
# Every entry is TREATMENT — how a thing is drawn and how it moves — because the subject is
# already carried by the user's prompt. Written as subject matter ("compasses, pillars") a
# style reads as content to insert, and an animation about quarterly revenue comes back
# wearing regalia; palette, line quality, construction rule and motion signature keep each
# one steering the drawing instead.
_STYLE_DEFAULT = "flat-vector"
_STYLES = {
    # ── Clean & graphic ──────────────────────────────────────────────────────────
    "flat-vector": (
        "Flat vector: solid flat fills with no gradients, shadows or texture of any kind. "
        "Clean geometric shapes with crisp edges, a restrained palette of 3-5 colors plus "
        "one accent, generous empty space. Shapes read as cut paper laid flat. Precise "
        "easing, nothing decorative that does not serve the idea."
    ),
    "minimal-line": (
        "Minimal line: a single consistent stroke weight, unfilled outlines, mostly empty "
        "canvas. One or two ink colors on a plain light ground. Forms described by contour "
        "alone — no fills, no shading. Movement is drawing and undrawing: strokes trace on "
        "along their own path, and out again. Restraint is the point."
    ),
    "motion-graphics": (
        "Motion graphics: bold kinetic shapes, snappy overshoot easing, dynamic "
        "composition, high contrast. Confident, energetic movement."
    ),
    "infographic": (
        "Infographic: information design. Labelled elements, leader lines and callouts, a "
        "legend where it earns its place, step-by-step reveals that build one idea at a "
        "time. Clear typographic hierarchy, generous margins, a readable palette where "
        "color CARRIES meaning rather than decorating. Numbers and units where they help."
    ),
    "kinetic-typography": (
        "Kinetic typography: the words ARE the animation. Text is the primary subject — "
        "set large, tracked tight, weighted for contrast — entering, scaling, masking and "
        "reflowing in time with its own emphasis. Strong grid, two type sizes at most per "
        "moment, a flat single-color ground. Shapes appear only to punctuate the text."
    ),
    # ── Illustrative ─────────────────────────────────────────────────────────────
    "hand-drawn": (
        "Hand-drawn: organic ink strokes with visible wobble and varying pressure, ends "
        "that overshoot slightly, shapes that do not quite close. Sketchbook palette — ink "
        "plus one or two washes — on off-white paper. Small deliberate imperfections and "
        "loose hatching for tone. Motion has a light hand-animated jitter."
    ),
    "whiteboard": (
        "Whiteboard: a hand-drawn feel — sketch-like strokes progressively drawn onto "
        "a white or off-white board, imperfect lines, handwriting-adjacent labels."
    ),
    "paper-cutout": (
        "Paper cutout: layered shapes with slightly rough torn or scissored edges, each "
        "sitting on its own plane with a soft offset drop shadow. Matte construction-paper "
        "palette. Depth comes from stacking and shadow, never from gradients. Pieces slide, "
        "hinge and swap in and out as if moved by hand."
    ),
    "comic-book": (
        "Comic book: heavy black ink outlines of varying weight, flat saturated fills, "
        "halftone dot shading, speed lines and impact bursts. Bold panel geometry with "
        "gutters, and hand-lettered caption boxes. Motion is punchy and posed — hold, snap "
        "to the next beat, hold — rather than smoothly interpolated."
    ),
    "engraved": (
        "Engraved: fine metallic line-work — gold, brass or sepia — on deep indigo, "
        "black or parchment, drawn as if etched into a plate. Visible construction "
        "geometry, compass arcs and ruled lines, strong symmetry, restrained ornament. "
        "Hatching and line density for tone rather than flat fills."
    ),
    # ── Technical ────────────────────────────────────────────────────────────────
    "blueprint": (
        "Blueprint: white and pale cyan line-work on deep blueprint blue over a fine "
        "grid. Dimension arrows, leader lines, section marks and technical annotation in "
        "light monospace. Orthographic construction, consistent stroke weights, every "
        "element drawn as if measured."
    ),
    "schematic": (
        "Schematic: a diagram of parts and connections. Nodes as simple boxed or circular "
        "symbols, joined by orthogonal connectors that route in right angles with junction "
        "dots and arrowheads. Monospace labels, a legend-like restraint, one accent color "
        "for the active path. Signal travels ALONG the connectors as the animation's motion."
    ),
    "isometric": (
        "Isometric: 2.5D forms on fixed 30-degree axes with no perspective convergence. "
        "Flat faces in three tonal steps for top, left and right; muted palette with a "
        "single accent. Objects assembled from stacked volumes, movement travelling "
        "along the isometric axes."
    ),
    "wireframe": (
        "Wireframe: geometry described by edges only — unfilled polygon meshes with visible "
        "vertices, hidden lines drawn faintly rather than removed. Thin uniform strokes, "
        "one or two colors on a dark or paper ground, subdivision lines showing how forms "
        "are built. Slow rotation and deformation reveal the structure."
    ),
    "data-visualization": (
        "Data visualization: plotted marks in a real coordinate space — axes with ticks and "
        "units, gridlines, a baseline. Bars, lines, areas, points or arcs that animate by "
        "GROWING along their encoding (bars from the baseline, lines drawing left to right) "
        "rather than sliding in. Sequential or categorical palette, direct labels over a "
        "legend, honest scales."
    ),
    # ── Digital & stylized ───────────────────────────────────────────────────────
    "neon-glow": (
        "Neon glow: glowing strokes with soft bloom against near-black, horizon grids "
        "receding to a vanishing point, magenta-to-cyan gradients. Light sources rather "
        "than filled shapes, additive glow where strokes overlap, slow confident drift "
        "with the occasional flicker."
    ),
    "pixel-art": (
        "Pixel art: everything drawn on a coarse pixel grid — pick a cell size (6-12 canvas "
        "pixels) and snap every shape, edge and position to it. Disable image smoothing, "
        "no anti-aliased curves, no sub-pixel motion: things move a whole cell at a time. "
        "Tight limited palette (8-16 colors), dithering instead of gradients, chunky "
        "readable forms. Animation steps at a low frame rate rather than easing smoothly."
    ),
    "retro-terminal": (
        "Retro terminal: monospace phosphor text on black — green or amber — with a faint "
        "scanline overlay and soft glow bleeding off the glyphs. Content types on character "
        "by character with a blinking block cursor; box-drawing rules, prompts and ASCII "
        "framing. Occasional flicker or roll. Everything aligns to a character grid."
    ),
    "low-poly": (
        "Low poly: forms faceted into visible triangles, each facet filled flat with its "
        "own tone so shading steps rather than blends. Crystalline silhouettes, a cool or "
        "sunset gradient palette sampled per facet, occasional visible edges. Motion is "
        "slow rotation and facet-by-facet assembly."
    ),
    "particle": (
        "Particle: form emerges from many small points — hundreds or thousands — rather "
        "than from drawn shapes. Points flow along fields, gather into a silhouette, hold, "
        "then disperse. Additive blending, fading trails, a dark ground with one or two "
        "luminous hues. Seed the point set deterministically at load time so the motion is "
        "reproducible."
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
    style_key = (style or _STYLE_DEFAULT).strip().lower()
    style_desc = _STYLES.get(style_key, _STYLES[_STYLE_DEFAULT])
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
