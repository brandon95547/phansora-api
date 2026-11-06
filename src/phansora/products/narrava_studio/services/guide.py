"""The guide in the editor: a question about Narrava Studio, answered from its how-to articles.

The articles are the studio's own (phansora: src/narrava-studio/guide/articles/), sent with each
question by the studio's server, so they change with the screens they describe. DeepSeek answers
only from them: in a few plain steps, with the studio's labels as the articles give them, and
saying so when they do not cover the question rather than guessing at a button that is not there.

It answers from where they are. The studio says which screen they are on, which tool is open and
what the project already has; asked "how do I begin a video?" in the editor, she used to answer
with the home page's first question, a step already taken (2026-11-06).

The articles go first in the system prompt, in the order they are sent, and the question after.
That keeps the long, unchanging part of every request the same from one question to the next,
which DeepSeek's prompt cache serves without reading it again.
"""
from __future__ import annotations

import logging
from typing import Any

from . import llm

logger = logging.getLogger("narrava-studio.guide")

ANSWER_CHARS = 3000
QUESTION_CHARS = 500
# Turns of the conversation so far shown with a question, so "and then?" means something.
HISTORY_TURNS = 4

# What she says when the articles do not cover a question, and what stands in for an empty reply.
NOT_COVERED = (
    "I don't have an answer for that one yet. Try asking about narrations, media, the timeline, "
    "captions, the storyboard or exporting."
)

SYSTEM = """
You are the guide in Narrava Studio, a video editor. You answer questions about how to use the
studio, from the how-to articles below and nothing else.

- Answer only from the articles. Never describe a button, menu, setting, shortcut or feature they
  do not mention, and never guess. If they do not cover the question, say so in one sentence and
  suggest what you can help with.
- Be brief and direct: a sentence or two, then numbered steps when there are steps. No preamble,
  no compliments, no "Great question".
- Use the studio's names for things exactly as the articles write them, in **bold**.
- Speak to the person as "you". Plain words, American spelling.
- If the question is not about using the studio, say you can only help with the studio.
- Answer from where they are ("Where they are", given with each question): start from the
  screen they are on and skip what they have already done. In the editor they are past the home
  page, so never send them back to choose the kind of video or its category to begin; tell them
  how to go on from the editor. Only point to the home page if they ask to start a different
  video.
""".strip()

REPLY = """
Return JSON only: {"answer": str, "articles": [int]}
"answer" is your reply. "articles" is the numbers of the articles you answered from, the most
useful first; [] when none covers the question.
""".strip()


def _clip(value: Any, limit: int) -> str:
    return str(value or "").strip()[:limit].strip()


def _system(articles: list) -> str:
    parts = [SYSTEM, "", "ARTICLES:"]
    for n, a in enumerate(articles, start=1):
        parts.append(f"\n[{n}] {a.title}\n{a.body.strip()}")
    parts += ["", REPLY]
    return "\n".join(parts)


def _user(req: Any) -> str:
    lines = []
    if req.video:
        lines.append(f"Their video: {_clip(req.video, 200)}")
    if req.where:
        lines.append(f"Where they are: {_clip(req.where, 600)}")
    history = req.history[-HISTORY_TURNS:]
    if history:
        lines.append("The conversation so far:")
        for turn in history:
            lines.append(f"Q: {_clip(turn.question, QUESTION_CHARS)}")
            lines.append(f"A: {_clip(turn.answer, ANSWER_CHARS)}")
        lines.append("")
    lines.append(f"Question: {_clip(req.question, QUESTION_CHARS)}")
    return "\n".join(lines)


def answer(req: Any) -> dict:
    """The answer, and the titles of the articles it came from. Always something to show: a
    reply the model got wrong becomes the not-covered sentence, never an error."""
    data = llm.deepseek_json(_system(req.articles), _user(req), max_output_tokens=1200)
    data = data if isinstance(data, dict) else {}
    text = _clip(data.get("answer"), ANSWER_CHARS) or NOT_COVERED
    used: list[str] = []
    for n in data.get("articles") or []:
        if isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= len(req.articles):
            title = req.articles[n - 1].title
            if title not in used:
                used.append(title)
    return {"answer": text, "articles": used[:3]}
