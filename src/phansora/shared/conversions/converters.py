"""Timeline → many-format converters.

Product-agnostic: every converter takes one canonical ``document`` dict and
returns a UTF-8 string. A document looks like::

    {
      "title": "The Great Flood",
      "context": "deluge myth",          # optional
      "nodes": [
        {"id": "origin", "parent": None, "kind": "origin", "type": None,
         "year": -1600, "era": None, "precision": "century",
         "title": "Eridu Genesis", "claim": "…", "ai_original": None,
         "confidence": 0.8,
         "citations": [{"title": "ETCSL", "url": "https://…"}]},
        {"id": "0", "parent": None, "kind": "event", ...},   # primary spine
        {"id": "0.1", "parent": "0", "kind": "branch", "type": "person", ...}
      ]
    }

Primary-path nodes have ``parent == None`` and are given in chronological order;
branch nodes carry a ``parent`` id. Edges are derived here so callers stay simple.
"""
from __future__ import annotations

import csv
import io
import json
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Tuple


# --------------------------------------------------------------------- helpers
def _nodes(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    ns = doc.get("nodes")
    return [n for n in ns if isinstance(n, dict)] if isinstance(ns, list) else []


def _spine(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Primary-path nodes (no parent), in the order supplied."""
    return [n for n in nodes if not n.get("parent")]


def _edges(nodes: List[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
    """(source_id, target_id, kind) — 'primary' along the spine, 'branch' for
    parent→child links."""
    edges: List[Tuple[str, str, str]] = []
    spine = _spine(nodes)
    for a, b in zip(spine, spine[1:]):
        edges.append((str(a.get("id")), str(b.get("id")), "primary"))
    ids = {str(n.get("id")) for n in nodes}
    for n in nodes:
        p = n.get("parent")
        if p and str(p) in ids:
            edges.append((str(p), str(n.get("id")), "branch"))
    return edges


def _children(nodes: List[Dict[str, Any]], parent_id: str) -> List[Dict[str, Any]]:
    return [n for n in nodes if str(n.get("parent") or "") == parent_id]


def _year_label(node: Dict[str, Any]) -> str:
    y = node.get("year")
    if isinstance(y, (int, float)) and not isinstance(y, bool):
        y = int(y)
        tag = "BCE" if y < 0 else "CE"
        prec = node.get("precision")
        suffix = f" ({prec})" if prec and prec not in ("exact", "year", "unknown") else ""
        return f"c. {abs(y)} {tag}{suffix}"
    era = node.get("era")
    return str(era) if era else "Unknown era"


def _claim(node: Dict[str, Any]) -> str:
    return str(node.get("claim") or node.get("summary") or "")


def _title(node: Dict[str, Any]) -> str:
    return str(node.get("title") or node.get("source_title") or "Untitled")


def _citations(node: Dict[str, Any]) -> List[Dict[str, str]]:
    cs = node.get("citations")
    out = []
    if isinstance(cs, list):
        for c in cs:
            if isinstance(c, dict) and c.get("url"):
                out.append({"title": str(c.get("title") or c["url"]), "url": str(c["url"])})
    return out


def _doc_title(doc: Dict[str, Any]) -> str:
    return str(doc.get("title") or "Timeline")


# ----------------------------------------------------------------- text formats
def to_json(doc: Dict[str, Any]) -> str:
    return json.dumps(doc, indent=2, ensure_ascii=False)


def to_csv(doc: Dict[str, Any]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "kind", "type", "parent", "year", "era", "when", "title", "claim", "confidence", "citations"])
    for n in _nodes(doc):
        cites = " ; ".join(c["url"] for c in _citations(n))
        w.writerow([
            n.get("id"), n.get("kind") or "", n.get("type") or "", n.get("parent") or "",
            n.get("year") if isinstance(n.get("year"), (int, float)) and not isinstance(n.get("year"), bool) else "",
            n.get("era") or "", _year_label(n), _title(n), _claim(n),
            n.get("confidence") if isinstance(n.get("confidence"), (int, float)) else "", cites,
        ])
    return buf.getvalue()


def to_markdown(doc: Dict[str, Any]) -> str:
    nodes = _nodes(doc)
    lines = [f"# {_doc_title(doc)}", ""]
    if doc.get("context"):
        lines += [f"*{doc['context']}*", ""]
    lines += ["## Timeline", ""]

    def render(node: Dict[str, Any], depth: int) -> None:
        indent = "  " * depth
        lines.append(f"{indent}- **{_year_label(node)}** — {_title(node)}")
        claim = _claim(node)
        if claim:
            lines.append(f"{indent}  {claim}")
        for c in _citations(node):
            lines.append(f"{indent}  - [{c['title']}]({c['url']})")
        for child in _children(nodes, str(node.get("id"))):
            render(child, depth + 1)

    for n in _spine(nodes):
        render(n, 0)
    return "\n".join(lines) + "\n"


def to_html(doc: Dict[str, Any]) -> str:
    nodes = _nodes(doc)
    esc = lambda s: (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

    def render(node: Dict[str, Any]) -> str:
        cites = "".join(
            f'<a href="{esc(c["url"])}" target="_blank" rel="noopener">{esc(c["title"])}</a>'
            for c in _citations(node)
        )
        kids = "".join(render(c) for c in _children(nodes, str(node.get("id"))))
        return (
            f'<li><span class="when">{esc(_year_label(node))}</span>'
            f'<span class="title">{esc(_title(node))}</span>'
            f'<p class="claim">{esc(_claim(node))}</p>'
            f'<div class="cites">{cites}</div>'
            f'{f"<ul>{kids}</ul>" if kids else ""}</li>'
        )

    body = "".join(render(n) for n in _spine(nodes))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(_doc_title(doc))}</title>
<style>
body{{font-family:system-ui,sans-serif;background:#0c0a17;color:#e8e6f0;max-width:820px;margin:0 auto;padding:32px 20px;line-height:1.55}}
h1{{font-size:1.7rem}} ul{{list-style:none;padding-left:18px;border-left:1px solid #2a2540}}
li{{margin:0 0 18px}} .when{{display:block;font:600 .72rem/1 ui-monospace,monospace;letter-spacing:.08em;color:#ffb84d}}
.title{{display:block;font-weight:600;margin-top:2px}} .claim{{margin:4px 0;color:#a29fbd}}
.cites a{{display:inline-block;margin:0 8px 4px 0;font-size:.78rem;color:#8b88a8}}
</style></head>
<body><h1>{esc(_doc_title(doc))}</h1><ul>{body}</ul></body></html>
"""


def to_dot(doc: Dict[str, Any]) -> str:
    nodes = _nodes(doc)
    q = lambda s: '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'
    lines = ["digraph Timeline {", '  rankdir=LR;', '  node [shape=box, style=rounded, fontname="Helvetica"];']
    for n in nodes:
        label = f"{_title(n)}\\n{_year_label(n)}"
        lines.append(f"  {q(n.get('id'))} [label={q(label)}];")
    for s, t, kind in _edges(nodes):
        style = "" if kind == "primary" else " [style=dashed]"
        lines.append(f"  {q(s)} -> {q(t)}{style};")
    lines.append("}")
    return "\n".join(lines) + "\n"


def to_mermaid(doc: Dict[str, Any]) -> str:
    nodes = _nodes(doc)
    safe = lambda s: "n_" + "".join(ch if ch.isalnum() else "_" for ch in str(s))
    txt = lambda s: str(s).replace('"', "'").replace("\n", " ")
    lines = ["graph LR"]
    for n in nodes:
        lines.append(f'  {safe(n.get("id"))}["{txt(_title(n))}<br/>{txt(_year_label(n))}"]')
    for s, t, kind in _edges(nodes):
        arrow = "-->" if kind == "primary" else "-.->"
        lines.append(f"  {safe(s)} {arrow} {safe(t)}")
    return "\n".join(lines) + "\n"


def to_timelinejs(doc: Dict[str, Any]) -> str:
    events = []
    for n in _nodes(doc):
        ev: Dict[str, Any] = {"text": {"headline": _title(n), "text": _claim(n)}}
        y = n.get("year")
        if isinstance(y, (int, float)) and not isinstance(y, bool):
            ev["start_date"] = {"year": int(y)}
        else:
            ev["text"]["headline"] = f"{_title(n)} — {_year_label(n)}"
        if n.get("kind") == "origin":
            ev["group"] = "Origin"
        elif n.get("parent"):
            ev["group"] = "Branch"
        else:
            ev["group"] = "Timeline"
        events.append(ev)
    payload = {
        "title": {"text": {"headline": _doc_title(doc), "text": str(doc.get("context") or "")}},
        "events": events,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def to_bibtex(doc: Dict[str, Any]) -> str:
    seen = set()
    out: List[str] = []
    i = 0
    for n in _nodes(doc):
        y = n.get("year")
        year = str(int(y)) if isinstance(y, (int, float)) and not isinstance(y, bool) else ""
        for c in _citations(n):
            if c["url"] in seen:
                continue
            seen.add(c["url"])
            i += 1
            key = "src" + str(i)
            fields = [f"  title = {{{c['title']}}}", f"  howpublished = {{\\url{{{c['url']}}}}}"]
            if year:
                fields.append(f"  year = {{{year}}}")
            note = _claim(n)
            if note:
                fields.append(f"  note = {{{note}}}")
            out.append("@misc{" + key + ",\n" + ",\n".join(fields) + "\n}")
    return "\n\n".join(out) + ("\n" if out else "")


def to_ris(doc: Dict[str, Any]) -> str:
    recs: List[str] = []
    for n in _nodes(doc):
        y = n.get("year")
        year = str(int(y)) if isinstance(y, (int, float)) and not isinstance(y, bool) else ""
        for c in _citations(n):
            lines = ["TY  - GEN", f"TI  - {c['title']}", f"UR  - {c['url']}"]
            if year:
                lines.append(f"PY  - {year}")
            if _claim(n):
                lines.append(f"N1  - {_claim(n)}")
            lines.append("ER  - ")
            recs.append("\n".join(lines))
    return "\n\n".join(recs) + ("\n" if recs else "")


# ------------------------------------------------------------------ xml formats
def _sub(container: ET.Element, tag: str, text: str = None, **attrs) -> ET.Element:
    el = ET.SubElement(container, tag, {k: str(v) for k, v in attrs.items() if v is not None})
    if text is not None:
        el.text = text
    return el


def _xml_str(root: ET.Element) -> str:
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")


def to_xml(doc: Dict[str, Any]) -> str:
    root = ET.Element("timeline", {"title": _doc_title(doc)})
    if doc.get("context"):
        root.set("context", str(doc["context"]))
    for n in _nodes(doc):
        attrs = {"id": str(n.get("id")), "kind": n.get("kind") or "", "parent": n.get("parent") or ""}
        if isinstance(n.get("year"), (int, float)) and not isinstance(n.get("year"), bool):
            attrs["year"] = str(int(n["year"]))
        node_el = _sub(root, "node", **attrs)
        _sub(node_el, "title", _title(n))
        _sub(node_el, "when", _year_label(n))
        if _claim(n):
            _sub(node_el, "claim", _claim(n))
        cs = _citations(n)
        if cs:
            cel = _sub(node_el, "citations")
            for c in cs:
                _sub(cel, "citation", c["title"], url=c["url"])
    return _xml_str(root)


def to_opml(doc: Dict[str, Any]) -> str:
    nodes = _nodes(doc)
    root = ET.Element("opml", {"version": "2.0"})
    head = _sub(root, "head")
    _sub(head, "title", _doc_title(doc))
    body = _sub(root, "body")

    def render(node: Dict[str, Any], into: ET.Element) -> None:
        el = _sub(into, "outline", text=f"{_year_label(node)} — {_title(node)}", _note=_claim(node) or None)
        for child in _children(nodes, str(node.get("id"))):
            render(child, el)

    for n in _spine(nodes):
        render(n, body)
    return _xml_str(root)


def to_graphml(doc: Dict[str, Any]) -> str:
    nodes = _nodes(doc)
    root = ET.Element("graphml", {"xmlns": "http://graphml.graphdrawing.org/xmlns"})
    for key_id, name in (("d_label", "label"), ("d_when", "when"), ("d_claim", "claim"), ("d_kind", "kind")):
        _sub(root, "key", id=key_id, **{"for": "node", "attr.name": name, "attr.type": "string"})
    _sub(root, "key", id="e_kind", **{"for": "edge", "attr.name": "kind", "attr.type": "string"})
    g = _sub(root, "graph", id="G", edgedefault="directed")
    for n in nodes:
        nel = _sub(g, "node", id=str(n.get("id")))
        _sub(nel, "data", _title(n), key="d_label")
        _sub(nel, "data", _year_label(n), key="d_when")
        if _claim(n):
            _sub(nel, "data", _claim(n), key="d_claim")
        _sub(nel, "data", n.get("kind") or "", key="d_kind")
    for i, (s, t, kind) in enumerate(_edges(nodes)):
        eel = _sub(g, "edge", id=f"e{i}", source=s, target=t)
        _sub(eel, "data", kind, key="e_kind")
    return _xml_str(root)


# --------------------------------------------------------------------- registry
# id -> (fn, media_type, extension, human label). Ordered by usefulness.
CONVERTERS: Dict[str, Tuple[Any, str, str, str]] = {
    "json": (to_json, "application/json", "json", "JSON"),
    "csv": (to_csv, "text/csv", "csv", "CSV"),
    "markdown": (to_markdown, "text/markdown", "md", "Markdown"),
    "html": (to_html, "text/html", "html", "HTML"),
    "mermaid": (to_mermaid, "text/plain", "mmd", "Mermaid"),
    "graphml": (to_graphml, "application/xml", "graphml", "GraphML"),
    "dot": (to_dot, "text/vnd.graphviz", "dot", "Graphviz DOT"),
    "opml": (to_opml, "text/x-opml", "opml", "OPML"),
    "xml": (to_xml, "application/xml", "xml", "XML"),
    "timelinejs": (to_timelinejs, "application/json", "timeline.json", "TimelineJS JSON"),
    "bibtex": (to_bibtex, "application/x-bibtex", "bib", "BibTeX"),
    "ris": (to_ris, "application/x-research-info-systems", "ris", "RIS"),
}


def convert(fmt: str, document: Dict[str, Any]) -> Tuple[str, str, str]:
    """Return (content, media_type, extension) for the given format id."""
    entry = CONVERTERS.get(fmt)
    if not entry:
        raise KeyError(fmt)
    fn, media, ext, _label = entry
    return fn(document or {}), media, ext


def list_formats() -> List[Dict[str, str]]:
    return [{"id": k, "label": v[3], "ext": v[2], "mime": v[1]} for k, v in CONVERTERS.items()]
