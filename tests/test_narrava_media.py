"""Narrava Studio stock media: what a result links to.

phansora's import route stores web rasters only (png, jpg, gif, webp). A result linking to
anything else — a Commons SVG or TIFF original, the placeholder's default SVG — was refused on
import ("That file type is not supported here") and left the scene hotlinked. These pin that
every result links to something the studio can import. Commons is replaced with a stub, so
they run offline.
"""
from types import SimpleNamespace

from phansora.products.narrava_studio.services import media

ORIG = "https://upload.wikimedia.org/wikipedia/commons/{p}?utm_content=original"
THUMB = "https://thumb.wikimedia.org/wikipedia/commons/thumb/{p}/960px-{n}.png?utm_content=thumbnail"


def _page(title, mime, path):
    name = path.rsplit("/", 1)[-1]
    return {
        "title": f"File:{title}",
        "imageinfo": [{
            "mime": mime,
            "url": ORIG.format(p=path),
            "thumburl": THUMB.format(p=path, n=name),
            "descriptionurl": f"https://commons.wikimedia.org/wiki/File:{title}",
            "extmetadata": {"Artist": {"value": "Someone"}, "LicenseShortName": {"value": "CC BY-SA 4.0"}},
        }],
    }


def test_commons_links_a_web_raster_as_it_is_and_anything_else_as_its_rendering(monkeypatch):
    pages = {
        "1": _page("Lighthouse.jpg", "image/jpeg", "a/ab/Lighthouse.jpg"),
        "2": _page("Lighthouse icon.svg", "image/svg+xml", "7/7d/Lighthouse_icon.svg"),
        "3": _page("Survey map.tif", "image/tiff", "c/c0/Survey_map.tif"),
        "4": _page("Foghorn.ogg", "application/ogg", "f/f0/Foghorn.ogg"),
    }

    def fake_get(url, **kwargs):
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"query": {"pages": pages}})

    monkeypatch.setattr(media.httpx, "get", fake_get)
    clips = media._wikimedia_images("lighthouse", segment_id="s1", limit=10)

    by_title = {c.title: c for c in clips}
    # Not an image at all: left out, as before.
    assert "Foghorn.ogg" not in by_title
    # A JPEG original is the best copy there is.
    assert by_title["Lighthouse.jpg"].url == ORIG.format(p="a/ab/Lighthouse.jpg")
    # An SVG or a TIFF links to the PNG Commons renders of it, never the original.
    assert by_title["Lighthouse icon.svg"].url.split("?")[0].endswith(".svg.png")
    assert by_title["Survey map.tif"].url.split("?")[0].endswith(".tif.png")
    for clip in clips:
        assert clip.url.split("?")[0].rsplit(".", 1)[-1] in {"jpg", "jpeg", "png", "gif", "webp"}


def test_the_placeholder_is_a_png():
    clip = media._placeholder("city at night", segment_id="s1", media_type="image")
    assert "/png?" in clip.url and "/png?" in clip.thumbnail_url
