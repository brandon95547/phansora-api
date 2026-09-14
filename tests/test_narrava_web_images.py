"""Narrava Studio web image search tests.

DuckDuckGo is replaced by an httpx MockTransport, so these run offline. The property they
pin is the one ddgs lost: a search that found nothing and a search that was throttled are
different answers.
"""
import httpx
import pytest

from phansora.products.narrava_studio.services import web_images

PAGE = '<html><script>vqd="4-123456789012345678901234567890"</script></html>'


def _use(monkeypatch, handler):
    real = httpx.Client

    def client(**kwargs):
        return real(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(web_images.httpx, "Client", client)


def _ddg(results, *, page_status=200, images_status=200):
    def handler(request):
        if request.url.path == "/i.js":
            assert request.url.params["vqd"] == "4-123456789012345678901234567890"
            if images_status != 200:
                return httpx.Response(images_status, text="If this error persists, please let us know")
            return httpx.Response(200, json={"results": results})
        return httpx.Response(page_status, text=PAGE if page_status == 200 else "")
    return handler


def test_results_carry_where_they_were_found(monkeypatch):
    _use(monkeypatch, _ddg([{
        "title": "Harbour at dawn",
        "image": "https://i.redd.it/qpy47hasatk61.jpg",
        "thumbnail": "https://tse1.mm.bing.net/th?id=OIP.x",
        "url": "https://www.reddit.com/r/santacruz/comments/lwtff2/",
    }]))

    [clip] = web_images.search_web_images("harbour at dawn")

    assert clip.type == "image"
    assert clip.url == "https://i.redd.it/qpy47hasatk61.jpg"
    assert clip.thumbnail_url == "https://tse1.mm.bing.net/th?id=OIP.x"
    assert clip.source == "reddit.com"
    assert clip.attribution == "Found on reddit.com"
    # The credits list prints the licence and its link; for a web picture those are "we do
    # not know" and the page that would tell you.
    assert clip.license == web_images.RIGHTS_UNKNOWN
    assert clip.license_url == "https://www.reddit.com/r/santacruz/comments/lwtff2/"


def test_nothing_found_is_an_empty_list(monkeypatch):
    _use(monkeypatch, _ddg([]))
    assert web_images.search_web_images("xqzvbnmplk wqrtyzx") == []


@pytest.mark.parametrize("where", ["page", "images"])
def test_a_throttled_search_is_not_an_empty_one(monkeypatch, where):
    """Regression guard for the ddgs failure: 202/403 must never come back as []."""
    if where == "page":
        _use(monkeypatch, _ddg([], page_status=202))
    else:
        _use(monkeypatch, _ddg([], images_status=403))

    with pytest.raises(web_images.WebSearchUnavailable, match="limiting searches"):
        web_images.search_web_images("harbour at dawn")


def test_unreachable_is_unavailable(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("no route", request=request)
    _use(monkeypatch, handler)

    with pytest.raises(web_images.WebSearchUnavailable, match="could not reach"):
        web_images.search_web_images("harbour at dawn")


def test_a_page_without_a_token_is_unavailable(monkeypatch):
    """DuckDuckGo changing its page is an outage for us, not an absence of pictures."""
    def handler(request):
        return httpx.Response(200, text="<html>no token here</html>")
    _use(monkeypatch, handler)

    with pytest.raises(web_images.WebSearchUnavailable):
        web_images.search_web_images("harbour at dawn")


def test_pictures_the_import_cannot_store_are_left_out(monkeypatch):
    _use(monkeypatch, _ddg([
        {"image": "https://example.com/logo.svg", "url": "https://example.com/"},
        {"image": "ftp://example.com/a.jpg", "url": "https://example.com/"},
        {"image": "", "url": "https://example.com/"},
        {"image": "https://example.com/a.jpg", "url": "https://example.com/"},
        {"image": "https://example.com/a.jpg", "url": "https://example.com/again"},
        {"image": "http://example.org/b.png?w=800", "url": "http://example.org/"},
    ]))

    urls = [c.url for c in web_images.search_web_images("anything")]

    assert urls == ["https://example.com/a.jpg", "http://example.org/b.png?w=800"]


def test_limit_is_honoured(monkeypatch):
    _use(monkeypatch, _ddg([
        {"image": f"https://example.com/{i}.jpg", "url": "https://example.com/"} for i in range(100)
    ]))
    assert len(web_images.search_web_images("anything", limit=12)) == 12
