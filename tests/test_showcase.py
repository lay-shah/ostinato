"""Check offline navigation for both public-facing pages."""

from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit


DOCS = Path(__file__).resolve().parents[1] / "docs"
PAGES = {"index.html", "PROJECT_REPORT.html"}


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = Counter()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids[attrs["id"]] += 1
        if tag == "a" and "href" in attrs:
            self.hrefs.append(attrs["href"])


def test_showcase_navigation():
    parsed = {}
    for page in PAGES:
        document = Links()
        document.feed((DOCS / page).read_text(encoding="utf-8"))
        assert all(count == 1 for count in document.ids.values()), page
        parsed[page] = document

    for page, document in parsed.items():
        assert PAGES - {page} <= {urlsplit(link).path for link in document.hrefs}
        for link in document.hrefs:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            target = url.path or page
            assert target in parsed, (page, link)
            if url.fragment:
                assert url.fragment in parsed[target].ids, (page, link)
