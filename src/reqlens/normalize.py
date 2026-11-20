"""Pure text-normalization helpers (no I/O). Everything here is unit-tested."""
from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from html.parser import HTMLParser

_BLOCK_TAGS = {"p", "br", "div", "li", "ul", "ol", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section"}
_SKIP_TAGS = {"script", "style"}


class _Stripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")
            if tag == "li":
                self.parts.append("- ")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def collapse_ws(text: str) -> str:
    text = re.sub(r"[ \t\r\f\v\u00a0]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def strip_html(raw: str) -> str:
    """HTML -> plain text. Handles Greenhouse's entity-escaped HTML (content is HTML-escaped once).

    Never executes or fetches anything; scripts/styles are dropped. Plain text passes through
    (with entities decoded)."""
    if not raw:
        return ""
    text = raw
    # Greenhouse returns '&lt;p&gt;...' : unescape first if it looks escaped.
    if "&lt;" in text and "<" not in text:
        text = html.unescape(text)
    p = _Stripper()
    try:
        p.feed(text)
        p.close()
        out = "".join(p.parts)
    except Exception:  # malformed markup: degrade to regex strip rather than fail the run
        out = re.sub(r"<[^>]+>", " ", text)
    return normalize_text(out)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or unicodedata.category(ch)[0] != "C")
    return collapse_ws(text)


def content_hash(title: str, company: str, location: str, description: str) -> str:
    parts = [normalize_text(x).lower() for x in (title, company, location, description)]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_COMPANY_SUFFIXES = {
    "inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation", "co", "company", "gmbh",
    "plc", "pvt", "private", "ag", "sa", "bv", "oy", "ab", "technologies", "technology", "labs",
}


def normalize_company(name: str) -> str:
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    toks = [t for t in s.split() if t]
    while len(toks) > 1 and toks[-1] in _COMPANY_SUFFIXES:
        toks.pop()
    return " ".join(toks)


_TITLE_STOP = {"a", "an", "the", "and", "of", "for", "to", "in", "at", "with", "m", "f", "d", "w", "x"}
_TITLE_SYNONYMS = {
    "sr": "senior", "jr": "junior", "swe": "software engineer", "sde": "software engineer",
    "eng": "engineer", "engineering": "engineer", "developer": "engineer", "dev": "engineer",
    "backend": "backend", "back": "back", "ml": "machine learning", "intern": "intern",
    "internship": "intern", "ii": "2", "iii": "3", "i": "1",
}


def title_tokens(title: str) -> list[str]:
    s = unicodedata.normalize("NFKD", title or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\([^)]*\)", " ", s)  # drop parentheticals such as "(Remote)"
    s = s.replace("c++", "cpp").replace("c#", "csharp")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    out: list[str] = []
    for t in s.split():
        if t in _TITLE_STOP:
            continue
        out.extend(_TITLE_SYNONYMS.get(t, t).split())
    return out


def normalize_title(title: str) -> str:
    return " ".join(title_tokens(title))


_LOC_ALIASES = {
    "sf": "san francisco", "nyc": "new york", "ny": "new york", "la": "los angeles",
    "usa": "united states", "us": "united states", "uk": "united kingdom", "blr": "bengaluru",
    "bangalore": "bengaluru", "bombay": "mumbai",
}
_LOC_STOP = {
    "office", "hq", "area", "bay", "greater", "metro", "region", "or", "and", "based",
    "new", "san", "los", "las", "st", "saint", "united",
    "on", "site", "onsite", "hybrid", "remote", "in", "person", "anywhere", "worldwide", "ok", "only", "friendly",
}


def location_tokens(loc: str) -> set[str]:
    s = unicodedata.normalize("NFKD", loc or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]+", " ", s)
    out: set[str] = set()
    for t in s.split():
        t = _LOC_ALIASES.get(t, t)
        out.update(x for x in t.split() if x not in _LOC_STOP)
    return out


def place_sets(loc: str) -> list[set[str]]:
    """One token set per listed place, using the CITY part only ('London, UK' -> {london}).

    Multi-location strings ('New York; San Francisco / Remote') yield several sets. Country/state
    tokens are deliberately dropped so 'London, UK' and 'Manchester, UK' do not look compatible."""
    out: list[set[str]] = []
    for seg in re.split(r"[;|/]| or ", loc or ""):
        city = seg.split(",")[0]
        toks = location_tokens(city)
        if toks:
            out.append(toks)
    return out
