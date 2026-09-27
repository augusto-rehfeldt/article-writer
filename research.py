"""Literature and news gathering.

Everything here is keyless and free: OpenAlex, Crossref, DOAJ, Semantic Scholar and
arXiv for papers; Google News RSS and GDELT for current events; Wikipedia; the
theory archives via their own APIs; Gutenberg, archive.org and Library Genesis for
books; Unpaywall and Sci-Hub for full text, plus whatever the user drops in
``library/``. Anna's Archive is a hand-download link in ``_fallback_links`` and
nothing more: Cloudflare blocks every keyless client, so scraping it only ever
cost three 45-second timeouts per gap and returned the link list anyway.

The point of the dossier is that the drafting model may only cite what is in it.
Anything else is a hallucination and gets flagged by ``verify_citations``.
"""

from __future__ import annotations

import atexit
import collections
import dataclasses
import email.utils
import functools
import html
import io
import json
import logging
import math
import os
import pathlib
import re
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
import unicodedata
import urllib.parse
from concurrent import futures
from xml.etree import ElementTree
from typing import Any

import warnings

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

ROOT = pathlib.Path(__file__).parent
LIBRARY = ROOT / "library"
# The user's Calibre library, read-only. Calibre's own default location is
# ~/Calibre Library; AW_CALIBRE overrides it. Consulted by title in
# ``calibre_book()`` when a work is missing — never scanned wholesale, because
# ``scan_library`` puts everything it finds in the dossier and a general-purpose
# ebook library is mostly fiction.
CALIBRE = pathlib.Path(os.environ.get("AW_CALIBRE")
                       or pathlib.Path.home() / "Calibre Library")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")
MAILTO = "research@example.org"  # OpenAlex/Crossref polite pool

SCIHUB_MIRRORS = ["https://sci-hub.se", "https://sci-hub.st", "https://sci-hub.ru",
                  "https://sci-hub.wf", "https://sci-hub.ee"]

# Theory archives searched through their own APIs. General search engines
# (DuckDuckGo, SearX, Bing) all block or rate-limit unauthenticated scraping,
# so each site is queried through whatever native endpoint it exposes:
# WordPress sites via /wp-json/wp/v2/search, libcom via its Drupal search.
WP_ARCHIVES = ["monthlyreview.org", "www.viewpointmag.com", "brooklynrail.org",
               "spectrejournal.com", "historicalmaterialism.org"]

# Seconds to wait between consecutive requests to the same host. arXiv and GDELT
# both hand out 429s aggressively; without this most of the dossier comes back empty.
_HOST_DELAY = {"export.arxiv.org": 8.0, "api.gdeltproject.org": 8.0,
               "api.semanticscholar.org": 3.5, "api.crossref.org": 1.0,
               "api.openalex.org": 0.4, "doaj.org": 1.5,
               "archive.org": 1.0, "libgen.li": 3.0, "libgen.vg": 3.0}

# Library Genesis mirrors that still answer. libgen.is/.rs/.st are dead from here;
# these two share the same index and the same /ads.php -> /get.php download flow.
LIBGEN_MIRRORS = ["https://libgen.li", "https://libgen.vg"]
_last_hit: dict[str, float] = {}
# Connectors run in parallel across hosts; the lock keeps each host serialised so
# the delay above is actually respected instead of being raced past.
_host_locks: dict[str, threading.Lock] = collections.defaultdict(threading.Lock)

_session = requests.Session()
_session.headers["User-Agent"] = UA


# --------------------------------------------------------------------------- #
# Source records
# --------------------------------------------------------------------------- #

_AGGREGATOR_VENUE = re.compile(
    r"biblioteca local|repositor|repository|la referencia|project muse|aecid", re.I)


def _clean_venue(venue: str) -> str:
    """Aggregators index the hosting archive, not the journal: OpenAlex answers
    «Project Muse (Johns Hopkins University)», «LA Referencia (Red Federada…)»,
    «NASA STI Repository». Citing those as the venue points at the mirror, not
    the work — drop them and let the DOI or URL speak."""
    return "" if _AGGREGATOR_VENUE.search(venue or "") else (venue or "").strip()


@dataclasses.dataclass
class Source:
    key: str = ""            # citation key used in the text, e.g. "Postone, 2006"
    title: str = ""
    authors: list[str] = dataclasses.field(default_factory=list)
    year: str = ""
    kind: str = "web"        # paper | book | news | web | video | local
    date: str = ""           # ISO publication date, when the connector knows it
    venue: str = ""
    url: str = ""
    doi: str = ""
    abstract: str = ""
    fulltext: str = ""       # populated only when we actually retrieved it
    origin: str = ""         # which connector produced it

    def citation(self, lang: str = "es") -> str:
        """APA 7 reference-list entry.

        APA wants "Apellido, N. N." with the surname first and initials after,
        and italicised titles — rendered here as markdown emphasis.
        """
        names = [_apa_name(a) for a in self.authors if a.strip()]
        if not names:
            author = "Anonymous" if lang == "en" else "Anónimo"
        elif len(names) == 1:
            author = names[0]
        elif len(names) <= 20:
            author = ", ".join(names[:-1]) + (", & " if lang == "en" else " y ") + names[-1]
        else:
            author = ", ".join(names[:19]) + ", … " + names[-1]
        year = self.year or ("n.d." if lang == "en" else "s.f.")
        if self.key and self.year and self.key.rsplit(", ", 1)[-1].startswith(self.year):
            year = self.key.rsplit(", ", 1)[-1]
        bits = [f"{author} ({year}). "]
        venue = _clean_venue(self.venue)
        if self.kind in ("paper", "news", "web") and venue:
            bits.append(f"{self.title}. *{venue}*.")
        elif venue:
            bits.append(f"*{self.title}*. {venue}.")
        else:
            bits.append(f"*{self.title}*.")
        if self.doi:
            bits.append(f" https://doi.org/{self.doi}")
        elif self.url and not self.url.startswith(("C:", "D:", "/")):
            bits.append(f" {self.url}")
        return "".join(bits)

    def brief(self, chars: int = 1400) -> str:
        body = self.fulltext or self.abstract
        # The exact date matters for news: the draft has to say "el 14 de agosto",
        # not "recientemente", and has to prefer the freshest figure available.
        when = f" [{self.date}]" if self.date else ""
        return (f"[{self.key}] {self.title} ({self.year}){when} — {self.kind}, "
                f"{self.venue or self.origin}\n{body[:chars]}").strip()


def _surname(authors: list[str]) -> str:
    if not authors:
        return "Anon"
    first = authors[0].strip()
    if "," in first:                      # already "Apellido, Nombre"
        return first.split(",")[0].strip() or "Anon"
    parts = first.split()
    return parts[-1] if parts else "Anon"


def _apa_name(author: str) -> str:
    """"Moishe Postone" -> "Postone, M."; leaves institutional names alone."""
    author = author.strip()
    if "," in author:                     # already "Apellido, N. N." — leave the dots alone
        return author
    parts = author.rstrip(".").split()
    if len(parts) < 2:
        return author
    initials = " ".join(f"{p[0].upper()}." for p in parts[:-1] if p)
    return f"{parts[-1]}, {initials}"


def assign_keys(sources: list[Source]) -> list[Source]:
    """Give each source a unique ``Apellido, año`` key.

    Follows the usual academic convention: when an author has more than one work
    in the same year, *all* of them get a letter (1998a, 1998b), never a bare
    year alongside a lettered one.
    """
    counts: collections.Counter[str] = collections.Counter(
        f"{_surname(s.authors)}, {s.year or 's/f'}" for s in sources)
    used: dict[str, int] = {}
    for s in sources:
        base = f"{_surname(s.authors)}, {s.year or 's/f'}"
        n = used.get(base, 0)
        used[base] = n + 1
        s.key = f"{base}{chr(ord('a') + n)}" if counts[base] > 1 else base
    return sources


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #

def _get(url: str, *, params: dict | None = None, timeout: int = 30,
         retries: int = 2, **kw: Any) -> requests.Response | None:
    host = urllib.parse.urlparse(url).netloc
    delay = _HOST_DELAY.get(host, 0.0)
    for attempt in range(retries + 1):
        try:
            if delay:
                with _host_locks[host]:
                    wait = delay - (time.monotonic() - _last_hit.get(host, 0.0))
                    if wait > 0:
                        time.sleep(wait)
                    r = _session.get(url, params=params, timeout=timeout, **kw)
                    _last_hit[host] = time.monotonic()
            else:
                r = _session.get(url, params=params, timeout=timeout, **kw)
            if r.status_code < 400:
                return r
            if r.status_code not in (429, 500, 502, 503) or attempt == retries:
                return None
        except requests.RequestException:
            if attempt == retries:
                return None
        time.sleep(6 + attempt * 8 if delay else 2 + attempt * 4)
    return None


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t\xa0]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _soup(markup) -> BeautifulSoup:
    # An XML document under an HTML parser raises XMLParsedAsHTMLWarning and
    # mangles entities; one carrying a declaration goes to the lxml XML parser,
    # and anything that fails to parse as XML falls back to the tolerant HTML one.
    if str(markup[:200]).lstrip().startswith("<?xml"):
        try:
            return BeautifulSoup(markup, "xml")
        except Exception:
            pass
    return BeautifulSoup(markup, "lxml")


def fetch_text(url: str, max_chars: int = 0, timeout: int = 25) -> str:
    """Readable text from an arbitrary URL (HTML or PDF), whole by default.

    ``max_chars=0`` reads the whole thing: a cut is a mutilated copy, and the only
    thing that genuinely needs bounding is what goes into the drafting context (see
    ``Source.brief``), which slices per call. Callers that still cap a specific
    thing can pass a value, but no fetch does by default.
    """
    r = _get(url, timeout=timeout, retries=0)
    if r is None:
        return ""
    ctype = r.headers.get("content-type", "")
    is_pdf = "pdf" in ctype or url.lower().endswith(".pdf")
    # A URL ending in .pdf (or claimed as one) can still answer HTML — a login
    # redirect, an interstitial. Feeding that to the PDF parser is what printed
    # "invalid pdf header" / "EOF marker not found"; only real PDFs (magic bytes)
    # go through, and anything else falls through to the HTML extraction below.
    if is_pdf and r.content[:4] == b"%PDF":
        return pdf_text(r.content)[:max_chars or None]
    soup = _soup(r.text)
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()
    main = soup.find("article") or soup.find("main") or soup.body or soup
    return _clean(main.get_text("\n"))[:max_chars or None]


def pdf_text(data: bytes) -> str:
    # pypdf logs an "invalid pdf header" / "EOF marker not found" warning at the
    # module level (not through warnings.warn, so it ignores warnings filters) for
    # every non-compliant PDF it recovers. Most of what Unpaywall/Sci-Hub hand us
    # is non-compliant, and the warnings scroll the stage's own log off the console.
    # The reader then recovers fine, so the warnings are noise — silence them.
    logging.getLogger("pypdf").setLevel(logging.CRITICAL)
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        # A [p. N] marker before each page is what lets the draft write
        # (Apellido, año, p. 302) for a verbatim quote instead of inventing the
        # page or dropping it. Only PDFs have pages: ePub/HTML/plain text carry
        # no markers and stay pageless on purpose.
        # ponytail: N is the PDF's own page index, so a book with front matter
        # can be off by a few; read the printed folio from the page text if it
        # ever matters enough to pay for.
        pages = ((i, (p.extract_text() or "").strip())
                 for i, p in enumerate(reader.pages, 1))
        # An empty page gets no marker: a trailing blank page in a scan would
        # otherwise leave the text ending on "[p. 312]", which _cut() reads as a
        # work that stops mid-sentence.
        return _clean("\n".join(f"\n[p. {i}]\n{t}" for i, t in pages if t))
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# Academic connectors
# --------------------------------------------------------------------------- #

def _deinvert(idx: dict[str, list[int]] | None) -> str:
    """OpenAlex stores abstracts as an inverted index; rebuild the prose."""
    if not idx:
        return ""
    positions: list[tuple[int, str]] = [(p, w) for w, ps in idx.items() for p in ps]
    return " ".join(w for _, w in sorted(positions))


def openalex(query: str, limit: int = 8, lang: str | None = None) -> list[Source]:
    params = {"search": query, "per-page": limit, "mailto": MAILTO,
              "sort": "relevance_score:desc"}
    if lang:
        params["filter"] = f"language:{lang}"
    r = _get("https://api.openalex.org/works", params=params)
    if r is None:
        return []
    out = []
    for w in r.json().get("results", []):
        out.append(Source(
            title=w.get("display_name") or "",
            authors=[a["author"]["display_name"] for a in w.get("authorships", [])[:6]
                     if a.get("author")],
            year=str(w.get("publication_year") or ""),
            kind="paper",
            venue=((w.get("primary_location") or {}).get("source") or {}).get("display_name") or "",
            doi=(w.get("doi") or "").replace("https://doi.org/", ""),
            url=(w.get("best_oa_location") or {}).get("pdf_url") or w.get("doi") or "",
            abstract=_deinvert(w.get("abstract_inverted_index")),
            origin="openalex"))
    return out


def crossref(query: str, limit: int = 8) -> list[Source]:
    r = _get("https://api.crossref.org/works",
             params={"query": query, "rows": limit, "mailto": MAILTO, "select":
                     "title,author,issued,DOI,abstract,container-title,type,URL"})
    if r is None:
        return []
    out = []
    for it in r.json().get("message", {}).get("items", []):
        issued = (it.get("issued") or {}).get("date-parts") or [[None]]
        abstract = re.sub(r"<[^>]+>", " ", it.get("abstract") or "")
        out.append(Source(
            title=(it.get("title") or [""])[0],
            authors=[f"{a.get('given','')} {a.get('family','')}".strip()
                     for a in it.get("author", [])[:6]],
            year=str(issued[0][0] or ""),
            kind="book" if "book" in (it.get("type") or "") else "paper",
            venue=(it.get("container-title") or [""])[0],
            doi=it.get("DOI", ""), url=it.get("URL", ""),
            abstract=_clean(abstract), origin="crossref"))
    return out


def arxiv(query: str, limit: int = 6) -> list[Source]:
    r = _get("https://export.arxiv.org/api/query",
             params={"search_query": f"all:{query}", "max_results": limit})
    if r is None:
        return []
    soup = BeautifulSoup(r.text, "xml")
    out = []
    for e in soup.find_all("entry"):
        out.append(Source(
            title=_clean(e.title.get_text()) if e.title else "",
            authors=[a.get_text().strip() for a in e.find_all("name")][:6],
            year=(e.published.get_text()[:4] if e.published else ""),
            kind="paper", venue="arXiv",
            url=e.id.get_text().strip() if e.id else "",
            abstract=_clean(e.summary.get_text()) if e.summary else "",
            origin="arxiv"))
    return out


def doaj(query: str, limit: int = 6) -> list[Source]:
    """Directory of Open Access Journals — strong Spanish-language coverage."""
    r = _get("https://doaj.org/api/search/articles/" + urllib.parse.quote(query),
             params={"pageSize": limit})
    if r is None:
        return []
    try:
        results = r.json().get("results", [])
    except (json.JSONDecodeError, ValueError):
        return []
    out = []
    for it in results:
        b = it.get("bibjson", {})
        link = next((l.get("url", "") for l in b.get("link", []) if l.get("url")), "")
        doi = next((i.get("id", "") for i in b.get("identifier", [])
                    if i.get("type") == "doi"), "")
        out.append(Source(
            title=b.get("title", ""),
            authors=[a.get("name", "") for a in b.get("author", [])][:6],
            year=str(b.get("year") or ""), kind="paper",
            venue=(b.get("journal") or {}).get("title", ""),
            doi=doi, url=link, abstract=_clean(b.get("abstract") or ""), origin="doaj"))
    return out


def semantic_scholar(query: str, limit: int = 6) -> list[Source]:
    r = _get("https://api.semanticscholar.org/graph/v1/paper/search",
             params={"query": query, "limit": limit,
                     "fields": "title,abstract,year,authors,venue,externalIds,openAccessPdf,tldr"})
    if r is None:
        return []
    out = []
    for p in r.json().get("data", []) or []:
        tldr = (p.get("tldr") or {}).get("text") or ""
        out.append(Source(
            title=p.get("title") or "",
            authors=[a.get("name", "") for a in p.get("authors", [])][:6],
            year=str(p.get("year") or ""), kind="paper", venue=p.get("venue") or "",
            doi=(p.get("externalIds") or {}).get("DOI", "") or "",
            url=(p.get("openAccessPdf") or {}).get("url", "") or "",
            abstract=_clean(p.get("abstract") or tldr), origin="semanticscholar"))
    return out


# --------------------------------------------------------------------------- #
# News, encyclopedia, open web
# --------------------------------------------------------------------------- #

def google_news(query: str, limit: int = 10, lang: str = "es",
                days: int | None = None) -> list[Source]:
    """Primary news source: Google News RSS. Keyless, unthrottled, and it
    actually answers, which GDELT only does intermittently.

    ``days`` appends Google's own ``when:Nd`` operator, which is the only way to
    stop the feed from mixing last week's news with articles from two years ago.
    """
    locale = {"es": ("es-419", "AR", "AR:es"), "en": ("en-US", "US", "US:en")}.get(
        lang, ("es-419", "AR", "AR:es"))
    q = f"{query} when:{days}d" if days else query
    r = _get("https://news.google.com/rss/search",
             params={"q": q, "hl": locale[0], "gl": locale[1], "ceid": locale[2]})
    if r is None:
        return []
    try:
        root = ElementTree.fromstring(r.content)
    except ElementTree.ParseError:
        return []
    out = []
    for item in root.findall(".//item")[:limit]:
        title = item.findtext("title") or ""
        src = item.find("source")
        outlet = (src.text if src is not None else "") or ""
        pub = item.findtext("pubDate") or ""
        snippet = BeautifulSoup(item.findtext("description") or "", "lxml").get_text(" ", strip=True)
        out.append(Source(title=title[:250], authors=[outlet] if outlet else [],
                          year=(re.search(r"\b(20\d{2})\b", pub).group(1)
                                if re.search(r"\b(20\d{2})\b", pub) else ""),
                          kind="news", date=_iso_date(pub), venue=outlet,
                          url=item.findtext("link") or "",
                          abstract=snippet[:800], origin="googlenews"))
    out.sort(key=lambda s: s.date, reverse=True)   # freshest first, always
    return out


def _iso_date(rfc822: str) -> str:
    """'Thu, 20 Aug 2026 17:19:13 GMT' -> '2026-08-20'. Empty when unparseable."""
    try:
        return email.utils.parsedate_to_datetime(rfc822).date().isoformat()
    except (TypeError, ValueError):
        return ""


def gdelt_news(query: str, limit: int = 10, months: int = 6,
               lang: str = "spanish") -> list[Source]:
    """Current events. GDELT indexes world news in ~65 languages, no key needed.

    Multi-word queries must be quoted as a phrase or GDELT returns an error page
    instead of JSON, so anything with a space gets wrapped.
    """
    q = f'"{query}"' if " " in query.strip() else query.strip()
    r = _get("https://api.gdeltproject.org/api/v2/doc/doc",
             params={"query": f"{q} sourcelang:{lang}", "mode": "artlist",
                     "maxrecords": limit, "format": "json",
                     "timespan": f"{max(1, months)}m", "sort": "hybridrel"})
    if r is None:
        return []
    try:
        arts = r.json().get("articles", [])
    except (json.JSONDecodeError, ValueError):
        return []
    return [Source(title=a.get("title", ""), authors=[a.get("domain", "")],
                   year=(a.get("seendate", "") or "")[:4], kind="news",
                   date="-".join(filter(None, (a.get("seendate", "")[:4],
                                               a.get("seendate", "")[4:6],
                                               a.get("seendate", "")[6:8]))),
                   venue=a.get("domain", ""), url=a.get("url", ""),
                   origin="gdelt") for a in arts]


def wikipedia(query: str, lang: str = "es", limit: int = 3) -> list[Source]:
    api = f"https://{lang}.wikipedia.org/w/api.php"
    r = _get(api, params={"action": "query", "list": "search", "srsearch": query,
                          "format": "json", "srlimit": limit})
    if r is None:
        return []
    out = []
    for hit in r.json().get("query", {}).get("search", []):
        title = hit["title"]
        e = _get(api, params={"action": "query", "prop": "extracts", "explaintext": 1,
                              "titles": title, "format": "json", "exintro": 0})
        text = ""
        if e is not None:
            pages = e.json().get("query", {}).get("pages", {})
            text = next(iter(pages.values()), {}).get("extract", "")
        out.append(Source(title=title, authors=["Wikipedia"], year=time.strftime("%Y"),
                          kind="web", venue=f"Wikipedia ({lang})",
                          url=f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title)}",
                          abstract=_clean(text)[:6000], origin="wikipedia"))
    return out


def wp_search(domain: str, query: str, limit: int = 3) -> list[dict[str, str]]:
    """Any WordPress site exposes /wp-json/wp/v2/search. No key, no scraping."""
    r = _get(f"https://{domain}/wp-json/wp/v2/search",
             params={"search": query, "per_page": limit}, timeout=25)
    if r is None:
        return []
    try:
        return [{"title": x.get("title", ""), "url": x.get("url", "")}
                for x in r.json() if x.get("url")]
    except (json.JSONDecodeError, ValueError, TypeError):
        return []


def libcom_search(query: str, limit: int = 4) -> list[dict[str, str]]:
    r = _get("https://libcom.org/search", params={"search_api_fulltext": query}, timeout=30)
    if r is None:
        return []
    soup = BeautifulSoup(r.text, "lxml")
    seen, out = set(), []
    for a in soup.select("article a"):
        href = a.get("href", "")
        title = a.get_text(" ", strip=True)
        if not href.startswith("/article/") or href in seen or len(title) < 8:
            continue
        seen.add(href)
        out.append({"title": title, "url": "https://libcom.org" + href})
        if len(out) >= limit:
            break
    return out


def wpcom_search(site: str, query: str, limit: int = 3) -> list[dict[str, str]]:
    """WordPress.com-hosted blogs (including the author's own) use a different API."""
    r = _get(f"https://public-api.wordpress.com/rest/v1.1/sites/{site}/posts",
             params={"search": query, "number": limit, "fields": "title,URL"})
    if r is None:
        return []
    try:
        return [{"title": p.get("title", ""), "url": p.get("URL", "")}
                for p in r.json().get("posts", [])]
    except (json.JSONDecodeError, ValueError):
        return []


def archive_search(query: str, per_site: int = 2) -> list[Source]:
    """Search the left-theory archives through their own endpoints."""
    hits: list[tuple[str, dict[str, str]]] = []
    for dom in WP_ARCHIVES:
        hits += [(dom, h) for h in wp_search(dom, query, per_site)]
    hits += [("libcom.org", h) for h in libcom_search(query, per_site + 1)]
    hits += [("revistalacueva.wordpress.com", h)
             for h in wpcom_search("revistalacueva.wordpress.com", query, 1)]
    out = []
    for dom, h in hits:
        body = fetch_text(h["url"])
        title = html.unescape(re.sub(r"<[^>]+>", "", h["title"]))
        out.append(Source(title=title[:200], authors=[dom],
                          year="", kind="web", venue=dom, url=h["url"],
                          abstract=(body or h["title"])[:1500], fulltext=body,
                          origin="archive"))
    return out


# --------------------------------------------------------------------------- #
# Full text acquisition: Sci-Hub, Anna's Archive, Open Library, Gutenberg
# --------------------------------------------------------------------------- #

def unpaywall_pdf(doi: str, email: str = MAILTO) -> bytes | None:
    """Legal open-access copy of a DOI, when one exists. Tried before Sci-Hub."""
    if not doi:
        return None
    r = _get(f"https://api.unpaywall.org/v2/{doi}", params={"email": email}, timeout=30)
    if r is None:
        return None
    try:
        loc = r.json().get("best_oa_location") or {}
    except (json.JSONDecodeError, ValueError):
        return None
    url = loc.get("url_for_pdf") or loc.get("url")
    if not url:
        return None
    pdf = _get(url, timeout=60)
    return pdf.content if pdf is not None and pdf.content[:4] == b"%PDF" else None


def scihub_pdf(doi: str) -> bytes | None:
    """Fetch a paywalled paper by DOI. Research use; mirrors rotate constantly.

    Mirrors sit behind JS challenges much of the time; failures here are normal
    and must stay cheap, hence ``retries=0`` and a short timeout.
    """
    if not doi:
        return None
    for mirror in SCIHUB_MIRRORS:
        r = _get(f"{mirror}/{doi}", timeout=20, retries=0)
        if r is None:
            continue
        if r.headers.get("content-type", "").startswith("application/pdf"):
            return r.content
        soup = BeautifulSoup(r.text, "lxml")
        src = ""
        for sel in ("embed#pdf", "iframe#pdf", "#article embed", "#article iframe"):
            el = soup.select_one(sel)
            if el and el.get("src"):
                src = el["src"]
                break
        if not src:
            m = re.search(r"location\.href\s*=\s*'([^']+\.pdf[^']*)'", r.text)
            src = m.group(1) if m else ""
        if not src:
            continue
        if src.startswith("//"):
            src = "https:" + src
        elif src.startswith("/"):
            src = mirror + src
        pdf = _get(src.split("#")[0], timeout=90)
        if pdf is not None and pdf.content[:4] == b"%PDF":
            return pdf.content
    return None


def gutenberg(query: str, limit: int = 3) -> list[Source]:
    r = _get("https://gutendex.com/books", params={"search": query})
    if r is None:
        return []
    out = []
    for b in r.json().get("results", [])[:limit]:
        txt_url = next((u for k, u in b.get("formats", {}).items()
                        if k.startswith("text/plain")), "")
        body = fetch_text(txt_url) if txt_url else ""
        out.append(Source(title=b.get("title", ""),
                          authors=[a["name"] for a in b.get("authors", [])],
                          year=str((b.get("authors") or [{}])[0].get("death_year") or ""),
                          kind="book", venue="Project Gutenberg",
                          url=txt_url or f"https://gutenberg.org/ebooks/{b.get('id')}",
                          abstract=body[:2000], fulltext=body, origin="gutenberg"))
    return out


# Under this an archive.org "book" is a catalogue record or a leaflet, not a work.
_ARCHIVE_MIN_CHARS = 15000


def archive_org_book(query: str) -> tuple[str, dict] | None:
    """OCR full text of a book from archive.org. Returns (text, catalogue record).

    Only works for items that are not lending-restricted — for those the ``_djvu.txt``
    file 403s and we move on. Tried before Library Genesis because it is free,
    fast and openly published.
    """
    r = _get("https://archive.org/advancedsearch.php",
             params={"q": f"{query} AND mediatype:texts",
                     "fl[]": ["identifier", "title", "creator", "year"],
                     "rows": 4, "output": "json"}, timeout=45, retries=1)
    if r is None:
        return None
    try:
        docs = r.json().get("response", {}).get("docs", [])
    except (json.JSONDecodeError, ValueError):
        return None
    for doc in docs:
        if not _covers(query, f"{doc.get('title','')} {doc.get('creator','')}"):
            continue
        meta = _get(f"https://archive.org/metadata/{doc.get('identifier','')}",
                    timeout=45, retries=0)
        if meta is None:
            continue
        try:
            j = meta.json()
        except (json.JSONDecodeError, ValueError):
            continue
        server, folder = j.get("server", ""), j.get("dir", "")
        names = [f["name"] for f in j.get("files", []) if f["name"].endswith("_djvu.txt")]
        if not (server and folder and names):
            continue
        body = ""
        # Multi-volume items split the text into one file per volume; take them all,
        # in order — a cut is a mutilated copy, and the dossier's own slice belongs
        # to ``Source.brief``, not to the file written into ``library/``.
        for name in sorted(names):
            got = _get(f"https://{server}{folder}/{urllib.parse.quote(name)}",
                       timeout=120, retries=0)
            if got is not None:
                body += _clean(got.text) + "\n\n"
        if len(body) >= _ARCHIVE_MIN_CHARS:
            return body, doc
    return None


def libgen_book(query: str, max_bytes: int = 200_000_000) -> tuple[bytes, str] | None:
    """Book file from Library Genesis. Returns (bytes, extension).

    Same posture the code already takes with Sci-Hub for papers: used only for what
    is not obtainable otherwise. epub before pdf (cleaner text, smaller download),
    smallest file first, three candidates at most — each download can take a minute.

    The ceiling was 40 MB, which is a *text* PDF's size, not a scan's. Measured
    2026-08-23: Library Genesis holds Finney y Jones, *Interstellar Migration and
    the Human Experience*, the exact edition asked for, as a 75 MB scan of 368
    pages — the only candidate for that work in either mirror, and it was dropped
    unread while the run reported the book as unfindable. Sorting still puts the
    small files first, so the cap only ever decides for a work that has nothing
    smaller.
    """
    for host in LIBGEN_MIRRORS:
        r = _get(f"{host}/index.php",
                 params={"req": query, "topics[]": "l", "res": "25"},
                 timeout=120, retries=0)
        if r is None:
            continue
        cands: list[tuple[int, int, str, str]] = []
        for tr in BeautifulSoup(r.text, "lxml").select("table tr"):
            a = tr.select_one("a[href*='ads.php?md5=']")
            if not a:
                continue
            md5 = re.search(r"md5=([0-9a-f]{32})", a["href"])
            tds = [td.get_text(" ", strip=True) for td in tr.select("td")]
            if not _covers(query, " ".join(tds[:2])):    # title and author columns
                continue
            row = " ".join(tds).lower()
            ext = next((e for e in ("epub", "pdf", "txt") if f" {e}" in f" {row}"), "")
            size = re.search(r"(\d+(?:\.\d+)?)\s*(kb|mb|gb)", row)
            mult = {"kb": 1_000, "mb": 1_000_000, "gb": 1_000_000_000}
            nbytes = int(float(size.group(1)) * mult[size.group(2)]) if size else 10_000_000
            if md5 and ext and nbytes <= max_bytes:
                cands.append((0 if ext == "epub" else 1, nbytes, md5.group(1), ext))
        for _, _, md5, ext in sorted(cands)[:3]:
            page = _get(f"{host}/ads.php", params={"md5": md5}, timeout=60, retries=0)
            if page is None:
                continue
            # The download key is minted per page view; it is not guessable.
            m = re.search(r"get\.php\?md5=([0-9a-f]{32})&(?:amp;)?key=(\w+)", page.text)
            if not m:
                continue
            f = _get(f"{host}/get.php", params={"md5": m.group(1), "key": m.group(2)},
                     timeout=300, retries=0)
            if f is not None and len(f.content) > 20000:
                return f.content, ext
    return None


# --------------------------------------------------------------------------- #
# Open web (headless browser)
# --------------------------------------------------------------------------- #

# Aggregators that answer any query with a paywall, a login wall or an upload of
# somebody's lecture notes. Fetching them costs a request and never yields text.
_JUNK_HOST = re.compile(
    r"(scribd|researchgate|academia\.edu|yumpu|studocu|slideshare|coursehero|chegg|"
    r"quizlet|linkedin|pinterest|facebook|instagram|youtube|amazon\.|ebay\.|x\.com|"
    r"twitter\.com|annas-archive|z-lib)", re.I)

_driver = None
_driver_lock = threading.Lock()


def _browser():
    """One headless Chrome for the whole run, or None if Selenium/Chrome is missing.

    Kept alive between searches: the driver takes ~3 seconds to start and the search
    itself takes ~8, so one browser per wanted work would double the stage.
    """
    global _driver
    if _driver is not None:
        return _driver
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
    except ImportError:
        return None
    opts = Options()
    for flag in ("--headless=new", "--disable-gpu", "--window-size=1280,900",
                 "--log-level=3", "--disable-blink-features=AutomationControlled",
                 f"--user-agent={UA}"):
        opts.add_argument(flag)
    try:
        _driver = webdriver.Chrome(options=opts)
    except Exception:      # noqa: BLE001 - no Chrome, no driver, no browser route
        return None
    _driver.set_page_load_timeout(60)
    atexit.register(_close_browser)
    return _driver


def _close_browser() -> None:
    global _driver
    if _driver is not None:
        try:
            _driver.quit()
        except Exception:  # noqa: BLE001 - already gone
            pass
        _driver = None


def browser_search(query: str, limit: int = 8) -> list[dict[str, str]]:
    """Web search results, through a real browser because nothing else answers.

    Every keyless HTTP route into a search engine is either blocked or poisoned:
    ``html.duckduckgo.com`` serves a 202 challenge page after a handful of queries,
    Bing and the public SearXNG instances answer a detected scraper with results for
    an unrelated query (a search for Boulding came back with Finnish office rentals),
    and Startpage/Anubis hold everything behind a JS check. duckduckgo.com's own
    Javascript site driven by headless Chrome returns the real result list, so that
    is the route. Serialised on a lock: one browser, and ``missing_books`` is threaded.
    """
    results: list[dict[str, str]] = []
    with _driver_lock:
        drv = _browser()
        if drv is None:
            return results
        try:
            drv.get("https://duckduckgo.com/?ia=web&q=" + urllib.parse.quote_plus(query))
        except Exception:  # noqa: BLE001 - timeout or a dead driver
            _close_browser()
            return results
        # The result list is rendered client-side; poll instead of sleeping blind.
        soup = None
        for _ in range(12):
            time.sleep(1.5)
            soup = BeautifulSoup(drv.page_source, "lxml")
            if soup.select('a[data-testid="result-title-a"]'):
                break
        for a in (soup.select('a[data-testid="result-title-a"]') if soup else []):
            url = a.get("href", "")
            if not url.startswith("http") or "duckduckgo.com" in url or _JUNK_HOST.search(url):
                continue
            results.append({"title": a.get_text(" ", strip=True), "url": url})
            if len(results) >= limit:
                break
    return results


def web_fulltext(wanted: str, log=print) -> Source | None:
    """Last resort for a wanted work: find its text loose on the open web.

    Every catalogue above matches on metadata, so all of them need the work's real
    title — and the model that wrote the reading list often gives a translated or
    approximate one ("Chrysalis: informe de diseño", "HERITAGE simulation generation
    ships"). A search engine tolerates that; a catalogue does not. Reports published
    straight to a project's own site and pre-digital essays mirrored on university
    servers have no DOI and no catalogue record at all, and this is the only way in.

    The ``_covers`` gate runs against the *downloaded text*, not the search snippet:
    a paper's first page carries its real title and its author list, which is exactly
    what the approximate wanted string has to be checked against.
    """
    author, title, year = _split_wanted(wanted)
    probe = _plain(f"{author} {title}")
    if len(_words(probe)) < 2:
        return None
    seen: set[str] = set()
    best, best_url, tried = "", "", 0
    # PDFs first, both in the query and inside each result page: an HTML hit for a
    # book or a report is usually the publisher's landing page — it clears the 3.000
    # character floor with a blurb and a table of contents and nothing else.
    # Four passes, not two, and they cost nothing unless the earlier ones came back
    # with a fragment: the loop below stops as soon as something whole has landed.
    for query in (f"{probe} pdf", probe,
                  f"{probe} full text pdf", f"{probe} texto completo libro pdf"):
        for hit in sorted(browser_search(query, 8),
                          key=lambda h: not h["url"].lower().endswith(".pdf")):
            if hit["url"] in seen or tried >= 12:
                continue
            seen.add(hit["url"])
            # The abs page is the abstract plus navigation; the paper is one URL over.
            url = re.sub(r"arxiv\.org/abs/([^/?#]+?)(?:v\d+)?$", r"arxiv.org/pdf/\1",
                         hit["url"])
            body = fetch_text(url, timeout=60)
            tried += 1
            # 6.000 characters, not the 3.000 a dossier stub needs: at this point the
            # candidate is a whole work, and anything shorter is a landing page, an
            # abstract or a catalogue record dressed up as one.
            if len(body) < 6000 or not _covers(probe, f"{hit['title']} {body[:4000]}"):
                continue
            # Longest wins rather than first-past-the-post. A publisher's sample and
            # the whole book both clear the floor and both carry the real title page,
            # so taking the first hit filed one chapter of Kurz's 212 pages as the
            # book. Bounded: twelve downloads at a minute each is already the ceiling
            # of what this route may cost, and a hit this size is not a sample.
            if len(body) > len(best):
                best, best_url = body, url
            if len(best) >= 150000:
                break
        # Stop only on something plainly whole. A text that ends mid-sentence is a
        # slice — «HERITAGE…» and *Der Kollaps* both came back that way — and earns
        # the next query pass instead of ending the search.
        if tried >= 12 or len(best) >= 150000 or (len(best) >= 60000 and not _cut(best)):
            break
    if not best:
        return None
    real = _titlepage(best, title, [author] if author else [], year)
    LIBRARY.mkdir(exist_ok=True)
    dst = _library_path(real["authors"][0] if real["authors"] else author,
                  real["title"], real["year"], "txt")
    write_atomic(dst, best)
    log(f"  · web OK: {_trim(wanted, 60)} ({_size(best)})")
    return Source(title=real["title"], authors=real["authors"],
                  year=real["year"], kind="book", venue=real["venue"], url=best_url,
                  abstract=best[:2000], fulltext=best, origin="web")


# A reading list written in Spanish asks for a work that was published in German and
# catalogued in English. Word overlap cannot bridge that; asking can.
_WORK_INFO_PROMPT = """Below is a work as a reading list asked for it. \
Tell us two things: under which other titles it was published or is catalogued, and \
whether it is a book or a short text.

Format:
{{"titulos": ["Der Kollaps der Modernisierung", "The Collapse of Modernization"],
  "tipo": "libro"}}

Rules:
- "titulos": the title of THAT SAME work in the other languages it was published in,
  starting with the original language. Two at most.
- Only the same work. Another book by the same author does not count, nor one of its
  chapters, nor a similar work: if «El colapso de la modernización» is what was asked
  for, «Geld ohne Wert» is wrong even though the same person signs it.
- If you do not know the work, or it is already in its original language, or you are
  unsure, return {{"titulos": []}}. An empty list is a correct answer.
- Do not translate word for word: we want the title it was actually published under.
  If you do not know it, do not invent it.
- "tipo" is "libro" if the work was published as a book (hundreds of pages), and
  "breve" if it is a journal article, a conference paper, a report, a standalone essay
  or a chapter. If you do not know, "breve".

Requested work:
{pedido}"""


@functools.lru_cache(maxsize=256)
def _work_info(author: str, title: str, year: str) -> dict:
    """What the work is called elsewhere, and whether it is a book.

    Two answers from one FLASH call, because both change what ``fetch_book`` accepts.

    *The titles.* The reading list is written in Spanish and the copy that exists —
    on the user's disk, in Library Genesis, in archive.org — is the German original:
    «El colapso de la modernización» is *Der Kollaps der Modernisierung*, and
    ``_covers`` sees two unrelated strings. Lowering the overlap threshold until
    those match is exactly what the ``ceil`` in ``_covers`` exists to prevent, since
    at that point two Kurz books match each other on the author's name alone. So the
    gate stays and the *probe* moves: each alternative is matched at full strength, a
    bad translation therefore matches nothing rather than anything, and whatever
    lands is still checked against the document's own title page by ``_titlepage``.

    *The type.* A length floor is the only thing that separates a book from one of
    its chapters, and it cannot be applied blind: Boulding's 1966 essay is 35.000
    characters and complete, while «El colapso de la modernización» at 31.000 is
    chapter one of 212 pages. Nothing in the downloaded text distinguishes them —
    only knowing what the work is does.
    """
    out = {"titles": (), "book": False}
    if len(_words(title)) < 2:
        return out
    try:
        import llm
        got = llm.chat_json(llm.FLASH, _WORK_INFO_PROMPT.format(
            pedido=f"{author}, {title} ({year})".strip(" ,")), temperature=0.0)
    except Exception:      # noqa: BLE001 - no provider, no answer, no JSON: no variants
        return out
    if not isinstance(got, dict):
        return out
    names = []
    for name in (got.get("titulos") or [])[:2]:
        # A variant that is the title back again buys nothing; one that shares no
        # significant word with it is a real alternative and not a paraphrase.
        if (isinstance(name, str) and 6 <= len(name.strip()) <= 150
                and len(_words(name)) >= 2 and not _covers(title, name)):
            names.append(name.strip())
    return {"titles": tuple(names),
            "book": str(got.get("tipo", "")).strip().lower().startswith("libro")}


# The reference list prints whatever is in ``Source``, so a work whose only name is
# the one the reading list guessed at gets cited under a title nobody published.
_TITLEPAGE_PROMPT = """Below is the beginning of a document that was just \
downloaded: cover, header or first page. Return its bibliographic data as the \
document itself gives it.

Format:
{{"titulo": "...", "autores": ["First Last"], "anio": "1966", "editorial": "..."}}

Rules:
- Copy the document's title. Do not translate, shorten or complete it. If it wraps
  over several lines, join it into one.
- Authors as the document writes them, given name first and surname after
  ("Kenneth E. Boulding"); someone else reverses them. If the text names none,
  leave [].
- "editorial" is the publisher or the journal it appeared in, if the document says so.
- The text comes from OCR: if a word is broken («Fr ́ ed ́ eric»), write the normal
  form («Frédéric»), and if the title is all capitals because the cover sets it so,
  return it in normal upper and lower case.
- Whatever the document does not say stays empty. Do not guess and do not use what
  you know from memory.

Reference from whoever asked for it (it may be misspelled or translated; do not copy it):
{pedido}

Document:
{texto}"""


def _titlepage(body: str, title: str, authors: list[str], year: str) -> dict:
    """Real bibliographic data, read off the document's own first page.

    Every route that ends here — Library Genesis and the open web — identifies the
    work by the string the *reading list* used, and that string is a model's guess:
    a translated title («Chrysalis: informe de diseño»), an approximate one
    («HERITAGE simulation generation ships»). Filing the download under it is how a
    real text ends up in the reference list under a title nobody ever published,
    which is the one thing this pipeline is not allowed to do.

    Parsing it out of the text does not work — a PDF's first line is the author, or
    a journal header, and the title itself wraps over three lines — and arXiv PDFs
    carry no metadata title, so the title page is read by the model that is already
    in the run. Everything it returns is checked back against the text: a title whose
    words are not in the opening page, an author whose surname is not there, or a
    year the document never prints is dropped and the caller's value stands. The
    model can only *correct* the guess here, never invent past it.
    """
    out = {"title": title, "authors": authors, "year": year, "venue": ""}
    try:
        import llm
        got = llm.chat_json(llm.FLASH, _TITLEPAGE_PROMPT.format(
            pedido=f"{', '.join(authors)}, {title} ({year})", texto=body[:3000]),
            temperature=0.0)
    except Exception:      # noqa: BLE001 - no provider, no answer, no JSON: keep the guess
        return out
    if not isinstance(got, dict):
        return out
    page = body[:4000]
    if isinstance(name := got.get("titulo"), str) and _covers(name, page):
        out["title"] = name.strip()
    named = _words(page)
    real = [a.strip() for a in got.get("autores") or []
            if isinstance(a, str) and _words(a) & named]
    if real:
        out["authors"] = real
    if (when := str(got.get("anio") or "")).isdigit() and when in page:
        out["year"] = when
    if isinstance(where := got.get("editorial"), str) and where.strip() in page:
        out["venue"] = where.strip()
    return out


def _trim(text: str, n: int) -> str:
    """Cut a title for the log, saying so when it was cut."""
    text = text.strip()
    return text if len(text) <= n else text[:n].rstrip() + "…"


def _size(body: str) -> str:
    """How much text we hold, in words — the unit everything else uses.

    Downloads are uncapped, so the number is the whole extracted text, not a
    ceiling. The «recortado» flag is gone with the caps.
    """
    return f"{len(body.split()):,} words"


def _cut(body: str) -> bool:
    """True when the text stops mid-sentence: the download did not finish.

    A dropped connection, a PDF whose second half is images and a publisher's
    sample all end in the middle of a phrase. Only asked of something short enough
    to *be* a fragment — a 1,5-million character anthology whose last line is an
    index entry is not a cut download.
    """
    if not body or len(body) >= 150000:
        return False
    return body.rstrip()[-1:] not in ".!?…»\"”'’"


def _partial(body: str) -> str:
    """Log suffix for a work that landed incomplete. Says nothing when it is whole.

    Two things this catches. A text that stops in the middle of a sentence was
    cut — a dropped connection, a PDF whose second half is images. And a
    book-length work that arrives at a chapter's length is a publisher's sample:
    «El colapso de la modernización» is 212 pages, and the PDF that came off the
    open web held 31.000 characters and cleared every floor above it, because a
    sample carries the real title page.

    It only ever warns. Deciding by length alone would throw away Boulding's 1966
    essay, which is genuinely 35.000 characters long and is not a book at all.
    """
    if not body:
        return ""
    if _cut(body):
        return " — cut off midway"
    return " — excerpt?" if len(body) < 60000 else ""


def _library_path(author: str, title: str, year: str, ext: str) -> pathlib.Path:
    """``library/Autor - Título (año).ext`` — the shape ``scan_library`` parses back.

    Saving under a slug instead loses the author and the year, and the source ends up
    keyed "postone, s/f" with a reference-list entry to match.
    """
    def clean(s: str) -> str:
        return re.sub(r"[<>:\"/\\|?*]+", "", s).strip(" .-")[:70]
    stem = " - ".join(filter(None, (clean(author), clean(title) or "obra")))
    return LIBRARY / f"{stem}{f' ({year})' if year else ''}.{ext}"


def _split_wanted(wanted: str) -> tuple[str, str, str]:
    """«Postone, Time, Labor and Social Domination (1993)» -> author, title, year."""
    year = (re.search(r"\((\d{4})\)", wanted) or re.search(r"\b(1[6-9]\d{2}|20\d{2})\b", wanted))
    # «Marin, Frédéric y Beluffi, Camille, "Computing the minimal crew…", JBIS (2018)»
    # has three commas before the title. When the model quoted it, believe the quotes:
    # partitioning on the first comma otherwise hands the catalogues a query made of
    # two author names, a journal and the article title, and they answer with noise.
    if (q := re.search(r"[«\"“'‘]([^«»\"“”'‘’]{6,150})[»\"”'’]", wanted)):
        return wanted[:q.start()].strip(" ,"), q.group(1).strip(), (year.group(1) if year else "")
    body = re.sub(r"\s*\((?:\d{4})\)\s*$", "", wanted).strip()
    author, sep, title = body.partition(",")
    if not sep:                       # no comma: the whole string is the title
        return "", body, (year.group(1) if year else "")
    return author.strip(), (title.strip() or body), (year.group(1) if year else "")


def fetch_book(wanted: str, log=print) -> Source | None:
    """Actually download a work the catalogues only knew by title.

    Whatever lands is written into ``library/`` in its original format, so the next
    run reads it off disk through ``scan_library`` instead of downloading it again.

    The whole chain runs once per title, and the titles are the one the reading list
    wrote plus the same work's name in the language it was published in — the copy
    that exists is often the original, and «El colapso de la modernización» does not
    match *Der Kollaps der Modernisierung* on any word. The alternatives cost one
    FLASH call and are only asked for when the first pass came back empty, because
    ``_titles`` is a generator: the second lap is never reached for a work that the
    catalogues already answered.
    """
    author, title, year = _split_wanted(wanted)
    LIBRARY.mkdir(exist_ok=True)
    info = _work_info(author, title, year)
    # A book is not 60.000 characters long. Below that floor a candidate is one of
    # its chapters, and a chapter may not be filed — or cited — as the book. Works
    # that are legitimately short carry no floor at all.
    floor = 60000 if info["book"] else 0

    best: Source | None = None
    best_size = 0

    def _take(src: Source, size: int) -> Source | None:
        """Accept a candidate, or hold it and keep looking if it is a fragment.

        Measured 2026-08-22: archive.org answers «El colapso de la modernización» with
        *El-colapso-de-la-modernizacion-capitulo-1.pdf* — chapter one, under the exact
        title asked for. It clears every gate, and first-past-the-post files it as the
        book and never reaches the original on the next lap. Under the floor the
        candidate is remembered rather than returned, so the search goes on and the
        fullest thing seen wins.

        A text that stops mid-sentence is held for the same reason even when it clears
        the floor: *Der Kollaps der Modernisierung* came off the open web at 12.307
        words ending in the middle of a phrase, which is a slice of a 212-page book,
        and returning it there meant the catalogues on the next lap were never asked.
        Held, not rejected — if nothing whole ever lands it is still what we return.
        """
        nonlocal best, best_size
        if size > best_size:
            best, best_size = src, size
        return None if size < floor or _cut(src.fulltext) else best

    def _from_disk(alt: str) -> Source | None:
        # Already on disk: a previous (or interrupted) run downloaded it, or the
        # user dropped it in by hand. url is the path, the same signature
        # ``scan_library`` produces, so ``dedupe`` folds the two into one source
        # instead of citing twice.
        query = _plain(f"{author} {alt}")
        for p in sorted(LIBRARY.rglob("*")):
            if not p.is_file() or not _covers(query, p.stem):
                continue
            body = read_local(p, 0)
            if len(body) > 3000:
                flag = _partial(body)
                log(f"  · already in library: {_trim(wanted, 60)} ({_size(body)}{flag})")
                # ``alt``, not the wanted title: on the second lap the file on disk is
                # the German edition, and APA cites the edition that was read.
                return Source(title=alt or p.stem, authors=[author] if author else [],
                              year=year, kind="book", venue="biblioteca local", url=str(p),
                              abstract=body[:2000], fulltext=body, origin="library")
        return None

    # Every title variant against the disk before any network lap. The copy in
    # library/ often carries its original-language title («Time, Labor and
    # Social Domination» on disk, «Tiempo, trabajo…» asked for), and the laps
    # below consult the disk one language at a time — measured 2026-08-23,
    # Postone's Spanish request re-downloaded the English edition that had been
    # on disk since the day before, because lap one reached libgen first.
    for alt in dict.fromkeys((title, *info["titles"])):
        if (got := _from_disk(alt)) and (s := _take(got, len(got.fulltext))):
            return s

    for alt in (title, *info["titles"]):
        # Rebuilt rather than passed through: the callees re-split it, and the quotes
        # make «Autor, Título (año)» read back exactly as it went in.
        asked = f"{author}, «{alt}» ({year})" if author else f"«{alt}» ({year})"
        # Catalogue search engines choke on the "Autor, Título (año)" punctuation.
        query = _plain(f"{author} {alt}")

        # The user's own Calibre library, before any download. A book the user owns
        # is the edition to read: it is never held back for something fuller.
        if (got := calibre_book(asked, log=log)):
            return got

        if (hit := archive_org_book(query)):
            body, doc = hit
            creator = str(doc.get("creator") or author or "")
            pub = str(doc.get("year") or year or "")
            path = _library_path(creator, alt or str(doc.get("title", "")), pub, "txt")
            if not path.exists():   # never overwrite what library/ already has
                write_atomic(path, body)
            flag = _partial(body)
            log(f"  · archive.org OK: {_trim(wanted, 60)} ({_size(body)}{flag})")
            got = Source(title=doc.get("title") or alt,
                         authors=[creator] if creator else [],
                         year=pub, kind="book",
                         venue="Internet Archive",
                         url=f"https://archive.org/details/{doc.get('identifier','')}",
                         abstract=body[:2000], fulltext=body, origin="archive.org")
            if (s := _take(got, len(body))):
                return s

        if (file := libgen_book(query)):
            data, ext = file
            path = _library_path(author, alt, year, ext)
            # Never write over what is already in library/: the user's own scan of a
            # book lives under exactly the name this builds, and the unlink below —
            # which fires whenever the download will not extract — would then delete
            # a file this run did not create.
            if path.exists():
                path = path.with_name(f"{path.stem} [libgen]{path.suffix}")
            path.write_bytes(data)
            body = read_local(path, 0)
            if len(body) > 3000:
                flag = _partial(body)
                log(f"  · libgen OK: {_trim(wanted, 60)} ({_size(body)}{flag})")
                # Library Genesis matched on the same guessed title, so the download is
                # named after the guess too until the title page says otherwise.
                real = _titlepage(body, alt, [author] if author else [], year)
                # No url: a local path is not a citable address, and the reference list
                # prints whatever is here.
                got = Source(title=real["title"], authors=real["authors"],
                             year=real["year"], kind="book", venue=real["venue"], url="",
                             abstract=body[:2000], fulltext=body, origin="libgen")
                if (s := _take(got, len(body))):
                    return s
            else:
                path.unlink(missing_ok=True)   # unreadable format: leave no junk behind

        # Catalogues first, the open web last: a catalogue hit carries real metadata
        # for the reference list, a web hit carries only what the model called the work.
        if (got := fetch_paper(asked, log=log) or web_fulltext(asked, log=log)):
            if (s := _take(got, len(got.fulltext))):
                return s
    if best is not None and best_size < floor:
        # Rejected, not filed. A chapter in the dossier is a chapter the drafting
        # model quotes as if it were the book, under a citation to the whole work.
        log(f"  · dropped: only an excerpt of «{_trim(title, 45)}» turned up "
            f"({_size(best.fulltext)}) — put the book in library/ and run again")
        return None
    return best


def fetch_paper(wanted: str, log=print) -> Source | None:
    """A wanted work that is a journal article or a preprint, not a book.

    archive.org and Library Genesis index books, so «Smith, "Estimation of a
    genetically viable population…", Acta Astronautica (2014)» comes back empty from
    both no matter how many mirrors we try — the paper is one Crossref DOI or one
    arXiv id away. Tried last, so a real book still goes through the book catalogues
    first. The candidate goes through ``_covers`` like any other catalogue hit, and
    counts only if ``_retrieve`` actually lands the text.
    """
    author, title, year = _split_wanted(wanted)
    probe = f"{author} {title}"
    # Significant words of the title, nothing else: arXiv's API ANDs the whole string,
    # so "for a … towards … b" and the author names push the paper off its own results.
    query = " ".join(sorted(_words(title)))
    if len(query.split()) < 3:
        return None
    for hit in arxiv(query, 4) + crossref(query, 4):
        if not _covers(probe, f"{hit.title} {' '.join(hit.authors)}"):
            continue
        if _retrieve(hit, log):
            log(f"  · {hit.origin} OK: {_trim(wanted, 60)} ({_size(hit.fulltext)})")
            return hit
    return None


# --------------------------------------------------------------------------- #
# Local library
# --------------------------------------------------------------------------- #

def read_local(path: pathlib.Path, max_chars: int = 0) -> str:
    """Text of a local file. ``max_chars=0`` (the default) reads the whole thing.

    Everything that caches a work into ``library/`` reads it uncapped: writing a
    capped text to disk stores a mutilated copy, and every later run inherits the
    cut without any way of knowing it happened. The only cut that legitimately
    happens is the per-call one in ``Source.brief`` when the drafting context is
    built; the stored source keeps the whole text.
    """
    cut = max_chars or None
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            return pdf_text(path.read_bytes())[:cut]
        if ext == ".epub":
            import ebooklib
            from ebooklib import epub
            # ignore_ncx: the spine alone orders the chapters; a missing or broken
            # NCX must not turn a readable book into "" (it only costs the TOC).
            # The XHTML documents carry a real XML declaration; an XML parse
            # respects their entities and empty elements, and an HTML parser over
            # them raises XMLParsedAsHTMLWarning before mangling some of them.
            # A hand-edited chapter that has stopped being well-formed is still
            # HTML, so it falls back to the tolerant HTML parser.
            book = epub.read_epub(str(path), options={"ignore_ncx": True})
            chunks = []
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                raw = item.get_content()
                text = _soup(raw).get_text("\n")
                if text.strip():
                    chunks.append(text)
            return _clean("\n".join(chunks))[:cut]
        if ext == ".docx":
            import docx
            return _clean("\n".join(p.text for p in docx.Document(str(path)).paragraphs))[:cut]
        if ext in (".htm", ".html"):
            return _clean(BeautifulSoup(path.read_text("utf-8", errors="ignore"), "lxml")
                          .get_text("\n"))[:cut]
        if ext in (".txt", ".md", ".org", ".tex"):
            return path.read_text("utf-8", errors="ignore")[:cut]
        if ext in (".azw3", ".azw", ".mobi", ".prc", ".fb2", ".lit", ".rtf", ".pdb"):
            return ebook_convert(path)[:cut]
    except Exception:
        return ""
    return ""


def ebook_convert(path: pathlib.Path) -> str:
    """Kindle and other Calibre-only formats, through Calibre's own converter.

    Returns "" when Calibre is not installed or the file is DRM'd; the caller then
    treats it as an unreadable file and falls through to the network.
    """
    exe = shutil.which("ebook-convert") or r"C:\Program Files\Calibre2\ebook-convert.exe"
    if not pathlib.Path(exe).exists():
        return ""
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "out.txt"
        try:
            subprocess.run([exe, str(path), str(out)], timeout=600,
                           capture_output=True, check=False)
        except (OSError, subprocess.SubprocessError):
            return ""
        return _clean(out.read_text("utf-8", errors="ignore")) if out.exists() else ""


def scan_library() -> list[Source]:
    """Index whatever the user dropped in ``library/`` (epub/pdf/docx/txt/html)."""
    LIBRARY.mkdir(exist_ok=True)
    out = []
    for p in sorted(LIBRARY.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in (
                ".pdf", ".epub", ".docx", ".txt", ".md", ".htm", ".html", ".tex", ".org"):
            continue
        body = read_local(p)
        if len(body) < 500:
            continue
        stem = p.stem
        # "Autor - Titulo (2004)" is the common filename shape for these libraries.
        m = re.match(r"^(.*?)\s*[-–—]\s*(.*?)\s*(?:\((\d{4})\))?$", stem)
        author, title, year = (m.group(1), m.group(2), m.group(3) or "") if m else ("", stem, "")
        out.append(Source(title=title or stem, authors=[author] if author else [],
                          year=year, kind="local", venue="biblioteca local",
                          url=str(p), abstract=body[:2000], fulltext=body,
                          origin="library"))
    return out


# --------------------------------------------------------------------------- #
# Calibre
# --------------------------------------------------------------------------- #

# Formats read straight off disk come first: everything below EPUB needs a run of
# ebook-convert, which costs a process launch and fails outright on DRM.
_FMT_RANK = {"TXT": 0, "MD": 0, "EPUB": 1, "HTMLZ": 2, "DOCX": 3, "PDF": 4}


def calibre_rows() -> list[dict[str, str]]:
    """Every file in the Calibre library with its metadata, best format first.

    Opened read-only: the library is the user's and Calibre may well have it open.
    Calibre's metadata beats parsing a filename — the publisher and the publication
    year come out of the catalogue, so the reference-list entry is a real APA one.
    """
    db = CALIBRE / "metadata.db"
    if not db.exists():
        return []
    query = """
        select b.id, b.title, b.path, b.pubdate, d.name, d.format,
               group_concat(distinct a.name), p.name
          from books b
          join data d on d.book = b.id
          left join books_authors_link ba on ba.book = b.id
          left join authors a on a.id = ba.author
          left join books_publishers_link bp on bp.book = b.id
          left join publishers p on p.id = bp.publisher
         group by d.id
    """
    try:
        con = sqlite3.connect(f"{db.as_uri()}?mode=ro", uri=True)
        try:
            found = con.execute(query).fetchall()
        finally:
            con.close()
    except sqlite3.Error:
        return []
    rows = []
    for bid, title, path, pubdate, name, fmt, authors, publisher in found:
        # Calibre writes 0101-01-01 for "no date"; anything before print counts as unset.
        year = str(pubdate or "")[:4]
        rows.append({"book": bid, "title": title or "", "path": path or "",
                     "name": name or "", "format": fmt or "",
                     "year": year if year.isdigit() and int(year) > 1450 else "",
                     "authors": (authors or "").replace(",", ", "),
                     "publisher": publisher or ""})
    rows.sort(key=lambda r: (r["book"], _FMT_RANK.get(r["format"].upper(), 9)))
    return rows


def calibre_book(wanted: str, log=print) -> Source | None:
    """Look a missing work up in the user's Calibre library before going online.

    Matched by title and author through ``_covers`` like every other catalogue, and
    the extracted text is cached into ``library/`` so the next run reads it there
    instead of running ebook-convert again. The Calibre library itself is never
    written to.
    """
    author, title, year = _split_wanted(wanted)
    query = f"{author} {title}".strip()
    for row in calibre_rows():
        if not _covers(query, f"{row['title']} {row['authors']}"):
            continue
        path = CALIBRE / row["path"] / f"{row['name']}.{row['format'].lower()}"
        body = read_local(path, 0)
        if len(body) < 3000:      # unreadable format, or DRM: try the next file
            continue
        authors = [a.strip() for a in row["authors"].split(",") if a.strip()]
        pub = row["year"] or year
        LIBRARY.mkdir(exist_ok=True)
        # Cache the extraction so the next run skips ebook-convert — under the
        # same rule as everywhere else: nothing overwrites a library/ file.
        dst = _library_path(authors[0] if authors else author,
                            row["title"], pub, "txt")
        if not dst.exists():
            write_atomic(dst, body)
        log(f"  · Calibre OK: {_trim(wanted, 60)} ({_size(body)}{_partial(body)})")
        return Source(title=row["title"] or title, authors=authors or ([author] if author else []),
                      year=pub, kind="book", venue=row["publisher"],
                      url=str(path), abstract=body[:2000], fulltext=body,
                      origin="calibre")
    return None


# --------------------------------------------------------------------------- #
# Dossier
# --------------------------------------------------------------------------- #

def gather(queries: list[str], *, news_queries: list[str] | None = None,
           news_queries_en: list[str] | None = None,
           book_titles: list[str] | None = None,
           per_query: int = 6, workers: int = 8,
           log=print) -> list[Source]:
    """Run every connector over the query set and return a deduplicated dossier.

    Connectors hit different hosts, so they run concurrently; ``_get`` still
    serialises and paces each host individually.
    """
    tasks: list[tuple[str, Any]] = []
    for q in queries:
        tasks += [(f"academia: {q}", lambda q=q: openalex(q, per_query)),
                  (f"crossref: {q}", lambda q=q: crossref(q, per_query // 2)),
                  (f"s2: {q}", lambda q=q: semantic_scholar(q, per_query // 2)),
                  (f"doaj: {q}", lambda q=q: doaj(q, per_query // 2)),
                  (f"wikipedia: {q}", lambda q=q: wikipedia(q, "es", 1))]
    # Book catalogues answer to titles, not to topic queries, so they get the
    # concrete "Autor, Título" list the plan asked for.
    for t in (book_titles or [])[:12]:
        tasks.append((f"gutenberg: {_trim(t, 40)}", lambda t=t: gutenberg(t, 1)))
    for q in queries[:3]:
        tasks.append((f"arxiv: {q}", lambda q=q: arxiv(q, 3)))
    for q in queries[:2]:
        tasks.append((f"theory archives: {q}", lambda q=q: archive_search(q, per_site=1)))
    for q in (news_queries or queries[:3]):
        tasks += [(f"news es: {q}", lambda q=q: google_news(q, 8, "es")),
                  (f"gdelt: {q}", lambda q=q: gdelt_news(q, 5))]
    for q in (news_queries_en or []):
        tasks.append((f"news en: {q}", lambda q=q: google_news(q, 6, "en")))

    found: list[Source] = []
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(fn): label for label, fn in tasks}
        for done in futures.as_completed(jobs):
            label = jobs[done]
            try:
                got = done.result()
            except Exception as e:  # noqa: BLE001 - one dead connector must not kill the dossier
                log(f"  · {label} failed ({type(e).__name__})")
                continue
            found += got
            log(f"  · {label} → {len(got)}")
    local = scan_library()
    if local:
        log(f"  · local library: {len(local)} files")
    return assign_keys(dedupe(found + local))


def dedupe(sources: list[Source]) -> list[Source]:
    seen: set[str] = set()
    out = []
    for s in sources:
        sig = (s.doi or s.url or s.title).lower().strip()
        norm = re.sub(r"\W+", "", sig)[:120]
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(s)
    return out


def _retrieve(s: Source, log) -> bool:
    # Logged once at the end, after the size is known and the text has cleared the
    # 1.200-character gate: a route that "succeeded" with 300 characters of paywall
    # notice used to print an OK line and then quietly fall through to the next one.
    text, via = "", ""
    if s.doi:
        # Legal open access first; Sci-Hub only for what is genuinely paywalled.
        for name, getter in (("unpaywall", unpaywall_pdf), ("sci-hub", scihub_pdf)):
            pdf = getter(s.doi)
            if pdf and len(text := pdf_text(pdf)) > 1200:
                via = name
                break
            text = ""
    if not text and (m := re.search(r"arxiv\.org/abs/(\S+)", s.url)):
        # The abs page is the abstract plus navigation; the paper is one URL over.
        pdf = _get(f"https://arxiv.org/pdf/{m.group(1)}", timeout=60)
        if pdf is not None and pdf.content[:4] == b"%PDF":
            text, via = pdf_text(pdf.content), "arxiv"
    if not text and s.url and s.url.startswith("http"):
        text, via = fetch_text(s.url), "web"
    if len(text) > 1200:
        log(f"  · {via} OK: {s.key} {_trim(s.title, 55)} ({_size(text)})")
        s.fulltext = text
        if not s.abstract:
            s.abstract = text[:1500]
        return True
    return False


def enrich_fulltext(sources: list[Source], budget: int = 12, workers: int = 6,
                    log=print, keys: set[str] | None = None) -> int:
    """Fetch real full text for the most promising sources. Returns how many landed.

    Tries roughly twice the budget in candidates, since most attempts fail:
    paywalls, dead links, and Sci-Hub's JS challenge. With ``keys``, only those
    sources are tried and the order cap is dropped: that is the targeted pass
    the draft runs once the outline says which keys will actually be cited.
    """
    todo = [s for s in sources if not s.fulltext and (s.doi or s.url)]
    if keys is None:
        todo = todo[:budget * 2]
    else:
        todo = [s for s in todo if s.key in keys]
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda s: _retrieve(s, log), todo))
    got = sum(results)
    log(f"  · full text obtained for {got}/{len(todo)} sources")
    return got


def _fallback_links(title: str) -> list[dict[str, str]]:
    """Places the user can grab a book by hand when scraping is blocked."""
    q = urllib.parse.quote(title)
    return [
        {"title": "Anna's Archive", "url": f"https://annas-archive.org/search?q={q}"},
        {"title": "Library Genesis", "url": f"https://libgen.is/search.php?req={q}"},
        {"title": "Z-Library", "url": f"https://z-lib.gs/s/{q}"},
        {"title": "Marxists Internet Archive", "url": f"https://www.marxists.org/search.htm?q={q}"},
        {"title": "Google Books", "url": f"https://www.google.com/search?tbm=bks&q={q}"},
        {"title": "Internet Archive", "url": f"https://archive.org/search?query={q}"},
        {"title": "Open Library", "url": f"https://openlibrary.org/search?q={q}"},
    ]


def _plain(text: str) -> str:
    """Search-engine-safe form of a «Autor, "Título", Revista (año)» string."""
    return re.sub(r"\s+", " ", re.sub(r"[«»\"“”'‘’()\[\]<>;]+", " ", text)).strip()


def _words(text: str) -> set[str]:
    text = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    return {w for w in re.findall(r"[a-z]{4,}", text)
            if w not in ("libro", "book", "vol", "tomo", "critica", "critique")}


def _covers(wanted: str, candidate: str) -> bool:
    """Does ``candidate`` (a title plus author blob) name the work we asked for?

    Word overlap, not substring: catalogue search is loose enough to answer
    "Kurz, Geld ohne Wert" with a declassified CIA cable, and a book filed under a
    wrong title poisons the dossier with citations to something nobody wrote.
    """
    probe = _words(wanted)
    if len(probe) < 2:
        return False
    # ceil, not round: at 60% of four words, rounding down lets "Robert Kurz, Dinero
    # sin valor" and "Robert Kurz, El colapso de la modernización" match on the
    # author's two names alone, and one book silently stands in for the other.
    return len(probe & _words(candidate)) >= max(2, math.ceil(len(probe) * 0.6))


def _have_title(sources: list[Source], wanted: str) -> bool:
    """A wanted work counts as covered when we hold its full text, not just a stub.

    Matching is on shared words, not on a concatenated prefix: the wanted string is
    "Autor, Título" while a catalogue record is "Título" plus a separate author list,
    so a prefix probe never matched and every book was reported missing.
    """
    return any(s.fulltext and len(s.fulltext) > 3000
               and _covers(wanted, f"{s.title} {' '.join(s.authors)}")
               for s in sources)


def missing_books(sources: list[Source], wanted: list[str], log=print,
                  budget: int = 32, workers: int = 4) -> list[dict]:
    """Download the works the catalogues only knew by title; report what is left.

    Anything retrieved is appended to ``sources`` (and re-keyed, since a new source
    changes the ``Apellido, año`` collision counts), so the dossier can cite it.
    Whatever still fails comes back as a gap with hand-download links.

    The budget used to be 8 and the reading list routinely asks for sixteen. The tail
    was never requested even once and was still logged as «falta el texto completo»,
    which reads exactly like a work that was hunted and not found: measured
    2026-08-22, Boulding, *Chrysalis* and Marin y Beluffi were all reported missing on
    a run in which nothing ever asked for them, and all three had landed on an earlier
    one. A gap has to mean the chain failed, so the ceiling is now above any plausible
    reading list; ``workers`` is what bounds the cost.
    """
    todo: list[str] = []
    for w in (x.strip() for x in wanted if x and x.strip()):
        # "Moishe Postone, Time, Labor and Social Domination" and
        # "Postone, Time, Labor, and Social Domination (1993)" are one gap.
        if _have_title(sources, w) or any(_covers(w, prev) for prev in todo):
            continue
        todo.append(w)

    got: dict[str, Source | None] = {}
    if todo:
        log(f"[bibliography] downloading {min(len(todo), budget)} work(s) to library/…")
        with futures.ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = {pool.submit(fetch_book, w, log): w for w in todo[:budget]}
            for done in futures.as_completed(jobs):
                try:
                    got[jobs[done]] = done.result()
                except Exception as e:  # noqa: BLE001 - a dead mirror must not kill the run
                    log(f"  · download failed for {_trim(jobs[done], 50)} ({type(e).__name__})")
                    got[jobs[done]] = None
    landed = [s for s in got.values() if s]
    if landed:
        sources[:] = dedupe(sources + landed)   # in place: the caller keeps this list
        assign_keys(sources)

    gaps = []
    for w in todo:
        if got.get(w):
            continue
        gaps.append({"wanted": w, "fallbacks": _fallback_links(w)})
        log(f"  · full text missing for: {w}")
    return gaps


# Parenthetical APA: (Postone, 2006), (Postone, 2006, p. 302), (Kurz y Jappe, 2016),
# (Marx et al., 1867, pp. 12-14), plus the author's older "(Kurz, 1998: s/n)" form,
# which a humanizing rewrite sometimes reintroduces.
_CITE_YEAR = r"(?:\d{4}[a-z]?|s\.?/?f\.?|n\.?d\.?)"
CITE_RE = re.compile(r"\(([^()]{2,90}?),\s*(" + _CITE_YEAR + r")(?:\s*[:,][^()]{0,25})?\)")

# Narrative APA: "Postone (1993) ha mostrado", "Kurz y Jappe (2016) sostienen".
# These are just as citable — and just as forgeable — as the parenthetical form, so
# they have to be verified too; without this they slipped through unchecked.
NARRATIVE_RE = re.compile(
    r"\b([A-ZÁÉÍÓÚÜÑ][\wáéíóúüñ'’-]{1,}"
    r"(?:\s+(?:y|e|&|and)\s+[A-ZÁÉÍÓÚÜÑ][\wáéíóúüñ'’-]{1,}|\s+et\s+al\.)?)"
    r"\s+\((" + _CITE_YEAR + r")(?:\s*[:,][^()]{0,25})?\)")

# Words that start a sentence and would be read as a surname by NARRATIVE_RE.
_NOT_A_SURNAME = {"el", "la", "los", "las", "un", "una", "en", "de", "del", "al", "y",
                  "esta", "este", "esa", "ese", "su", "sus", "desde", "hasta", "como",
                  "cuando", "ya", "aun", "aún", "según", "durante", "hacia", "entre",
                  "anarres", "cultura", "tierra", "estado", "urss", "argentina"}


# letters NFKD does not split into base + accent, and so would simply vanish
_UNSPLIT = str.maketrans({"ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D",
                          "ß": "ss", "æ": "ae", "Æ": "Ae", "œ": "oe", "Œ": "Oe", "ı": "i"})


def _fold(s: str) -> str:
    """Case- and accent-blind: a rewrite that writes (Zizek, 2009) for the key
    «Žižek, 2009» cites the same work, and flagging it as invented sends a real
    citation to be deleted."""
    s = s.translate(_UNSPLIT)
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


_NOT_A_SURNAME_FOLDED = {_fold(w) for w in _NOT_A_SURNAME}


def verify_citations(text: str, sources: list[Source]) -> tuple[list[str], list[str]]:
    """Cross-check every in-text citation against the dossier.

    Returns (used_keys, unknown_citations). Unknown ones are very likely
    hallucinated and must be either removed or grounded before publication.
    """
    # APA groups several works in one parenthesis; check each independently.
    text = re.sub(r"\(([^()\n]*;[^()\n]*)\)",
                  lambda m: " ".join(f"({part.strip()})" for part in m.group(1).split(";")), text)
    known = {_fold(s.key): s.key for s in sources}
    def year_key(value: str) -> str:
        return "s/f" if re.fullmatch(r"s\.?/?f\.?|n\.?d\.?", value) else value

    used, unknown = [], []
    for pattern in (CITE_RE, NARRATIVE_RE):
        for m in pattern.finditer(text):
            author, year = m.group(1).strip(), m.group(2)
            if pattern is NARRATIVE_RE and _fold(author.split()[0]) in _NOT_A_SURNAME_FOLDED:
                continue
            if _fold(f"{author}, {year}") in known:
                used.append(known[_fold(f"{author}, {year}")])
                continue
            # Full names and bilingual joins may vary; the work's year may not.
            parts = re.split(r"\s+(?:y|e|and)\s+|\s*&\s*|,\s*",
                             re.sub(r"\bet\.?\s*al\.?", "", author))
            named = {_fold(p.split()[-1]) for p in parts if p.split()}
            matches = [s.key for s in sources if s.authors and s.key
                       and year_key(s.key.rsplit(", ", 1)[-1]) == year_key(year)
                       and named <= {_fold(_surname([a])) for a in s.authors}]
            hit = matches[0] if len(matches) == 1 else None
            (used.append(hit) if hit else unknown.append(m.group(0).strip()))
    return sorted(set(used)), sorted(set(unknown))


def write_atomic(path: pathlib.Path, text: str) -> None:
    """Write via a temp file and os.replace: Ctrl-C exits the process at once
    (os._exit), and a half-written cache file would read back as a finished stage."""
    path = pathlib.Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def save_dossier(sources: list[Source], path: pathlib.Path) -> None:
    write_atomic(path, json.dumps([dataclasses.asdict(s) for s in sources],
                                  ensure_ascii=False, indent=2))


def load_dossier(path: pathlib.Path) -> list[Source]:
    return [Source(**d) for d in json.loads(path.read_text(encoding="utf-8"))]


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s.lower() if c.isascii() and (c.isalnum() or c == " ")).strip()


def bibliography(sources: list[Source], used: list[str], lang: str = "es") -> str:
    """APA reference list: only cited works, alphabetical by author surname.

    The same work often enters the dossier twice — once from a catalogue with a
    DOI, once off disk or the open web as «s.f.» / «biblioteca local» — so the
    list is deduped by (author surname, title), keeping the richer record.
    """
    keep = [s for s in sources if s.key in used]
    picked: dict[tuple[str, str], Source] = {}
    for s in keep:
        k = (_norm(_surname(s.authors)), _norm(s.title))
        cur = picked.get(k)
        if cur is None or (bool(s.doi), bool(s.url)) > (bool(cur.doi), bool(cur.url)):
            picked[k] = s
    return "\n\n".join(sorted({s.citation(lang=lang) for s in picked.values()},
                              key=lambda x: unicodedata.normalize("NFKD", x).lower()))
