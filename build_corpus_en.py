"""Build the English reference corpus (corpus_en/).

English mode (--english) needs a calibration target the way the author's own
articles are the target in Spanish. There is no English corpus of "his" voice,
so this assembles the next best thing: HUMAN-written English academic prose
from three keyless sources,

  arXiv      recent papers in society-facing categories (the modern paper register)
  Gutenberg  public-domain social-theory classics (the long-essay register)
  library/   English works the runs already downloaded and cached on disk

Everything lands chunked into ~3.500-word .md files, the same window size the
Spanish fingerprint measures, so style.corpus_ranges(lang="en") compares like
with like.

    python build_corpus_en.py            # build/refresh corpus_en/
    python build_corpus_en.py --guide    # + EN fingerprint and PRO style guide
"""

from __future__ import annotations

import re
import sys

import llm
import research
import style

OUT = style.CORPUS_EN

ARXIV_QUERIES = [
    "platform labor society algorithmic management",
    "artificial intelligence society ethics policy",
    "social media democracy misinformation",
    "climate society adaptation economics",
]
GUTENBERG_TITLES = [
    "Theory of the Leisure Class",
    "The Economic Consequences of the Peace",
    "The Theory of the Moral Sentiments",
    "Evolution of Modern Medicine",
    "The Mind in the Making",
]

_EN_WORD = re.compile(r"\b(?:the|of|and|to|in|that|is|was|for|with)\b", re.I)


def is_english(text: str) -> bool:
    """Cheap stopword ratio over a sample; Spanish shares too little lexicon
    with these ten English function words to clear it (measured ~26% for real
    English prose, single digits for Spanish)."""
    sample = text[:12000]
    words = re.findall(r"[a-záéíóúñü]+", sample.lower())
    if len(words) < 200:
        return False
    hits = len(_EN_WORD.findall(sample))
    return hits / max(len(words), 1) > 0.18


def _strip_gutenberg_boilerplate(body: str) -> str:
    m = re.search(r"\*+ ?START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^*]*\*+\s*(.*)", body, re.S | re.I)
    body = m.group(1) if m else body
    m = re.search(r"(.*)\*+ ?END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", body, re.S | re.I)
    return (m.group(1) if m else body).strip()


def _slug(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:70]
    return s or "sin-titulo"


def _write_chunks(slug: str, body: str, seen: set[str], log=print,
                  chunk_words: int = 3500, max_parts: int = 6) -> int:
    """Split into paragraph-boundary chunks so windows measure like-for-like."""
    # pdf_text marks pages with «[p. N]»; drop the markers, keep the text —
    # they open most paragraphs in a PDF and would filter everything out.
    body = re.sub(r"\[p+\.\s*\d+\]", "", body)
    paras = [p.strip() for p in body.split("\n\n")
             if len(p.split()) > 1 and not p.strip().startswith("http")]
    n = written = 0
    buf: list[str] = []
    count = 0

    def flush():
        nonlocal buf, count, written, n
        if count < 800:                      # degenerate fragment, not prose
            buf, count = [], 0
            return
        name = f"{slug}__p{n}.md" if n else f"{slug}.md"
        if name in seen:
            name = f"{slug}__x{n}{written}.md"
        (OUT / name).write_text("\n\n".join(buf), encoding="utf-8")
        seen.add(name)
        written += 1
        n += 1
        buf, count = [], 0

    for p in paras:
        w = len(p.split())
        if w > chunk_words:                  # a wall of text: hard-split it
            p = "\n\n".join(" ".join(p.split()[i:i + 300])
                            for i in range(0, len(p.split()), 300))
        buf.append(p)
        count += w
        if count >= chunk_words:
            flush()
            if n >= max_parts:
                break
    if count >= 800 and n < max_parts:
        flush()
    return written


def main() -> None:
    OUT.mkdir(exist_ok=True)
    existing = {p.name for p in OUT.glob("*.md")}
    seen: set[str] = set()
    total = 0

    print("[corpus_en] arXiv (papers recientes, registro académico actual)…")
    for q in ARXIV_QUERIES:
        for s in research.arxiv(q, limit=5):
            slug = f"arxiv-{_slug(s.title)}"
            if any(x.startswith(slug) for x in existing | seen):
                continue
            if research._retrieve(s, print) and is_english(s.fulltext):
                total += _write_chunks(slug, s.fulltext, seen)

    print("[corpus_en] Gutenberg (clásicos de teoría social, dominio público)…")
    for title in GUTENBERG_TITLES:
        slug = f"gutenberg-{_slug(title)}"
        if any(x.startswith(slug) for x in existing | seen):
            continue
        for s in research.gutenberg(title, limit=2):
            body = _strip_gutenberg_boilerplate(s.fulltext or "")
            # gutendex matches loosely: verify the hit IS the work asked for,
            # else a same-topic book sneaks in under its name.
            if len(body.split()) > 20000 and is_english(body) \
                    and research._covers(title, s.abstract[:1500] + " " + body[:4000]):
                total += _write_chunks(slug, body, seen, max_parts=4)
                break

    print("[corpus_en] library/ (obras en inglés ya descargadas por los runs)…")
    try:
        lib = [s for s in research.scan_library()
               if s.fulltext and is_english(s.fulltext)]
        print(f"  · {len(lib)} obras en inglés en library/")
        for s in lib:
            slug = f"library-{_slug(s.title)}"
            if any(x.startswith(slug) for x in existing | seen):
                continue
            total += _write_chunks(slug, s.fulltext, seen, max_parts=3)
    except Exception as e:  # noqa: BLE001 - an empty library must not kill the build
        print(f"  · library/ no disponible ({type(e).__name__})")

    print(f"[corpus_en] {total} archivos nuevos en {OUT}/ "
          f"({len(list(OUT.glob('*.md')))} en total)")
    if "--guide" in sys.argv:
        style.articles_en.cache_clear()
        style.corpus_fingerprint(refresh=True, lang="en")
        print(f"[corpus_en] huella inglesa: {style.FINGERPRINT_EN.name}")
        try:
            style.build_guide(refresh=True, lang="en")
            print(f"[corpus_en] guía en inglés: {style.GUIDE_EN.name}")
        except Exception as e:  # noqa: BLE001 - providers down: fingerprint still lands
            print(f"[corpus_en] guía no pudo generarse ({type(e).__name__}: {e}); "
                  "corré de nuevo con --guide cuando haya proveedor")


if __name__ == "__main__":
    main()
