"""Download the reference corpus (Revista La Cueva articles) used to model the author's style.

WordPress.com exposes a public REST API for hosted blogs, which is far more reliable
than scraping rendered HTML: it returns the post body as stored, without theme markup.
"""

import html
import json
import pathlib
import re
import sys
import urllib.request

SITE = "revistalacueva.wordpress.com"
API = f"https://public-api.wordpress.com/rest/v1.1/sites/{SITE}/posts/slug:"

SLUGS = [
    "la-wertkritik-y-el-socialismo-real-en-la-union-sovietica",
    "el-proyecto-cybersyn-cibernetica-y-socialismo-en-america-latina",
    "de-lo-que-es-el-fetichismo-de-la-mercancia-y-sobre-si-podemos-librarnos-de-el-anselm-jappe",
    "wollt-ihr-den-totalen-trieb-la-jouissance-lacaniana-entre-el-objet-petit-a-y-la-pulsion-de-muerte",
    "de-la-revolucion-rusa-las-conspiraciones-el-antisemitismo-y-la-literatura-argentina",
    "de-la-pasion-de-lo-real",
    "conceptos-claves-del-marxismo-4-la-economia-clasica-marxista-ii",
    "de-la-tendencia-decreciente-de-la-tasa-de-ganancia",
    "conceptos-claves-del-marxismo-2-la-filosofia-clasica-marxista-ii",
    "conceptos-claves-del-marxismo-2-la-economia-politica-i",
    "conceptos-claves-del-marxismo-1-k-marx",
    "los-logros-mas-importantes-de-la-historia-del-socialismo",
    "bioshock-infinite-un-videojuego-atravesado-por-la-lucha-de-clases",
    "populismo-vs-republica-desmontando-mitos-y-mentiras",
    "de-estatuas-paises-vecinos-y-odio-neoliberal",
    "el-chapo-guzman-y-el-narco-estado-mexicano-que-hay-en-comun",
]

OUT = pathlib.Path(__file__).parent / "corpus"


def to_text(raw_html: str) -> str:
    """Strip markup but keep paragraph breaks and footnote/quote structure."""
    s = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", "", raw_html)
    s = re.sub(r"(?i)<br\s*/?>", "\n", s)
    s = re.sub(r"(?i)</(p|div|li|h[1-6]|blockquote)>", "\n\n", s)
    s = re.sub(r"(?i)<li[^>]*>", "- ", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t ]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return "\n".join(line.strip() for line in s.split("\n")).strip()


def main() -> int:
    OUT.mkdir(exist_ok=True)
    index = []
    for slug in SLUGS:
        try:
            with urllib.request.urlopen(API + slug, timeout=45) as r:
                post = json.load(r)
        except Exception as e:  # noqa: BLE001 - one bad slug must not kill the run
            print(f"FAIL {slug}: {e}", file=sys.stderr)
            continue
        body = to_text(post.get("content", ""))
        title = html.unescape(post.get("title", slug))
        text = f"# {title}\n\nURL: {post.get('URL')}\nFecha: {post.get('date')}\n\n{body}\n"
        (OUT / f"{slug}.md").write_text(text, encoding="utf-8")
        index.append({"slug": slug, "title": title, "url": post.get("URL"),
                      "date": post.get("date"), "words": len(body.split())})
        print(f"OK {len(body.split()):>5}w  {title}")
    (OUT / "_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(index)}/{len(SLUGS)} articles -> {OUT}")
    return 0 if index else 1


if __name__ == "__main__":
    raise SystemExit(main())
