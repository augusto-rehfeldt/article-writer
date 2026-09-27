"""Publishing, under a pen name. Two backends:

- ``devto`` — dev.to (DEV Community), free hosting with full markdown and CSS
  themes. One API key, no OAuth: Settings → Extensions → DEV Community API.
    DEVTO_API_KEY=xxxxxxxx
    AW_PUBLISH_TARGET=devto        # or leave unset: a DEVTO_API_KEY picks it
- ``wp`` — self-hosted WordPress via the core REST API with an application
  password (Users → Profile; revocable, plain Basic auth).
    AW_WP_URL=https://misitio.example
    AW_WP_USER=srsombra
    AW_WP_APP_PASSWORD=xxxx xxxx xxxx xxxx xxxx xxxx
target() decides; AW_PUBLISH_TARGET overrides the auto-detection.

Default status is ``draft``: an unattended pipeline that publishes straight to
the front page has no undo.
"""

from __future__ import annotations

import html
import json
import os
import pathlib
import re
import unicodedata

import requests

from pipeline import slugify

# Markdown emitted by ``pipeline.assemble`` is deliberately plain: an H1 title,
# H2 section headings, paragraphs, and *emphasis* in the reference list. A real
# markdown dependency would buy nothing here.
_STRONG = re.compile(r"\*\*(.+?)\*\*", re.S)
_EM = re.compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)")


def markdown_to_html(md: str) -> str:
    """Minimal converter for the shapes the pipeline actually emits."""
    out: list[str] = []
    for block in md.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        head = re.match(r"^(#{1,4})\s+(.*)", block)
        if head:
            level = min(len(head.group(1)) + 1, 6)  # article H1 becomes page H2
            out.append(f"<h{level}>{_inline(head.group(2))}</h{level}>")
            continue
        # quotations over 40 words go in a block of their own (APA_RULES)
        if block.startswith(">"):
            quoted = "\n".join(re.sub(r"^>\s?", "", ln) for ln in block.split("\n"))
            out.append(f"<blockquote><p>{_inline(quoted).replace(chr(10), '<br />' + chr(10))}</p></blockquote>")
            continue
        out.append("<p>" + _inline(block).replace("\n", "<br />\n") + "</p>")
    return "\n\n".join(out)


def _inline(text: str) -> str:
    text = html.escape(text, quote=False)
    text = _STRONG.sub(r"<strong>\1</strong>", text)
    return _EM.sub(r"<em>\1</em>", text)


def split_title(md: str) -> tuple[str, str]:
    """Pull the leading ``# Title`` out of the document; it becomes the post title."""
    lines = md.strip().split("\n")
    if lines and lines[0].startswith("# "):
        return lines[0][2:].strip(), "\n".join(lines[1:]).strip()
    return "", md.strip()


def target() -> str:
    """Which backend gets the post. A DEVTO_API_KEY alone selects dev.to."""
    return os.environ.get("AW_PUBLISH_TARGET") or (
        "devto" if os.environ.get("DEVTO_API_KEY") else "wp")


def configured() -> bool:
    if target() == "devto":
        return bool(os.environ.get("DEVTO_API_KEY"))
    return all(os.environ.get(k) for k in
               ("AW_WP_URL", "AW_WP_USER", "AW_WP_APP_PASSWORD"))


# Cover images: a real, freely licensed picture from Wikimedia Commons, chosen by
# FLASH among search hits. Measured 2026-09-27: keyless Pollinations (and g4f's
# image models, which route to it) silently serve the small "sana" model whatever
# model is asked for, which is why the old generated covers looked cheap. Commons
# URLs are public and stable, so dev.to can fetch them server-side. A cover is
# decoration: no good match means no cover. AW_COVER=0 turns it off for a run.
_COMMONS_API = os.environ.get("AW_COVER_API", "https://commons.wikimedia.org/w/api.php")
_UA = {"User-Agent": "article-writer/1.0 (https://github.com/augusto-rehfeldt/article-writer)"}
_DEVTO_API = "https://dev.to/api"
_DEFAULT_TAGS = "cienciaficcion,ensayo,sociologia"

_COVER_QUERIES = """Find a cover image on Wikimedia Commons for this essay.

Title: {title}
Opening: {hint}

Give 4 English search queries for concrete, photographable or painted subjects the
essay is about (objects, places, scenes, historical artworks, space art), not
abstract concepts. Most specific first. Answer JSON: {{"queries": ["...", ...]}}"""

_COVER_PICK = """Pick the best cover image for this essay among Wikimedia Commons files.

Title: {title}
Opening: {hint}

Candidates:
{candidates}

A good cover shows the essay's subject, is striking at banner size, and is not a
diagram, map, chart, logo, document scan, AI-generated or stock-looking
illustration, or an unrelated thing that merely
shares a word with the query. A plain but on-topic photograph beats no cover.
Answer JSON: {{"pick": <number>}}, or {{"pick": -1}} only if every candidate
is off-topic or unusable."""


def _clean(html_text: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", html_text or "")).split())


def _commons(query: str, timeout: int) -> list[dict]:
    """Landscape bitmap files at least 1200px wide, with their credit fields."""
    r = requests.get(_COMMONS_API, headers=_UA, timeout=timeout, params=dict(
        action="query", format="json", generator="search", gsrnamespace=6,
        gsrsearch=f"filetype:bitmap {query}", gsrlimit=10, prop="imageinfo",
        iiprop="url|size|mime|extmetadata", iiurlwidth=1600))
    r.raise_for_status()
    out = []
    for page in (r.json().get("query") or {}).get("pages", {}).values():
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata") or {}
        w, h = info.get("width", 0), info.get("height", 1)
        if w < 1200 or not 1.2 <= w / h <= 2.4 or info.get("mime") not in ("image/jpeg", "image/png"):
            continue
        # Commons hosts generated images too; replacing generated covers with one defeats the point.
        tags = " ".join(str((meta.get(k) or {}).get("value", ""))
                        for k in ("Categories", "ImageDescription")).lower()
        if re.search(r"ai[- ]generated|generated by (?:ai|dall|midjourney|stable diffusion|chatgpt)"
                     r"|\bdall-?e\b|midjourney|stable diffusion|pd-algorithm", tags):
            continue
        out.append({"title": page["title"].removeprefix("File:").rsplit(".", 1)[0],
                    "url": (info.get("thumburl") or info["url"]).split("?")[0],  # drop utm_*
                    "page": info.get("descriptionurl", ""),
                    "description": _clean((meta.get("ImageDescription") or {}).get("value", ""))[:200],
                    "artist": _clean((meta.get("Artist") or {}).get("value", ""))[:80],
                    "license": (meta.get("LicenseShortName") or {}).get("value", "")})
    return out


def find_cover(title: str, hint: str = "", *, timeout: int = 30, log=print) -> dict | None:
    """The best-fitting Commons image for the article, or None."""
    import llm
    try:
        queries = llm.chat_json(llm.FLASH, _COVER_QUERIES.format(title=title, hint=hint),
                                temperature=0.3).get("queries") or [title]
        seen, cands = set(), []
        for q in [str(q) for q in queries][:4]:
            for c in _commons(q, timeout):
                if c["url"] not in seen:
                    seen.add(c["url"])
                    cands.append(c)
        if not cands:
            log("[publicar] tapa: Commons no devolvió imágenes útiles; sigo sin tapa.")
            return None
        cands = cands[:24]
        listing = "\n".join(f"{i}. {c['title']} — {c['description']}" for i, c in enumerate(cands))
        pick = int(llm.chat_json(llm.FLASH, _COVER_PICK.format(
            title=title, hint=hint, candidates=listing), temperature=0.1).get("pick", -1))
    except Exception as exc:  # noqa: BLE001 - decoration must not stop publishing
        log(f"[publicar] tapa falló ({exc}); sigo sin tapa.")
        return None
    if not 0 <= pick < len(cands):
        log("[publicar] tapa: ninguna imagen encaja; sigo sin tapa.")
        return None
    log(f"[publicar] tapa: {cands[pick]['title']} ({cands[pick]['license']})")
    return cands[pick]


def credit(cover: dict) -> str:
    """Attribution line; CC BY/BY-SA covers require it, public domain gets it anyway."""
    by = f", {cover['artist']}" if cover.get("artist") else ""
    lic = f", {cover['license']}" if cover.get("license") else ""
    return f"*Cover image: [{cover['title']}]({cover['page']}){by}{lic}, via Wikimedia Commons.*"


def _hint(body: str) -> str:
    """Opening prose, past the byline and headings."""
    paras = [p for p in body.split("\n\n") if p.strip() and not p.lstrip().startswith("#")]
    return re.sub(r"\s+", " ", " ".join(paras[1:4] or paras))[:400]


def make_cover(cover: dict, *, timeout: int = 60, log=print) -> bytes | None:
    """The chosen image as raw bytes, or None on any failure."""
    try:
        r = requests.get(cover["url"], headers=_UA, timeout=timeout)
        r.raise_for_status()
        return r.content
    except Exception as exc:  # noqa: BLE001
        log(f"[publicar] tapa no se pudo bajar ({exc}); sigo sin tapa.")
        return None


def _attach_cover(post: dict, title: str, cover: dict, *, site: str, auth: tuple,
                  timeout: int, log=print) -> None:
    """Upload the image to /media and set it as the post's featured image.

    A cover is decoration: any failure here logs and returns, it never
    un-publishes an article that already went through.
    """
    image = make_cover(cover, log=log)
    if not image:
        return
    name = slugify(title) or "tapa"
    png = cover["url"].lower().endswith(".png")
    ext, mime = ("png", "image/png") if png else ("jpg", "image/jpeg")
    # The post already exists. An exception escaping from here would skip the
    # receipt in publish_run, and the next resume would post the article again.
    try:
        r = requests.post(
            f"{site}/wp-json/wp/v2/media", data=image, timeout=timeout, auth=auth,
            headers={"Content-Disposition": f'attachment; filename="{name}.{ext}"',
                     "Content-Type": mime})
        if r.status_code >= 400:
            log(f"[publicar] media subió mal ({r.status_code}): {r.text[:200]}")
            return
        media_id = r.json().get("id")
        r = requests.post(f"{site}/wp-json/wp/v2/posts/{post['id']}",
                          json={"featured_media": media_id},
                          timeout=timeout, auth=auth)
    except Exception as exc:  # noqa: BLE001 - decoration must not cost the receipt
        log(f"[publicar] tapa falló después de publicar ({exc}); sigo sin tapa.")
        return
    if r.status_code >= 400:
        log(f"[publicar] tapa no quedó asignada ({r.status_code})")
    else:
        log("[publicar] tapa asignada como imagen destacada.")


def _created(r, log) -> dict:
    """The created post's JSON. The post exists once the status is 2xx: a body
    that will not parse must still yield a receipt, or a resume posts it again."""
    try:
        post = r.json()
    except ValueError:
        post = None
    if not isinstance(post, dict):
        # Only 201 means created. A WordPress on plain permalinks answers the
        # REST route with its home page and a 200: nothing was posted, and a
        # receipt would mark the run published forever.
        if r.status_code != 201:
            raise RuntimeError(f"HTTP {r.status_code} sin JSON: el destino no creó el artículo")
        log("[publicar] creado (HTTP 201) pero la respuesta no es JSON; guardo recibo igual.")
        post = {"id": None, "link": None, "status_code": r.status_code}
    return post


def _devto_front(title: str, *, status: str, excerpt: str, cover: dict | None) -> str:
    """YAML front matter. The API's ``cover_image`` JSON field is only honoured
    on read, never applied on create or update, so metadata rides in the body."""
    # dev.to tags are ASCII alphanumerics only: «sociología» is a 422.
    tags = [t for t in (re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", t).lower())
                        for t in os.environ.get("AW_TAGS", _DEFAULT_TAGS).split(",")) if t][:4]
    # Front matter is YAML: an unquoted «El valor: una sombra» is a mapping error
    # and the whole post is refused. A JSON string is a valid YAML double-quoted one.
    quote = lambda s: json.dumps(s, ensure_ascii=False)  # noqa: E731
    lines = ["---", f"title: {quote(title or 'Sin título')}",
             f"published: {'true' if status == 'publish' else 'false'}",
             "tags: " + ",".join(tags)]
    if excerpt:
        lines.append(f"description: {quote(excerpt[:140])}")
    if cover:
        lines.append(f"cover_image: {quote(cover['url'])}")
    # The closing fence is what makes it front matter: without it dev.to reads no
    # title at all and answers 422 «Title can't be blank».
    lines.append("---")
    return "\n".join(lines)


def _publish_devto(markdown: str, *, status: str, excerpt: str, cover: dict | None,
                   timeout: int, log) -> dict:
    """POST the article to dev.to. Markdown goes as-is; dev.to renders its own
    H1 from ``title``, so the stripped body carries the byline and sections.
    The cover is a Commons URL — dev.to fetches it server-side."""
    title, body = split_title(markdown)
    front = _devto_front(title, status=status, excerpt=excerpt, cover=cover)
    article = {"body_markdown": front + "\n\n" + body}
    r = requests.post(f"{_DEVTO_API}/articles", json={"article": article},
                      headers={"api-key": os.environ["DEVTO_API_KEY"]},
                      timeout=timeout)
    if r.status_code >= 400:
        log(f"[publicar] dev.to respondió {r.status_code}: {r.text[:300]}")
        r.raise_for_status()
    post = _created(r, log)
    log(f"[publicar] {status}: {post.get('url') or post.get('id')}")
    return post


def publish(markdown: str, *, status: str = "draft", excerpt: str = "",
            timeout: int = 90, log=print) -> dict | None:
    """POST one article. Returns the created post's JSON, or None if not configured.

    ``status`` is ``draft`` or ``publish``. Anything else is rejected rather than
    handed to WordPress, which would silently coerce it.
    """
    if status not in ("draft", "publish", "pending", "private"):
        raise ValueError(f"estado inválido para WordPress: {status!r}")
    if not configured():
        log("[publicar] falta la configuración del destino (AW_PUBLISH_TARGET / "
            "DEVTO_API_KEY o AW_WP_*); no subo nada.")
        return None
    title, body = split_title(markdown)
    cover = None
    if os.environ.get("AW_COVER", "") != "0":
        # The cover should show what the article is about, not just its headline:
        # the opening paragraphs carry the concrete subject matter.
        cover = find_cover(title, _hint(body), log=log)
    if cover:
        body = f"{body}\n\n{credit(cover)}"
        markdown = f"# {title}\n\n{body}"
    if target() == "devto":
        return _publish_devto(markdown, status=status, excerpt=excerpt,
                              cover=cover, timeout=timeout, log=log)
    site = os.environ["AW_WP_URL"].rstrip("/")
    payload = {"title": title or "Sin título", "content": markdown_to_html(body),
               "status": status}
    if excerpt:
        payload["excerpt"] = excerpt[:600]
    r = requests.post(
        f"{site}/wp-json/wp/v2/posts", json=payload, timeout=timeout,
        auth=(os.environ["AW_WP_USER"], os.environ["AW_WP_APP_PASSWORD"]))
    if r.status_code >= 400:
        log(f"[publicar] WordPress respondió {r.status_code}: {r.text[:300]}")
        r.raise_for_status()
    post = _created(r, log)
    log(f"[publicar] {status}: {post.get('link') or post.get('id')}")
    if cover:
        _attach_cover(post, title, cover, site=site,
                      auth=(os.environ["AW_WP_USER"], os.environ["AW_WP_APP_PASSWORD"]),
                      timeout=timeout, log=log)
    return post


def publish_run(run_dir: pathlib.Path, *, status: str = "draft", log=print) -> dict | None:
    """Publish ``05_final.md`` from a run folder, once.

    The receipt in ``09_publicado.json`` is what keeps a resumed or looping run
    from posting the same article twice.
    """
    receipt = run_dir / "09_publicado.json"
    if receipt.exists():
        log(f"[publicar] ya estaba publicado: {run_dir.name}")
        return None
    final = run_dir / "05_final.md"
    if not final.exists():
        log(f"[publicar] no hay 05_final.md en {run_dir}")
        return None
    post = publish(final.read_text(encoding="utf-8"), status=status, log=log)
    if post:
        receipt.write_text(json.dumps(
            {"id": post.get("id"), "link": post.get("link") or post.get("url"),
             "status": status},
            ensure_ascii=False, indent=2), encoding="utf-8")
    return post


_CREDIT_RE = re.compile(r"\n*\*Cover image: .*?via Wikimedia Commons\.\*\s*$", re.S)


def replace_covers(output: pathlib.Path, *, dry_run: bool = False, timeout: int = 90,
                   log=print) -> int:
    """Swap the cover of every article already on dev.to for a Commons pick.

    Works from the live body_markdown, so edits made on dev.to survive; only the
    front matter's cover_image and the credit line are rewritten. Returns how
    many posts changed.
    """
    if target() != "devto" or not configured():
        raise RuntimeError("replace_covers only speaks dev.to (DEVTO_API_KEY)")
    head = {"api-key": os.environ["DEVTO_API_KEY"]}
    mine = {}
    for page in range(1, 20):
        r = requests.get(f"{_DEVTO_API}/articles/me/all", headers=head, timeout=timeout,
                         params={"per_page": 1000, "page": page})
        r.raise_for_status()
        mine.update({a["id"]: a for a in r.json()})
        if len(r.json()) < 1000:
            break
    changed = 0
    for receipt in sorted(output.glob("*/09_publicado.json")):
        pid = json.loads(receipt.read_text(encoding="utf-8")).get("id")
        art = mine.get(pid)
        if not art:
            log(f"[tapas] {receipt.parent.name}: el post {pid} no está en dev.to; salto.")
            continue
        md = art["body_markdown"]
        m = re.match(r"---\n(.*?)\n---\n?", md, re.S)
        if not m:  # rewriting it would have to invent title/published lines
            log(f"[tapas] {art['title']}: sin front matter; salto.")
            continue
        front = m.group(1)
        if "wikimedia.org" in front:  # already replaced
            continue
        body = _CREDIT_RE.sub("", md[m.end():]).rstrip()
        cover = find_cover(art["title"], _hint(body), log=log)
        generated = "pollinations.ai" in front
        if not cover and not generated:
            continue
        # No fitting photo still beats the old generated cover: drop it.
        log(f"[tapas] {art['title']}\n        → {cover['url'] if cover else 'sin tapa'}")
        if dry_run:
            continue
        front = "\n".join(ln for ln in front.split("\n") if not ln.startswith("cover_image:"))
        if cover:
            front += f"\ncover_image: {json.dumps(cover['url'], ensure_ascii=False)}"
        new = f"---\n{front.strip()}\n---\n\n{body.lstrip()}"
        if cover:
            new += f"\n\n{credit(cover)}"
        r = requests.put(f"{_DEVTO_API}/articles/{pid}", headers=head, timeout=timeout,
                         json={"article": {"body_markdown": new}})
        if r.status_code >= 400:
            log(f"[tapas] dev.to respondió {r.status_code}: {r.text[:200]}")
            continue
        changed += 1
    return changed


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv(pathlib.Path(__file__).with_name(".env"))
    if "--replace-covers" not in sys.argv:
        sys.exit("uso: python publish.py --replace-covers [--dry-run]")
    n = replace_covers(pathlib.Path(__file__).with_name("output"),
                       dry_run="--dry-run" in sys.argv)
    print(f"[tapas] {n} tapas reemplazadas.")
