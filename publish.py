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
import random
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


# Cover images. Pollinations.ai generates from a URL prompt: no key, no quota,
# no account. AW_COVER=0 turns it off for a run.
_COVER_URL = os.environ.get("AW_COVER_URL",
                            "https://image.pollinations.ai/prompt/")
_DEVTO_API = "https://dev.to/api"
_DEFAULT_TAGS = "cienciaficcion,ensayo,sociologia"


def _cover_prompt(title: str, hint: str = "") -> str:
    topic = f"{title}. {hint}" if hint else title
    return (f"grandiose epic science fiction artwork, hand-drawn illustration in "
            f"classic sci-fi book cover style, vast monumental scale, dramatic "
            f"cinematic lighting, highly detailed linework and shading, depicting: "
            f"{topic}, awe-inspiring, no text, no letters")


def _cover_url(title: str, hint: str = "") -> str:
    from urllib.parse import quote
    return (f"{_COVER_URL}{quote(_cover_prompt(title, hint))}"
            f"?width=1024&height=576&nologo=true&seed={random.randrange(1 << 30)}")


def make_cover(title: str, *, hint: str = "", timeout: int = 180,
               log=print) -> bytes | None:
    """One Pollinations image as raw bytes, or None on any failure."""
    try:
        r = requests.get(_cover_url(title, hint), timeout=timeout)
        if r.status_code >= 400 or len(r.content) < 5000:
            log(f"[publicar] tapa: pollinations respondió {r.status_code}, "
                f"{len(r.content)} bytes; sigo sin tapa.")
            return None
        return r.content
    except Exception as exc:
        log(f"[publicar] tapa falló ({exc}); sigo sin tapa.")
        return None


def _attach_cover(post: dict, title: str, *, site: str, auth: tuple,
                  hint: str = "", timeout: int, log=print) -> None:
    """Upload the image to /media and set it as the post's featured image.

    A cover is decoration: any failure here logs and returns, it never
    un-publishes an article that already went through.
    """
    image = make_cover(title, hint=hint, log=log)
    if not image:
        return
    name = slugify(title) or "tapa"
    # The post already exists. An exception escaping from here would skip the
    # receipt in publish_run, and the next resume would post the article again.
    try:
        r = requests.post(
            f"{site}/wp-json/wp/v2/media", data=image, timeout=timeout, auth=auth,
            headers={"Content-Disposition": f'attachment; filename="{name}.jpg"',
                     "Content-Type": "image/jpeg"})
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


def _publish_devto(markdown: str, *, status: str, excerpt: str, hint: str = "",
                   timeout: int, log) -> dict:
    """POST the article to dev.to. Markdown goes as-is; dev.to renders its own
    H1 from ``title``, so the stripped body carries the byline and sections.
    The cover is a Pollinations URL — dev.to fetches it server-side.
    Metadata rides in YAML front matter: the API's ``cover_image`` JSON field
    is only honoured on read, never applied on create or update."""
    title, body = split_title(markdown)
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
    if os.environ.get("AW_COVER", "") != "0":
        lines.append(f"cover_image: {quote(_cover_url(title, hint))}")
    # The closing fence is what makes it front matter: without it dev.to reads no
    # title at all and answers 422 «Title can't be blank».
    lines.append("---")
    article = {"body_markdown": "\n".join(lines) + "\n\n" + body}
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
    if target() == "devto":
        hint = re.sub(r"\s+", " ", split_title(markdown)[1])[:250]
        return _publish_devto(markdown, status=status, excerpt=excerpt,
                              hint=hint, timeout=timeout, log=log)
    site = os.environ["AW_WP_URL"].rstrip("/")
    title, body = split_title(markdown)
    # The cover should show what the article is about, not just its headline:
    # the opening sentences carry the concrete subject matter.
    hint = re.sub(r"\s+", " ", body)[:250]
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
    if os.environ.get("AW_COVER", "") != "0":
        _attach_cover(post, title, site=site,
                      auth=(os.environ["AW_WP_USER"], os.environ["AW_WP_APP_PASSWORD"]),
                      hint=hint, timeout=timeout, log=log)
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
