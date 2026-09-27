"""Add the author's university work to the style corpus.

``D:/facultad`` holds ~2200 files, most of which are *readings by other people*
(PDFs) or date-stamped lecture notes. Only the author's own argumentative writing
belongs in a style corpus, so this filters in three passes:

  1. Extension — the author writes in Word; the readings are PDFs.
  2. Filename — "Antropología 24-9.docx" is a dated lecture note, not an essay.
  3. Content — FLASH reads a sample and says whether it is the author arguing,
     someone else's text pasted in, notes, or a transcript.

Accepted files land in ``corpus_facultad/`` and are picked up automatically by
``style.articles()``.

They are a STYLE reference only. Nothing here ever enters the bibliography: the
dossier is built by ``research.gather()``, which reads ``library/`` and the online
connectors, never ``corpus_facultad/``. Do not wire the two together — the author's
old coursework is not a citable source.

    python ingest_facultad.py [ruta] [--limit N] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).parent
load_dotenv(ROOT / ".env")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import llm  # noqa: E402
import research  # noqa: E402

OUT = ROOT / "corpus_facultad"
DEFAULT_ROOT = pathlib.Path("D:/facultad")

# "Antropología 24-9.docx", "Alemán (nivel II) 13-10.docx" — lecture notes by date.
DATED_NOTES = re.compile(r"\b\d{1,2}-\d{1,2}\s*(?:\(\d+\))?$")
MIN_WORDS = 600

CLASSIFY = """Este es un fragmento de un archivo de la carpeta de facultad de un estudiante \
de sociología (Augusto Rehfeldt). Necesito saber si el texto lo ESCRIBIÓ ÉL como \
argumentación propia, o si es otra cosa.

Categorías de autoría:
- "propia": lo escribió él argumentando (monografía, parcial domiciliario, ensayo, \
informe de investigación, reseña crítica, trabajo práctico).
- "ajena": es un texto de otro autor (capítulo de libro, paper, programa de la materia, \
consigna, material de cátedra) guardado en su carpeta.
- "apuntes": notas de clase, esquemas, listas, fórmulas, resúmenes de lecturas ajenas \
sin voz propia.
- "transcripcion": desgrabación de entrevistas o clases.

Devolvé JSON: {{"autoria": "...", "genero": "monografia|parcial|ensayo|informe|resena|\
tp|programa|apuntes|otro", "confianza": 0.0-1.0, "motivo": "una frase"}}

=== FRAGMENTO ===
{sample}"""

KEEP_GENRES = {"monografia", "parcial", "ensayo", "informe", "resena", "tp"}


def candidates(root: pathlib.Path) -> list[pathlib.Path]:
    files = [p for p in root.rglob("*")
             if p.suffix.lower() in (".docx", ".rtf", ".txt", ".odt")
             and not p.name.startswith("~$")]
    return [p for p in files if not DATED_NOTES.search(p.stem)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=str(DEFAULT_ROOT))
    ap.add_argument("--limit", type=int, default=0, help="classify only N files")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--min-words", type=int, default=MIN_WORDS)
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    if not root.exists():
        print(f"{root} does not exist")
        return 1
    OUT.mkdir(exist_ok=True)

    cands = candidates(root)
    print(f"[facultad] {len(cands)} candidate files in {root}")
    long_enough = []
    for p in cands:
        text = research.read_local(p)
        if len(text.split()) >= args.min_words:
            long_enough.append((p, text))
    print(f"[facultad] {len(long_enough)} with >= {args.min_words} words")
    if args.limit:
        long_enough = long_enough[:args.limit]

    # Classifying is the only paid step here, so verdicts are cached by path+mtime.
    cache_path = OUT / "_verdicts.json"
    cache: dict = json.loads(cache_path.read_text(encoding="utf-8")) \
        if cache_path.exists() else {}

    manifest, kept_words = [], 0
    for p, text in long_enough:
        stamp = f"{p}|{int(p.stat().st_mtime)}"
        verdict = cache.get(stamp)
        if verdict is None:
            try:
                verdict = llm.chat_json(llm.FLASH, CLASSIFY.format(sample=text[:4000]),
                                        temperature=0.1)
            except Exception as e:  # noqa: BLE001
                print(f"  ? {p.name[:60]}: classification failed ({type(e).__name__})")
                continue
            cache[stamp] = verdict
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        words = len(text.split())
        autoria = str(verdict.get("autoria") or "?")
        genero = str(verdict.get("genero") or "?")
        try:
            confianza = float(verdict.get("confianza", 0))
        except (TypeError, ValueError):
            confianza = 0.0
        keep = autoria == "propia" and genero in KEEP_GENRES and confianza >= 0.6
        print(f"  {'+' if keep else '-'} {words:>6}w  {autoria:<13} {genero:<11} {p.name[:52]}")
        if not keep or args.dry_run:
            continue
        verdict = {"autoria": autoria, "genero": genero, "confianza": confianza}
        slug = re.sub(r"[^\w\-]+", "-", p.stem, flags=re.U).strip("-").lower()[:70]
        header = f"# {p.stem}\n\nOrigen: {p}\nGénero: {genero}\n\n"
        (OUT / f"{slug}.md").write_text(header + text, encoding="utf-8")
        manifest.append({"file": str(p), "slug": slug, "words": words, **verdict})
        kept_words += words

    if not args.dry_run:
        (OUT / "_index.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[facultad] accepted {len(manifest)} texts, {kept_words} words -> {OUT}")
    print("[facultad] rebuild the style with: python main.py --refresh-style")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
