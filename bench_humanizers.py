"""Compare third-party humanizers against ours, on the same text.

Five candidates, all scored with the project's own gate (local_score + the PRO/FLASH
judges) so the numbers sit next to the calibration table in CLAUDE.md:

  corpus    the author's real prose            (human baseline)
  crudo     04_draft.md                        (plain AI, pre-humanize)
  nuestro   04_draft_humanizado.md             (humanize._rewrite)
  texthum   texthumanize(crudo)                (pip, offline, 25 langs)
  ida_y_v   crudo translated es->en->es by FLASH (lynote-ai's "translation chaining")

Citations are counted before and after: a humanizer that drops or mangles them is
disqualified no matter what it scores.

    python bench_humanizers.py [--judges]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

import humanize
import llm
import research

RUN = Path("output/20260823-la-mision-como-gran-otro-naves-generacionales-por-fuera-de-l")
CORPUS = Path("corpus/de-la-tendencia-decreciente-de-la-tasa-de-ganancia.md")
SPAN = 6000  # one judge window, so every candidate is judged the same way


def excerpt(path: Path, n: int = SPAN) -> str:
    body = path.read_text(encoding="utf-8")
    body = re.sub(r"^---.*?---\s*", "", body, flags=re.S)  # corpus front-matter
    return body[:n]


def cites(text: str) -> set[str]:
    out = {f"{a.strip()} {y}" for a, y in research.CITE_RE.findall(text)}
    out |= {f"{m.group(1).strip()} {m.group(2)}" for m in research.NARRATIVE_RE.finditer(text)}
    return out


def texthum(text: str) -> str:
    from texthumanize import humanize as th
    return th(text, lang="es", profile="academic", intensity=70).text


ROUNDTRIP = ("Traducí al {lang} el siguiente texto. Traducción completa y fiel, sin "
             "comentarios, sin resumir, sin agregar nada. Devolvé sólo la traducción.\n\n{t}")


def roundtrip(text: str) -> str:
    en = llm.chat(llm.FLASH, ROUNDTRIP.format(lang="inglés", t=text), temperature=0.7)
    return llm.chat(llm.FLASH, ROUNDTRIP.format(lang="español rioplatense", t=en),
                    temperature=0.7).strip()


def main() -> None:
    load_dotenv()
    llm.configure()
    crudo = excerpt(RUN / "04_draft.md")
    cands = {
        "corpus": excerpt(CORPUS),
        "crudo": crudo,
        "nuestro": excerpt(RUN / "04_draft_humanizado.md"),
    }
    for name, fn in (("texthum", texthum), ("ida_y_vuelta", roundtrip)):
        try:
            cands[name] = fn(crudo)
        except Exception as e:  # noqa: BLE001 - a broken candidate is a result too
            print(f"{name}: falló ({type(e).__name__}: {e})")

    base = cites(crudo)
    rows = []
    for name, text in cands.items():
        local = humanize.local_score(text)
        c = cites(text)
        row = {"name": name, "words": len(text.split()), "local": local["score"],
               "citas": len(c), "citas_perdidas": sorted(base - c) if name != "corpus" else [],
               "issues": local["issues"][:4]}
        if "--judges" in sys.argv:
            row["judges"] = {j["model"]: j.get("ai_probability")
                             for j in humanize.llm_judges(text)}
        rows.append(row)
        Path(f"bench_{name}.md").write_text(text, encoding="utf-8")

    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
