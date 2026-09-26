"""Does the rewrite loop help, now that it no longer chases the stylometric floors?

`humanize()` used to hand `_rewrite` every message `local_score` produced,
including the five that only say «this number is below the corpus floor». Those
correlate -0.85 with the judges. This runs the loop both ways over the same
Spanish text and reports the worst judge before and after.

    python bench_reescritura.py [rondas]
"""
from __future__ import annotations

import json
import pathlib
import sys

from dotenv import load_dotenv

import bench2
import humanize
import llm
import style

OUT = pathlib.Path("bench")
SRC = OUT / "guided_sonnet_es.md"


def judge(text: str) -> dict:
    got = {}
    for label, backend, model in bench2.JUDGES:
        try:
            bench2.use(backend, model)
            r = humanize.llm_judges(text, models=[model], log=lambda *_: None)
            got[label] = float(r[0]["ai_probability"]) if r else None
        except Exception as e:  # noqa: BLE001
            print(f"  · {label}: {type(e).__name__}")
            got[label] = None
    vals = [v for v in got.values() if v is not None]
    return {"judges": got, "peor": max(vals) if vals else None}


def main(rounds: int = 2) -> None:
    load_dotenv()
    humanize.LANG = "es"
    text = SRC.read_text("utf-8")
    sb = style.style_block_compact()
    out: dict = {}

    for label, only_text in (("con_pisos", False), ("sin_pisos", True)):
        cur = text
        for r in range(1, rounds + 1):
            local = humanize.local_score(cur, lang="es")
            issues = list(local["issues_texto" if only_text else "issues"])
            bench2.use("claude", "sonnet")
            cur = humanize._rewrite(cur, issues, sb, print)
            out[f"{label}_r{r}"] = {"local": humanize.local_score(cur, "es")["score"],
                                    **judge(cur)}
            print(f"{label} ronda {r}: {out[f'{label}_r{r}']}")
        (OUT / f"reescrito_{label}.md").write_text(cur, encoding="utf-8")

    out["entrada"] = {"local": humanize.local_score(text, "es")["score"], **judge(text)}
    print("\nentrada:", out["entrada"])
    (OUT / "reescritura.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), "utf-8")


if __name__ == "__main__":
    main(int(sys.argv[1]) if sys.argv[1:] else 2)
