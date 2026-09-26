"""How much of the judges' verdict is the text, and how much is the draw?

Four prompt-level interventions moved the worst-judge score by nothing (median
90 before and after), while the same arm swung 78-94 between two samples of the
*same* prompt. Before optimising further it is worth knowing whether the metric
is measuring the text at all: the gate is `max` over four judges, and a `max`
over noisy draws rises with the number of draws whatever the text says.

So: score the same text several times, judge by judge, and read the spread.
Human windows and generated ones, side by side.

    python bench_ruido.py [repeats]
"""
from __future__ import annotations

import json
import pathlib
import statistics
import sys

from dotenv import load_dotenv

import bench2
import humanize
import llm

OUT = pathlib.Path("bench")
RESULTS = OUT / "ruido.json"

# Two real human windows and two generated ones, all English, all ~1.200 words.
# `human_en_1_limpio` is the same Postone window with the scan's soft hyphens
# («interpreta¬ tions») and hard line wraps removed: `corpus_en/` is built out of
# OCR'd PDFs and Gutenberg dumps, so every English "human" window carries
# mechanical debris no language model can emit. If the floor moves when the
# debris goes, the floor was measuring the scanner, not the writer.
#
# The Spanish side is the one that matters for the pipeline and it is clean:
# `corpus/` is scraped from WordPress, so `human_es_1` is modern published prose
# with no scanner in the middle. Judged against a generated Spanish section, it
# is the honest floor.
TEXTS = {"human_en_1.md": "en", "human_en_1_limpio.md": "en", "human_en_3.md": "en",
         "reglas_viejo_1.md": "en", "reglas_nuevo_1.md": "en",
         "human_es_1.md": "es", "human_es_6.md": "es", "guided_sonnet_es.md": "es"}


def main(repeats: int = 3) -> None:
    load_dotenv()
    rows = json.loads(RESULTS.read_text("utf-8")) if RESULTS.exists() else {}

    for name, lang in TEXTS.items():
        humanize.LANG = lang
        text = (OUT / name).read_text("utf-8")
        row = rows.setdefault(name, {})
        for label, backend, model in bench2.JUDGES:
            draws = row.setdefault(label, [])
            while len(draws) < repeats:
                try:
                    bench2.use(backend, model)
                    r = humanize.llm_judges(text, models=[model], log=lambda *_: None)
                    draws.append(float(r[0]["ai_probability"]) if r else None)
                except Exception as e:  # noqa: BLE001
                    print(f"  · {name} / {label}: {type(e).__name__}: {e}")
                    break
                RESULTS.write_text(json.dumps(rows, ensure_ascii=False, indent=2), "utf-8")
        print(f"{name}: " + "  ".join(
            f"{l}={row.get(l)}" for l, _, _ in bench2.JUDGES))

    print()
    for name, row in rows.items():
        # The gate is the worst judge, so the honest per-draw figure is the worst
        # of the four judges *within one draw*, not the worst number ever seen.
        per_draw = []
        for i in range(repeats):
            got = [row[l][i] for l, _, _ in bench2.JUDGES
                   if len(row.get(l, [])) > i and row[l][i] is not None]
            if got:
                per_draw.append(max(got))
        flat = [v for l, _, _ in bench2.JUDGES for v in row.get(l, []) if v is not None]
        if per_draw:
            print(f"{name:22s} peor-juez por tirada {per_draw}  "
                  f"| todas las lecturas {min(flat):.0f}-{max(flat):.0f} "
                  f"(sd {statistics.pstdev(flat):.1f})")


if __name__ == "__main__":
    main(int(sys.argv[1]) if sys.argv[1:] else 3)
