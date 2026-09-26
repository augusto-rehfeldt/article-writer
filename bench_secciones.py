"""Score real, dossier-grounded pipeline sections with the bench2 judge pool.

bench2 measured 54 *standalone* sections: no dossier, no citations, no page
markers. The judges name concrete data and citations as their main human
signal, so that bench never tested the thing the pipeline actually ships.
This scores `output/<run>/04_secNN.md` the same way bench2 scored its human
windows — one 1.200-word window, four judges, worst reported per text — so the
numbers land on the same axis as `bench2.md`.

    python bench_secciones.py [output/<run>] [--lang en]
"""
from __future__ import annotations

import glob
import json
import re
import sys
from concurrent import futures
from pathlib import Path

from dotenv import load_dotenv

import bench2
import humanize
import llm
import style

CITE = re.compile(r"\([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ'’-]+(?:\s+(?:y|and|&)\s+[\wÁÉÍÓÚÑáéíóúñ'’-]+)?"
                  r"(?:\s+et\s+al\.)?,?\s+\d{4}")
CACHE = Path("bench/secciones.json")


def window(body: str) -> str:
    """The largest ~1.200-word window, matching bench2's human samples."""
    return max(style.windows(body, 1200), key=lambda w: len(w.split()))


def judge(label: str, backend: str, model: str, text: str) -> float | None:
    try:
        bench2.use(backend, model)
        r = humanize.llm_judges(text, models=[model], log=lambda *_: None)
        return float(r[0].get("ai_probability", -1)) if r else None
    except Exception as e:  # noqa: BLE001 - a dead judge must not kill the table
        print(f"  · juez {label} no disponible: {type(e).__name__}: {e}")
        return None


def main() -> None:
    load_dotenv()
    argv = sys.argv[1:]
    lang = "es"
    if "--lang" in argv:
        i = argv.index("--lang")
        lang = argv[i + 1]
        del argv[i:i + 2]          # or the value is read back as the run directory
    run = argv[0] if argv else sorted(glob.glob("output/*"))[-1]
    humanize.LANG = lang

    files = sorted(glob.glob(f"{run}/04_sec[0-9][0-9].md"))
    CACHE.parent.mkdir(exist_ok=True)
    rows = json.loads(CACHE.read_text("utf-8")) if CACHE.exists() else {}

    for f in files:
        body = Path(f).read_text(encoding="utf-8")
        if len(body.split()) < 900:          # bench2's floor for a human window
            continue
        w = window(body)
        row = rows.setdefault(Path(f).name, {})
        row["pal"] = len(w.split())
        row["citas"] = len(CITE.findall(w))
        row["local"] = humanize.local_score(w, lang=lang)["score"]
        row.setdefault("judges", {})
        # Judges are sequential per text (each one repoints the global router)
        # but the four texts' judges of the same family share a backend, so the
        # cheap parallelism is across judges, not files. `use()` mutates module
        # state, so this stays serial on purpose.
        for label, backend, model in bench2.JUDGES:
            if row["judges"].get(label) is None:
                row["judges"][label] = judge(label, backend, model, w)
        got = [v for v in row["judges"].values() if v is not None]
        row["peor"] = max(got) if got else None
        print(f"{Path(f).name}: pal={row['pal']} citas={row['citas']} "
              f"local={row['local']} judges={row['judges']} peor={row['peor']}")
        CACHE.write_text(json.dumps(rows, ensure_ascii=False, indent=2), "utf-8")

    scored = [r["peor"] for r in rows.values() if r.get("peor") is not None]
    if scored:
        print(f"\npeor juez sobre secciones reales: {min(scored)}–{max(scored)} "
              f"(bench2 standalone: 68+ ES / 78+ EN; piso humano EN ≤15, ES ≤28)")


if __name__ == "__main__":
    main()
