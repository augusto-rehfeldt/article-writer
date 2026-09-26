"""A/B one section's drafting prompt against the judges.

`bench_secciones.py` measured that the section obeying every rule in
`SECTION_PROMPT` (sec08, 92) reads *more* machine-written than the one that
obeys almost none (sec03, 62). This re-drafts the same section from the same
prompt with a block of rules removed and judges both arms, so the comparison
is prompt-only: identical dossier, identical synopsis, identical `evitar`.

    python bench_reglas.py capture output/<run> 8 viejo   # freeze the real prompt
    python bench_reglas.py run    [n_samples]             # draft + judge every arm

`capture` is run once per version of `SECTION_PROMPT`: check the old prompt out,
capture it as `viejo`, restore, capture as `nuevo`. Both are then drafted with
the same model at the same temperature against the same section, so the only
difference between the arms is the prompt.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import statistics
import sys

from dotenv import load_dotenv

import bench2
import humanize
import llm
import pipeline
import research

OUT = pathlib.Path("bench")
RESULTS = OUT / "reglas.json"


def prompt_path(name: str) -> pathlib.Path:
    return OUT / (f"prompt_seccion.txt" if name == "viejo"
                  else f"prompt_seccion_{name}.txt")

# The seven "human, not machine" rules, verbatim from SECTION_PROMPT. Arm B
# drops them; everything else in the prompt is untouched.
SEVEN = ("- Escribí como un humano que discute, no como un modelo que expone.",
         "- TENÉ POSICIÓN, NO PLEITO.")


def _block(prompt: str, start: str, end: str) -> tuple[int, int]:
    i = prompt.index(start)
    return i, prompt.index(end, i)


def arms() -> dict[str, str]:
    """Every captured prompt, one arm each.

    The seven "human, not machine" rules were measured on 2026-08-25 and stay:
    stripping them made the section slightly *worse* (median worst judge 86 with
    them, 91.5 without, 92.5 with the rhythm quotas gone too). `sin7()` keeps the
    surgery around for a re-test with more samples; nothing calls it.
    """
    return {("viejo" if p.name == "prompt_seccion.txt" else p.stem.split("_", 2)[2]):
            p.read_text("utf-8") for p in sorted(OUT.glob("prompt_seccion*.txt"))}


def sin7(prompt: str) -> str:
    i, j = _block(prompt, *SEVEN)
    return prompt[:i] + prompt[j:]


def capture(run_dir: str, n: int, name: str = "viejo") -> None:
    """Run `pipeline.draft()` until it asks for section ``n``, and keep the prompt."""
    load_dotenv()
    d = pathlib.Path(run_dir)
    topic = json.loads((d / "01_topic.json").read_text("utf-8"))
    outline = json.loads((d / "03_outline.json").read_text("utf-8"))
    sources = research.load_dossier(d / "02_dossier.json")
    pipeline.LANG = "en"

    cache = d / f"04_sec{n:02d}.md"
    keep = cache.read_text("utf-8") if cache.exists() else None
    if keep is not None:
        cache.unlink()          # or draft() serves it from disk and never builds a prompt

    grabbed: list[str] = []
    real = llm.chat

    def spy(model, prompt, **kw):
        if "# ENCARGO" in prompt:
            grabbed.append(prompt)
            raise KeyboardInterrupt      # nothing past this section needs writing
        return real(model, prompt, **kw)

    llm.chat = spy
    try:
        run = pipeline.Run(fmt="paper", mode="auto", dir=d)
        pipeline.draft(run, topic, outline, sources)
    except KeyboardInterrupt:
        pass
    finally:
        llm.chat = real
        if keep is not None:
            cache.write_text(keep, encoding="utf-8")   # the run is left as it was

    if not grabbed:
        sys.exit("no section prompt was built (every section cached?)")
    dest = prompt_path(name)
    dest.write_text(grabbed[-1], encoding="utf-8")
    print(f"[bench] prompt de la sección {n} guardado ({len(grabbed[-1])}c) en {dest}")


def run(samples: int = 2) -> None:
    load_dotenv()
    humanize.LANG = "en"
    rows = json.loads(RESULTS.read_text("utf-8")) if RESULTS.exists() else {}

    # The drafting model is the only lever measured to move the judges (bench2:
    # opus 68 / sonnet 82 in Spanish), so it is part of the arm name, not a
    # hidden constant. `sonnet` is what the 20260824 run actually drafted with.
    drafter = os.environ.get("AW_BENCH_DRAFTER", "sonnet")
    for arm, text in arms().items():
        for s in range(1, samples + 1):
            name = f"{arm}_{s}" if drafter == "sonnet" else f"{arm}@{drafter}_{s}"
            row = rows.setdefault(name, {})
            body = OUT / f"reglas_{name}.md"
            if not body.exists():
                bench2.use("claude", drafter)
                out = pipeline._strip_preamble(llm.chat(llm.FLASH, text, temperature=0.9))
                body.write_text(out, encoding="utf-8")
            out = body.read_text("utf-8")
            row["pal"] = len(out.split())
            row["local"] = humanize.local_score(out, lang="en")["score"]
            row.setdefault("judges", {})
            for label, backend, model in bench2.JUDGES:
                if row["judges"].get(label) is None:
                    try:
                        bench2.use(backend, model)
                        r = humanize.llm_judges(out, models=[model], log=lambda *_: None)
                        row["judges"][label] = float(r[0]["ai_probability"]) if r else None
                    except Exception as e:  # noqa: BLE001
                        print(f"  · juez {label}: {type(e).__name__}")
                        row["judges"][label] = None
            got = [v for v in row["judges"].values() if v is not None]
            row["peor"] = max(got) if got else None
            print(f"{name}: pal={row['pal']} local={row['local']} "
                  f"judges={row['judges']} peor={row['peor']}")
            RESULTS.write_text(json.dumps(rows, ensure_ascii=False, indent=2), "utf-8")

    print()
    for arm in arms():
        got = [r["peor"] for k, r in rows.items()
               if k.rsplit("_", 1)[0] == arm and r.get("peor") is not None]
        if got:
            print(f"{arm:16s} peor juez: {sorted(got)} mediana {statistics.median(got)}")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    if sys.argv[1:2] == ["capture"]:
        capture(sys.argv[2], int(sys.argv[3]),
                sys.argv[4] if sys.argv[4:] else "viejo")
    else:
        run(int(sys.argv[2]) if sys.argv[2:] else 2)
