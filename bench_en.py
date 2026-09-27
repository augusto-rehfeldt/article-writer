"""Small English bench: opus vs the gpt-6 family as drafters, and whether
``humanize._rewrite`` helps or hurts each of them.

Reuses bench2's prompt (``styled_prompt("en", full=True)``, the best condition
measured on 2026-08-24) and its human windows. A model reads its own family's
prose as human, so the report gives each text's worst judge twice: over the
whole panel, and over the judges from *other* families only.

    python bench_en.py            # gen, rewrite, score, report (cached in bench/en/)
"""

from __future__ import annotations

import json
import pathlib
import re
import time

from dotenv import load_dotenv

import bench2
import humanize
import llm
import style

OUT = bench2.OUT / "en"
SCORES = OUT / "scores.json"
SAMPLES = 2
# (label, backend, model, family)
DRAFTERS = [
    ("opus-5.5", "claude", "claude-opus-5-5", "anthropic"),
    ("gpt-6-luna", "oauth", "gpt-6-luna", "openai"),
    ("gpt-6-sol", "oauth", "gpt-6-sol", "openai"),
    ("gpt-6-astra", "oauth", "gpt-6-astra", "openai"),
]
REWRITERS = ("opus-5.5", "gpt-6-luna")
JUDGES = [
    ("opus-5.5", "claude", "claude-opus-5-5", "anthropic"),
    ("sonnet", "claude", "sonnet", "anthropic"),
    ("gpt-6-astra", "oauth", "gpt-6-astra", "openai"),
    ("deepseek-v4.1-flash", "hyper", "deepseek-v4.1-flash", "deepseek"),
]


def _save(name: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(body, encoding="utf-8")


def gen() -> None:
    for i in range(1, 4):  # the human floor: bench2's cached real windows
        src = bench2.OUT / f"human_en_{i}.md"
        if not (OUT / src.name).exists():
            _save(src.name, re.sub(r"^<!--.*?-->\s*", "", src.read_text(encoding="utf-8"), flags=re.S))
    prompt = bench2.styled_prompt("en", full=True)
    for label, backend, model, _ in DRAFTERS:
        for n in range(1, SAMPLES + 1):
            name = f"draft_{label}_{n}.md"
            if (OUT / name).exists():
                continue
            bench2.use(backend, model)
            t = time.time()
            try:
                body = llm.chat(model, prompt, temperature=0.8, fallback=False).strip()
            except Exception as e:  # noqa: BLE001 - a dead model is a result
                print(f"[bench] {name}: FALLÓ ({type(e).__name__}: {str(e)[:160]})")
                continue
            _save(name, body)
            print(f"[bench] {name}: {len(body.split())} palabras en {time.time() - t:.0f}s")


def rewrite() -> None:
    """The rewrite loop's step, once, by the drafter itself, fed the text issues."""
    sb = style.style_block_compact(lang="en")
    for label, backend, model, _ in DRAFTERS:
        src = OUT / f"draft_{label}_1.md"
        name = f"rewrite_{label}_1.md"
        if label not in REWRITERS or not src.exists() or (OUT / name).exists():
            continue
        text = src.read_text(encoding="utf-8")
        issues = humanize.local_score(text, lang="en")["issues_texto"]
        bench2.use(backend, model)
        out = humanize._rewrite(text, issues, sb, print, lang="en")
        if out == text:
            print(f"[bench] {name}: rewrite rejected every block; not saved")
            continue
        _save(name, out)
        print(f"[bench] {name}: {len(out.split())} palabras")


def score() -> None:
    data = json.loads(SCORES.read_text(encoding="utf-8")) if SCORES.exists() else {}
    humanize.LANG = "en"
    for p in sorted(OUT.glob("*.md")):
        body = p.read_text(encoding="utf-8")
        row = data.setdefault(p.name, {"words": len(body.split())})
        row.setdefault("local", humanize.local_score(body, lang="en")["score"])
        judges = row.setdefault("judges", {})
        for label, backend, model, _ in JUDGES:
            if judges.get(label) is not None:
                continue
            bench2.use(backend, model)
            try:
                r = humanize.llm_judges(body, models=[model], log=lambda *_: None, lang="en")
                judges[label] = float(r[0]["ai_probability"]) if r else None
            except Exception as e:  # noqa: BLE001
                print(f"[bench] juez {label} sobre {p.name}: {type(e).__name__}")
                judges[label] = None
            SCORES.write_text(json.dumps(data, indent=1), encoding="utf-8")
        print(f"[bench] {p.name}: local={row['local']} {judges}")


def _family(name: str) -> str | None:
    for label, *_, fam in DRAFTERS:
        if f"_{label}_" in name:
            return fam
    return None


def report() -> str:
    data = json.loads(SCORES.read_text(encoding="utf-8"))
    head = ["text", "words", "local"] + [j[0] for j in JUDGES] + ["worst", "worst other-family"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, row in sorted(data.items()):
        fam = _family(name)
        got = {j[0]: row["judges"].get(j[0]) for j in JUDGES}
        vals = [v for v in got.values() if v is not None]
        other = [got[j[0]] for j in JUDGES if j[3] != fam and got[j[0]] is not None]
        cell = lambda v: "—" if v is None else f"{v:.0f}"  # noqa: E731
        lines.append("| " + " | ".join(
            [name.removesuffix(".md"), str(row["words"]), f"{row['local']:.0f}"]
            + [cell(got[j[0]]) for j in JUDGES]
            + [cell(max(vals) if vals else None), cell(max(other) if other else None)]) + " |")
    table = "\n".join(lines)
    (OUT / "report.md").write_text(table + "\n", encoding="utf-8")
    return table


if __name__ == "__main__":
    load_dotenv(pathlib.Path(__file__).with_name(".env"))
    gen()
    rewrite()
    score()
    print(report())
