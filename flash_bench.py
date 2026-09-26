"""Flash-model bake-off: glm-5.3-flash vs qwen3.8-flash vs deepseek-v4-flash.

Drafts the same ES and EN prompts with each candidate, scores every text with
the project's own gate (local_score + LLM judges), and prints latency plus a
verdict table. Reuses bench2 machinery; caches into bench/flash/.

    python flash_bench.py          # gen + score + report
"""

from __future__ import annotations

import json
import pathlib
import time

from dotenv import load_dotenv

load_dotenv()

import bench2
import humanize
import llm

OUT = pathlib.Path(__file__).parent / "bench" / "flash"
SCORES = OUT / "scores.json"

CANDIDATES = [
    # (label, backend, model, thinking) — `thinking` models spend max_tokens on
    # reasoning before answering, so a small cap returns empty content.
    ("glm-5.3-flash", "hyper", "glm-5.3-flash", True),
    ("qwen3.8-flash", "hyper", "qwen3.8-flash", False),
    ("deepseek-v4-flash", "hyper", "deepseek-v4-flash-0731", True),
    # the current FLASH default, for scale
    ("deepseek-v4-pro", "hyper", "deepseek-v4-pro-0813", True),
]
WORDS = 1200


JUDGE_ONLY = [
    ("kimi-k3", "hyper", "kimi-k3"),   # cross-family; glm-5.2 skipped as same-family
]


def _read(p: pathlib.Path) -> str:
    return p.read_text(encoding="utf-8")


def gen() -> dict[str, float]:
    OUT.mkdir(parents=True, exist_ok=True)
    lat: dict[str, float] = {}
    for label, backend, model, thinking in CANDIDATES:
        for lang, prompt in (("es", bench2.PLAIN_ES), ("en", bench2.PLAIN_EN)):
            name = f"raw_{label}_{lang}.md"
            if (OUT / name).exists():
                continue
            bench2.use(backend, model)
            t = time.time()
            try:
                body = llm.chat(model, prompt, temperature=0.8, fallback=False,
                                max_tokens=16384 if thinking else None)
            except Exception as e:  # noqa: BLE001 - a dead model is a result
                print(f"[flash] {name}: FALLÓ ({type(e).__name__}: {str(e)[:120]})")
                continue
            dt = time.time() - t
            lat[label] = round(dt, 1)
            (OUT / name).write_text(body.strip(), encoding="utf-8")
            print(f"[flash] {name}: {len(body.split())} palabras en {dt:.0f}s")
    meta = OUT / "latency.json"
    merged = json.loads(_read(meta)) if meta.exists() else {}
    merged.update(lat)
    meta.write_text(json.dumps(merged, indent=1), encoding="utf-8")
    return merged


def score() -> None:
    data = json.loads(_read(SCORES)) if SCORES.exists() else {}
    for p in sorted(OUT.glob("raw_*.md")):
        lang = "en" if p.name.endswith("_en.md") else "es"
        body = _read(p)
        row = data.setdefault(p.name, {"lang": lang, "words": len(body.split())})
        if "local" not in row:
            humanize.LANG = lang
            row["local"] = humanize.local_score(body, lang=lang)["score"]
        row.setdefault("judges", {})
        for label, backend, model in JUDGE_ONLY:
            if row["judges"].get(label) is not None:
                continue
            if backend == "oauth":
                continue  # local proxy down this session; not a measurement
            bench2.use(backend, model)
            humanize.LANG = lang
            try:
                r = humanize.llm_judges(body, models=[model], log=lambda *_: None)
                row["judges"][label] = float(r[0].get("ai_probability", -1)) if r else None
            except Exception as e:  # noqa: BLE001
                print(f"[flash] juez {label} sobre {p.name}: {type(e).__name__}")
                row["judges"][label] = None
        SCORES.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[flash] {p.name}: local={row['local']} judges={row['judges']}")


def report() -> None:
    data = json.loads(_read(SCORES))
    lat = json.loads(_read(OUT / "latency.json")) if (OUT / "latency.json").exists() else {}
    judge_names = [l for l, _, _ in JUDGE_ONLY]
    head = ["modelo", "lang", "pal", "seg", "local"] + judge_names
    print("\n| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    agg: dict[str, list[float]] = {}
    for name in sorted(data):
        r = data[name]
        label = name.removeprefix("raw_").removesuffix(".md")
        model = "_".join(label.split("_")[:-1])
        js = [str(r["judges"].get(j)) for j in judge_names]
        vals = [r["local"]] + [v for v in r["judges"].values() if v is not None]
        agg.setdefault(model, []).extend(vals)
        print(f"| {label} | {r['lang']} | {r['words']} | {lat.get(model, '—')} "
              f"| {r['local']} | " + " | ".join(js) + " |")
    print("\n## promedio (menor = menos detectable como IA)")
    for m, vs in sorted(agg.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        print(f"- {m}: {sum(vs) / len(vs):.1f}")


if __name__ == "__main__":
    gen()
    score()
    report()
