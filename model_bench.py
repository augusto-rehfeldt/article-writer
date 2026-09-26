"""Five-model bake-off on the two jobs a model does here: drafting and humanizing.

Candidates: opus (claude), muse-spark-1.3 free (zen), qwen3.8-flash, glm-5.3-flash,
deepseek-v4.1-flash (hyper). Per model and language:

  plain_*   PLAIN prompt                 (what it writes untold)
  styled_*  bench2.styled_prompt(full)   (article-writer drafting conditions)
  hum_*     humanize.humanize(HUM_SRC) with the model as rewriter and kimi-k3 as in-loop judge
            (self_hum_* keep the self-judged run: the pipeline as configured)

Every text is then scored by a fixed panel no candidate belongs to (local_score +
kimi-k3 + minimax-m3 on hyper), next to human windows for calibration. External
detectors are stubbed out: they cost credits and would make rounds incomparable.

    python model_bench.py gen|hum|score|report [--only label]

muse runs through the opencode CLI; after that hop the hyper link stopped answering
in the same process, so run muse with `--only muse-spark-1.3` in its own process.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

from dotenv import load_dotenv

load_dotenv()

import bench2
import humanize
import llm
import style
import ui

OUT = pathlib.Path(__file__).parent / "bench" / "models"
SCORES = OUT / "scores.json"
META = OUT / "meta.json"

CANDIDATES = [
    ("opus", "claude", "opus"),
    ("qwen3.8-flash", "hyper", "qwen3.8-flash"),
    ("glm-5.3-flash", "hyper", "glm-5.3-flash"),
    ("deepseek-v4.1-flash", "hyper", "deepseek-v4.1-flash"),
    ("muse-spark-1.3", "zen", "muse-spark-1.3-contributor-free"),
]
# opus replaces gpt-5.6-terra (dropped): the humanize loop never sees it, so it is
# the held-out read on hum_*. It is also a candidate and soft on its own prose, so
# the opus rows are ranked without it.
PANEL = [("kimi-k3", "hyper", "kimi-k3"), ("minimax-m3", "hyper", "minimax-m3"),
         ("opus", "claude", "opus")]
HUMAN = ["human_es_1", "human_es_2", "human_en_1", "human_en_2"]
# Humanizer input per language. raw_sonnet_en.md is Spanish (a bench2 leftover), so
# English uses kimi-k3's plain draft: English, 1239 words, no candidate's family.
# Spanish uses qwen3.8-max's (93 in bench2): kimi read raw_sonnet_es at 15-45, so
# three of five candidates passed round 1 without rewriting a word. Threshold 0
# makes every candidate rewrite all three rounds; the loop keeps kimi's best.
HUM_SRC = {"es": "raw_qwen3.8-max_es.md", "en": "raw_kimi-k3_en.md"}


def _only() -> list[tuple[str, str, str]]:
    if "--only" in sys.argv:
        want = sys.argv[sys.argv.index("--only") + 1]
        return [c for c in CANDIDATES if c[0] == want]
    return CANDIDATES


def _meta() -> dict:
    return json.loads(META.read_text(encoding="utf-8")) if META.exists() else {}


def _save_meta(key: str, val: dict) -> None:
    m = _meta()
    m[key] = val
    META.write_text(json.dumps(m, indent=1), encoding="utf-8")


def _run(name: str, fn) -> None:
    if (OUT / name).exists():
        return
    t = time.time()
    try:
        body, extra = fn()
    except Exception as e:  # noqa: BLE001 - a dead model is a result
        print(f"[models] {name}: FALLÓ ({type(e).__name__}: {str(e)[:200]})", flush=True)
        _save_meta(name, {"error": f"{type(e).__name__}: {str(e)[:200]}"})
        return
    dt = round(time.time() - t, 1)
    lang = name.rsplit("_", 1)[1][:2]
    if body.strip() and style.detect_language(body) not in (None, lang):
        print(f"[models] {name}: idioma equivocado", flush=True)
        (OUT / f"{name}.rejected").write_text(body, encoding="utf-8")
        _save_meta(f"{name}.rejected", {"error": "wrong language", "secs": dt})
        return
    if not body.strip():
        print(f"[models] {name}: vacío", flush=True)
        _save_meta(name, {"error": "empty", "secs": dt})
        return
    (OUT / name).write_text(body.strip(), encoding="utf-8")
    _save_meta(name, {"secs": dt, **extra})
    print(f"[models] {name}: {len(body.split())} palabras en {dt}s", flush=True)


def gen() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for label, backend, model in _only():
        for lang in ("es", "en"):
            for kind, prompt in (("plain", bench2.PLAIN_ES if lang == "es" else bench2.PLAIN_EN),
                                 ("styled", bench2.styled_prompt(lang, full=True))):
                def draft(prompt=prompt, backend=backend, model=model):
                    bench2.use(backend, model)
                    return llm.chat(model, prompt, temperature=0.8, fallback=False), {}
                _run(f"{kind}_{label}_{lang}.md", draft)


LOOP_JUDGE = ("hyper", "kimi-k3")
_judges = humanize.llm_judges


def hum() -> None:
    """The candidate only rewrites; kimi-k3 decides every round for every model.

    Self-judging measured the judge, not the rewriter: muse read the Sonnet draft at
    14% and returned it untouched. kimi steers the loop, so minimax-m3 is the
    held-out read of the result.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    humanize.external_detectors = lambda *a, **k: []
    for label, backend, model in _only():
        def judged(text, models=None, log=ui.log, lang=None, backend=backend, model=model):
            bench2.use(*LOOP_JUDGE)
            try:
                return _judges(text, models=[LOOP_JUDGE[1]], log=log, lang=lang)
            finally:
                bench2.use(backend, model)
        humanize.llm_judges = judged
        for lang in ("es", "en"):
            raw = (bench2.OUT / HUM_SRC[lang]).read_text(encoding="utf-8")

            def rewrite(raw=raw, lang=lang, backend=backend, model=model):
                bench2.use(backend, model)
                humanize.LANG = lang
                out, rep = humanize.humanize(raw, rounds=3, threshold=0.0, log=ui.log, lang=lang)
                return out, {"passed": rep.get("passed"), "loop_score": rep.get("final_score"),
                             "rounds": len(rep.get("rounds", [])),
                             "selected_round": rep.get("selected_round"),
                             "kept": round(len(out.split()) / len(raw.split()), 2)}
            _run(f"hum_{label}_{lang}.md", rewrite)


def score() -> None:
    data = json.loads(SCORES.read_text(encoding="utf-8")) if SCORES.exists() else {}
    texts = sorted(OUT.glob("*_*_*.md")) + [bench2.OUT / f"{h}.md" for h in HUMAN]
    raws = [bench2.OUT / HUM_SRC[l] for l in ("es", "en")]
    for p in texts + raws:
        lang = "en" if "_en" in p.stem else "es"
        body = p.read_text(encoding="utf-8")
        row = data.setdefault(p.name, {"lang": lang, "words": len(body.split())})
        humanize.LANG = lang
        row.setdefault("local", humanize.local_score(body, lang=lang)["score"])
        row.setdefault("judges", {})
        for label, backend, model in PANEL:
            if row["judges"].get(label) is not None:
                continue
            bench2.use(backend, model)
            try:
                r = humanize.llm_judges(body, models=[model], log=lambda *_: None, lang=lang)
                row["judges"][label] = float(r[0]["ai_probability"]) if r else None
            except Exception as e:  # noqa: BLE001
                print(f"[models] juez {label} sobre {p.name}: {type(e).__name__}", flush=True)
        SCORES.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[models] {p.name}: local={row['local']} {row['judges']}", flush=True)


def report() -> None:
    data = json.loads(SCORES.read_text(encoding="utf-8"))
    meta = _meta()
    print("| texto | pal | seg | local | kimi | minimax | extra |\n|---|---|---|---|---|---|---|")
    for name in sorted(data):
        r, m = data[name], meta.get(name, {})
        extra = {k: v for k, v in m.items() if k != "secs"}
        print(f"| {name[:-3]} | {r['words']} | {m.get('secs', '—')} | {r['local']} | "
              f"{r['judges'].get('kimi-k3')} | {r['judges'].get('minimax-m3')} | {extra or ''} |")
    for name, m in meta.items():
        if "error" in m:
            print(f"- FALLÓ {name}: {m['error']}")


if __name__ == "__main__":
    {"gen": gen, "hum": hum, "score": score, "report": report}[sys.argv[1]]()
