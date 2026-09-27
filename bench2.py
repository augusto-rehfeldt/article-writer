"""Humanizer + model bench, ES and EN, against real human prose.

Four questions in one harness:

  1. Does any third-party humanizer beat ``humanize._rewrite``?
  2. Does any off-the-shelf detector separate human from AI in *Spanish*?
  3. Which detector is worth wiring in for English?
  4. Which drafting model produces the least machine-sounding prose?

Everything is scored with the project's own gate (``local_score`` + LLM judges)
plus every downloadable classifier, and every candidate is measured against the
same human floor: real windows of ``corpus/`` (the author) and ``corpus_en/``
(real papers and books).

    python bench2.py gen      # draft the raw AI texts, one per model per language
    python bench2.py hum      # run every humanizer over the same raw text
    python bench2.py score    # local + judges + classifiers over everything
    python bench2.py report   # markdown tables

Stages cache into bench/. Delete a file to redo just that one.
"""

from __future__ import annotations

import json
import pathlib
import random
import re
import sys
import time

from dotenv import load_dotenv

import humanize
import llm
import style
import ui

OUT = pathlib.Path(__file__).parent / "bench"
SCORES = OUT / "scores.json"

# (label, backend, model). The qwen/kimi/deepseek three go through `hyper`, not the
# local `opencode` CLI: the CLI serves the same models off opencode-go but spends one
# fresh process per completion with no streaming, and a styled-size prompt sat past
# 30 minutes without answering. `x-preview` lives only on zen, which has no key.
MODELS = [
    ("opus", "claude", "opus"),
    ("sonnet", "claude", "sonnet"),
    ("gpt-5.6-terra", "oauth", "gpt-5.6-terra"),
    ("qwen3.8-max", "hyper", "qwen3.8-max"),
    ("kimi-k3", "hyper", "kimi-k3"),
    ("deepseek-v4-pro", "hyper", "deepseek-v4-pro-0813"),
    # zen has no OPENCODE_API_KEY, but the opencode CLI reaches the same catalogue
    # under its `opencode/` prefix and the free tier answers without one. `use()`
    # repoints OPENCODE_PROVIDER for this row alone.
    ("x-preview", "zen-cli", "x-preview-f-free"),
]
# Judging is per text and per judge, so the pool is kept to one model per family:
# two Anthropic, one OpenAI, one Chinese-open-weights. kimi-k3 is the cross-family
# read the 2026-08-22 pool had and the Claude/OpenAI pair does not.
# gpt-5.6-terra is dropped as a judge (2026-09-25); deepseek-v4.1-flash takes its seat.
# kimi-k3 is dropped too (2026-09-26): opus 5.5 takes its seat, gpt-6-astra joins.
JUDGES = [m for m in MODELS if m[0] in ("opus", "sonnet")] + [
    ("deepseek-v4.1-flash", "hyper", "deepseek-v4.1-flash"),
    ("claude-opus-5-5", "claude", "claude-opus-5-5"),
    ("gpt-6-astra", "oauth", "gpt-6-astra")]
WORDS = 1200

TOPIC_ES = ("la automatización del trabajo y lo que le pasa al valor cuando la "
            "fuerza de trabajo deja de ser el centro de la producción")
TOPIC_EN = ("the automation of work and what happens to value when labour power "
            "stops being the centre of production")

# The plain-AI baseline: what a model writes when nobody tells it how to write.
PLAIN_ES = (f"Escribí una sección de ensayo de ciencias sociales, en español, de unas "
            f"{WORDS} palabras, sobre {TOPIC_ES}. Devolvé sólo el texto.")
PLAIN_EN = (f"Write a {WORDS}-word section of a social-science essay about {TOPIC_EN}. "
            "Return only the text.")


def _read(p: pathlib.Path) -> str:
    return p.read_text(encoding="utf-8")


def _write(name: str, body: str) -> pathlib.Path:
    OUT.mkdir(exist_ok=True)
    p = OUT / name
    p.write_text(body, encoding="utf-8")
    return p


def use(backend: str, model: str) -> None:
    """Point the router at one model on one provider, both roles.

    `zen-cli` is not an `llm.PROVIDERS` entry: it is the `go` route with
    `OPENCODE_PROVIDER` flipped from `opencode-go` to `opencode` (OpenCode Zen),
    which is how x-preview is reachable without an `OPENCODE_API_KEY`.
    """
    llm.OPENCODE_PROVIDER = "opencode" if backend == "zen-cli" else "opencode-go"
    llm.configure(chain=["go" if backend == "zen-cli" else backend],
                  pro=model, flash=model)


# ---------------------------------------------------------------- human floor

def human_samples(n: int = 6, seed: int = 7) -> None:
    """Real windows of real prose, the size the judges actually read."""
    rng = random.Random(seed)
    for lang, arts in (("es", style.articles()), ("en", style.articles_en())):
        pool = []
        for slug, body in arts:
            for w in style.windows(body, 1200):
                if len(w.split()) > 900:
                    pool.append((slug, w))
        for i, (slug, w) in enumerate(rng.sample(pool, min(n, len(pool))), 1):
            _write(f"human_{lang}_{i}.md", f"<!-- {slug} -->\n{w}")
        print(f"[bench] {lang}: {min(n, len(pool))} ventanas humanas de {len(pool)}")


# ------------------------------------------------------------------ stage gen

def gen() -> None:
    human_samples()
    for label, backend, model in MODELS:
        for lang, plain in (("es", PLAIN_ES), ("en", PLAIN_EN)):
            for kind, prompt in (("raw", plain),
                                 ("styled", styled_prompt(lang)),
                                 ("guided", styled_prompt(lang, full=True))):
                name = f"{kind}_{label}_{lang}.md"
                if (OUT / name).exists():
                    continue
                use(backend, model)
                t = time.time()
                try:
                    body = llm.chat(model, prompt, temperature=0.8, fallback=False)
                except Exception as e:  # noqa: BLE001 - a dead model is a result
                    print(f"[bench] {name}: FALLÓ ({type(e).__name__}: {str(e)[:120]})")
                    continue
                _write(name, body.strip())
                print(f"[bench] {name}: {len(body.split())} palabras en {time.time() - t:.0f}s")


def styled_prompt(lang: str, full: bool = False) -> str:
    """What the pipeline actually asks for: the style block plus the rhythm quotas.

    Not the real SECTION_PROMPT (that one needs a dossier), but the same two
    levers — verbatim samples of the target voice, and counted quotas — so the
    comparison measures the models under the conditions they really draft in.

    ``full`` swaps the compact block (samples only, what the rewriter sees) for
    the whole one (PRO-written style guide + fingerprint + samples, what drafting
    sees). In English that guide only exists since 2026-08-23, so the two
    variants together measure what building it bought.
    """
    sb = style.style_block(lang=lang) if full else style.style_block_compact(lang=lang)
    if lang == "en":
        return (f"{sb}\n\n# TASK\nWrite a {WORDS}-word section of a social-science "
                f"essay about {TOPIC_EN}, in the voice of the samples above.\n"
                "Quotas, count them before answering: at least 8 sentences under 9 "
                "words, each carrying a concrete datum; at least 4 periods over 45 "
                "words with chained subordination; at least 12 parenthetical asides; "
                "at most 3 explanatory colons; paragraphs of visibly uneven length. "
                "Open on a concrete particular, not a thesis. No metadiscursive "
                "connectives, no programme in the close, no bullet lists.\n"
                "Return only the text.")
    return (f"{sb}\n\n# TAREA\nEscribí una sección de ensayo de ciencias sociales de "
            f"unas {WORDS} palabras sobre {TOPIC_ES}, en la voz de las muestras de arriba.\n"
            "Cuotas, contalas antes de responder: al menos 8 oraciones de menos de 9 "
            "palabras, cada una con un dato concreto; al menos 4 períodos de más de 45 "
            "palabras con subordinación encadenada; al menos 12 incisos entre paréntesis; "
            "como máximo 3 dos puntos explicativos; párrafos de largo visiblemente "
            "desparejo. Abrí con un particular concreto, no con una tesis. Sin "
            "conectores metadiscursivos, sin programa en el cierre, sin viñetas.\n"
            "Devolvé sólo el texto.")


# ------------------------------------------------------------------ stage hum

ROUNDTRIP = ("Translate the following text into {lang}. Complete and faithful "
             "translation, no commentary, no summarising, nothing added. Return "
             "only the translation.\n\n{t}")

LYNOTE_REWRITE = (
    "Rewrite the text below so it reads as natural human writing: vary sentence "
    "length, break the uniform rhythm, cut filler, keep every fact and every "
    "citation. Same language as the input. Return only the rewritten text.\n\n{t}")


def _texthum(text: str, lang: str) -> str:
    from texthumanize import humanize as th
    return th(text, lang=lang, profile="academic", intensity=70).text


def _humano(text: str, lang: str) -> str:
    from humano import humanize as hm
    return hm(text, strength="high").get("humanized_content", "")


def _roundtrip(text: str, lang: str) -> str:
    """lynote-ai's translation chain: hop through a distant language and back."""
    use("claude", "sonnet")
    mid = llm.chat("sonnet", ROUNDTRIP.format(lang="Finnish", t=text), temperature=1.0)
    back = llm.chat("sonnet", ROUNDTRIP.format(
        lang="rioplatense Spanish" if lang == "es" else "English", t=mid), temperature=1.0)
    return back.strip()


def _lynote(text: str, lang: str) -> str:
    """Their v1.5 «standard pipeline»: translation chain, then an LLM rewrite."""
    use("claude", "sonnet")
    return llm.chat("sonnet", LYNOTE_REWRITE.format(t=_roundtrip(text, lang)),
                    temperature=1.3).strip()


def _ours_with(backend: str, model: str):
    """Our rewrite loop, driven by one named model as FLASH.

    The rewriter *is* FLASH, so which model it is changes the result — and in a
    default `.env` FLASH is deepseek, not sonnet. Both are measured.
    """
    def run(text: str, lang: str) -> str:
        use(backend, model)
        humanize.LANG = lang
        out, _ = humanize.humanize(text, rounds=3, threshold=25.0, log=ui.log)
        return out
    return run


HUMANIZERS = [("ours", _ours_with("claude", "sonnet")),
              ("ours-deepseek", _ours_with("hyper", "deepseek-v4-pro-0813")),
              ("texthum", _texthum), ("humano", _humano),
              ("roundtrip", _roundtrip), ("lynote", _lynote)]


def hum() -> None:
    """Every humanizer over the same raw text, so the input is not a variable."""
    for lang in ("es", "en"):
        src = OUT / f"raw_sonnet_{lang}.md"
        if not src.exists():
            print(f"[bench] falta {src.name}; corré gen primero")
            continue
        raw = _read(src)
        for label, fn in HUMANIZERS:
            name = f"hum_{label}_{lang}.md"
            if (OUT / name).exists():
                continue
            t = time.time()
            try:
                body = fn(raw, lang)
            except Exception as e:  # noqa: BLE001 - a broken humanizer is a result
                print(f"[bench] {name}: FALLÓ ({type(e).__name__}: {str(e)[:160]})")
                continue
            if not body.strip():
                print(f"[bench] {name}: devolvió vacío")
                continue
            _write(name, body.strip())
            print(f"[bench] {name}: {len(body.split())} palabras en {time.time() - t:.0f}s")


# ---------------------------------------------------------------- classifiers

# Every downloadable AI-text classifier worth trying, scored on the same texts as
# the LLM judges so «does it separate human from AI» is answered with numbers.
# desklib publishes a custom head (mean pooling + one logit), not a
# SequenceClassification checkpoint, so it needs its own loader.
CLASSIFIERS = [
    ("hello-roberta", "Hello-SimpleAI/chatgpt-detector-roberta", "seq"),
    ("desklib", "desklib/ai-text-detector-v1.01", "desklib"),
    ("desklib-acad", "desklib/ai-text-detector-academic-v1.01", "desklib"),
    ("tmr", "Oxidane/tmr-ai-text-detector", "seq"),
    ("xlmr-multi", "yaya36095/xlm-roberta-text-detector", "seq"),
    # AuTexTification (IberLEF 2023) is the reference Spanish MGT benchmark; these
    # are the only published fine-tunes on it. Trained against BLOOM-era
    # generators, so the question is whether they transfer to 2026 models at all.
    ("autext", "arincon/roberta-base-autextification", "seq"),
    ("autext-oai", "arincon/roberta-base-openai-detector-autextification", "seq"),
]
_clf_cache: dict[str, object] = {}


def _load(name: str, repo: str, kind: str):
    if name in _clf_cache:
        return _clf_cache[name]
    try:
        if kind == "desklib":
            import torch
            from torch import nn
            from transformers import AutoModel, AutoTokenizer

            class Desklib(nn.Module):
                def __init__(self, repo: str) -> None:
                    super().__init__()
                    self.model = AutoModel.from_pretrained(repo)
                    self.classifier = nn.Linear(self.model.config.hidden_size, 1)

                def forward(self, ids, mask):
                    h = self.model(input_ids=ids, attention_mask=mask).last_hidden_state
                    m = mask.unsqueeze(-1).float()
                    pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
                    return torch.sigmoid(self.classifier(pooled)).squeeze(-1)

            tok = AutoTokenizer.from_pretrained(repo)
            net = Desklib(repo)
            # The published weights carry the head; load_state_dict over the hub
            # file is the only route, AutoModel alone drops the classifier.
            from huggingface_hub import hf_hub_download
            try:
                sd_path = hf_hub_download(repo, "model.safetensors")
                from safetensors.torch import load_file
                sd = load_file(sd_path)
            except Exception:
                sd = torch.load(hf_hub_download(repo, "pytorch_model.bin"),
                                map_location="cpu")
            net.load_state_dict(sd, strict=False)
            net.eval()

            def run(text: str) -> float:
                enc = tok(text, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    return float(net(enc["input_ids"], enc["attention_mask"])[0])
        else:
            from transformers import pipeline as hf_pipeline
            pipe = hf_pipeline("text-classification", model=repo,
                               truncation=True, max_length=512)

            def run(text: str) -> float:
                r = pipe(text)[0]
                lab = r["label"].lower()
                ai = lab.endswith("1") or "ai" in lab or "fake" in lab or "chatgpt" in lab
                return r["score"] if ai else 1 - r["score"]
    except Exception as e:  # noqa: BLE001 - a model that will not load is skipped
        print(f"[bench] clasificador {name} no cargó: {type(e).__name__}: {str(e)[:120]}")
        run = None
    _clf_cache[name] = run
    return run


def classify(text: str) -> dict:
    """Every classifier's WORST 512-token window, as the judges do."""
    out = {}
    for name, repo, kind in CLASSIFIERS:
        run = _load(name, repo, kind)
        if run is None:
            continue
        try:
            probs = [run(text[i:i + 1800]) for i in range(0, min(len(text), 20000), 1700)]
            out[name] = round(100 * max(probs), 1)
        except Exception as e:  # noqa: BLE001
            print(f"[bench] {name} falló: {type(e).__name__}")
    return out


# ---------------------------------------------------------------- stage score

def score() -> None:
    data = json.loads(_read(SCORES)) if SCORES.exists() else {}
    files = sorted(p for p in OUT.glob("*.md"))
    for p in files:
        lang = "en" if re.search(r"_en\.md$|_en_\d+\.md$", p.name) else "es"
        body = re.sub(r"^<!--.*?-->\s*", "", _read(p), flags=re.S)
        row = data.setdefault(p.name, {"lang": lang, "words": len(body.split())})
        if "local" not in row:
            humanize.LANG = lang
            row["local"] = humanize.local_score(body, lang=lang)["score"]
            row["issues"] = humanize.local_score(body, lang=lang)["issues"][:4]
        if "clf" not in row:
            row["clf"] = classify(body)
        row.setdefault("judges", {})
        for label, backend, model in JUDGES:
            if row["judges"].get(label) is not None:
                continue          # a None is a failed judge, not a measurement
            use(backend, model)
            humanize.LANG = lang
            try:
                r = humanize.llm_judges(body, models=[model], log=lambda *_: None)
                row["judges"][label] = float(r[0].get("ai_probability", -1)) if r else None
            except Exception as e:  # noqa: BLE001
                print(f"[bench] juez {label} sobre {p.name}: {type(e).__name__}")
                row["judges"][label] = None
            SCORES.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                              encoding="utf-8")
        print(f"[bench] {p.name}: local={row['local']} "
              f"judges={row['judges']} clf={row['clf']}")
    SCORES.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


# --------------------------------------------------------------- stage report

def report() -> None:
    data = json.loads(_read(SCORES))
    clf_names = [n for n, _, _ in CLASSIFIERS]
    for lang in ("es", "en"):
        rows = {k: v for k, v in data.items() if v["lang"] == lang}
        print(f"\n## {lang.upper()}\n")
        head = ["texto", "pal", "local"] + [l for l, _, _ in JUDGES] + clf_names
        print("| " + " | ".join(head) + " |")
        print("|" + "---|" * len(head))
        for name in sorted(rows):
            r = rows[name]
            cells = [name[:-3], str(r["words"]), str(r["local"])]
            cells += [str(r["judges"].get(l)) for l, _, _ in JUDGES]
            cells += [str(r["clf"].get(c, "—")) for c in clf_names]
            print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    load_dotenv()
    llm.INTERACTIVE = False
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    {"gen": gen, "hum": hum, "score": score, "report": report,
     "human": human_samples}[cmd]()
