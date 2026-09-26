"""Bilingual editing with local style diagnostics and optional detector feedback.

Style measurements guide editing; they do not establish authorship. The configured
PRO/FLASH judges and external detectors supply the threshold scores. Missing scores
remain indeterminate, and every report identifies the exact text that was measured.
"""

from __future__ import annotations

import collections
import functools
import hashlib
import json
import math
import os
import re
import statistics
from concurrent import futures

import requests

import llm
import research
import style
import ui

# Phrases that scream "LLM" in Spanish academic prose. Any of these that ALSO
# appear in the author's corpus are dropped at runtime — they'd be his own tics.
LLM_TELLS = [
    "es importante destacar", "cabe destacar", "cabe mencionar", "es fundamental entender",
    "en el mundo actual", "en la era digital", "en el vasto", "un aspecto clave",
    "juega un papel crucial", "desempeña un papel", "profundizar en", "desentrañar",
    "a lo largo de la historia", "en resumen", "en definitiva, podemos", "en conclusión, podemos",
    "no es más que un reflejo", "abordar esta cuestión", "una mirada más profunda",
    "resulta crucial", "resulta esencial", "es crucial", "de vital importancia",
    "en última instancia, se trata de", "por un lado", "por otro lado",
    "en primer lugar,", "en segundo lugar,", "en tercer lugar,",
    "este artículo explora", "este ensayo explora", "exploraremos", "analizaremos cómo",
    "invita a reflexionar", "nos invita a", "queda claro que", "sin lugar a dudas",
    "el objetivo de este", "en el presente artículo se",
    "tanto a nivel", "a nivel global", "el panorama actual", "un enfoque holístico",
    "multifacético", "matizado", "la intersección entre", "arrojar luz sobre",
    "sirve como recordatorio", "más allá de la mera", "no solo … sino también",
    "es un fenómeno complejo", "un llamado a la acción", "en última instancia, la clave",
    # Antithesis and pseudo-profound stock, each verified absent from the
    # author's corpus (2026-08-23) before being added.
    "no se trata de una cuestión", "la clave está en", "se trata de entender",
    "pone sobre la mesa", "abre la posibilidad", "plantea la pregunta de",
    "surge la pregunta", "cabe preguntarse", "no es un detalle menor",
    "no es gratuito", "no es anecdótico", "todo lo contrario", "nada más lejos",
    "cobra sentido", "adquiere sentido", "dicho de otro modo",
    "visto así", "leído así",
]

STRUCTURAL_TELLS = [
    (r"(?m)^\s*[-*•]\s", 0.02, "listas con viñetas (el autor casi no las usa)"),
    (r"\bEn conclusión\b", 0.0005, "cierre con «En conclusión»"),
    (r"—", 0.02, "sobreuso de raya (—)"),
    (r"\b(Primero|Segundo|Tercero),", 0.0008, "enumeración escolar"),
    (r"(?m)^#{2,}\s", 0.02, "exceso de subtítulos"),
    # Explanatory colons (setup:elaboration) are the single strongest AI tell
    # in Spanish academic prose: corpus averages 6-7/1k words, drafts hit 13+/1k.
    (r"[a-záéíóúñ)]\s*:\s+[a-záéíóúñ«]", 0.009,
     "densidad de dos puntos explicativos (patrón setup:elaboración); el corpus "
     "usa ~6/1k palabras, este texto excede ese techo"),
    # The rewriter's short-sentence quota manufactures «X no es Y.» verdicts;
    # the author almost never writes them (corpus max 0.12/1k words, once in
    # 54 articles; our humanized drafts hit 5 in 4.2k words).
    (r"(?m)^[^#!\n]{0,100}\b[Nn]o\s+(?:es|son|está|están|hay|tiene)\b[^#\n]{0,50}\.\s*$",
     0.15, "párrafos-sentencia negativos («X no es Y.» / «No hay X.» como párrafo suelto): "
           "el autor casi nunca los escribe; una oración breve tiene que llevar dato concreto"),
]
EN_STRUCTURAL_TELLS = [
    (r"(?m)^[^#!\n]{0,100}\b(?:It'?s not|It is not|This is not|That is not|There (?:is|are) no)\b"
     r"[^#\n]{0,50}\.\s*$", 0.15,
     "verdict paragraphs («It's not X.» as a standalone line): manufactured depth, "
     "a classic model giveaway"),
]


_WORD_RE = re.compile(r"[a-záéíóúüñ]{2,}", re.I)

# Article language. "en" (--english / pipeline.LANG) swaps every Spanish tell
# list and prompt here for an English counterpart. The stylometric floors stay:
# burstiness, sentence rhythm and paragraph asymmetry are language-agnostic.
LANG = "es"

# Verified against Wikipedia's "Signs of AI writing" and the harshaneel/humanize
# signal list; deliberately conservative because there is no English corpus to
# whitelist the author's own tics with.
ENGLISH_TELLS = [
    "delve into", "delving into", "it is important to note", "it's important to note",
    "it is worth noting", "in today's world", "in today's fast-paced",
    "plays a crucial role", "serves as a reminder", "a testament to",
    "the ever-evolving", "navigating the complexities", "in conclusion,",
    "a rich tapestry", "vibrant tapestry", "when it comes to",
    "at the end of the day", "in the realm of", "shed light on",
    "pave the way for", "not only … but also", "it's not just about",
    "is more than just", "game-changer", "unlock the potential",
    "embark on a journey", "underscores the importance", "in summary,",
    "the landscape of", "boasts a", "seamlessly integrate",
]


def repeated_phrases(text: str, n: int = 6, times: int = 2, limit: int = 14,
                     cap: int = 30) -> list[str]:
    """Verbatim word sequences the text serves more than once.

    Sections are drafted independently and rewritten block by block, so the same
    formula comes back three sections later word for word. A reader reads that as
    a machine, and the judges name it explicitly.

    Overlapping windows are merged into one span, so a repeated sentence reports
    as a single hit instead of as a dozen shifted views of itself.
    """
    words = _WORD_RE.findall(text.lower())
    if len(words) <= n:
        return []
    grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    counts = collections.Counter(grams)
    spans: list[list[int]] = []
    for i, g in enumerate(grams):
        if counts[g] < times:
            continue
        if spans and i - spans[-1][1] <= 1:
            spans[-1][1] = i
        else:
            spans.append([i, i])
    out, seen = [], set()
    for a, b in spans:
        phrase = " ".join(words[a:min(b + n, a + cap)])
        if phrase not in seen:
            seen.add(phrase)
            out.append(phrase)
        if len(out) >= limit:
            break
    return out


@functools.lru_cache(maxsize=1)
def _corpus_allowed() -> frozenset[str]:
    """Tells the author actually uses. Penalising his own tics would be absurd."""
    joined = " ".join(b for _, b in style.articles()).lower()
    return frozenset(t for t in LLM_TELLS if t.split("…")[0].strip() in joined)


@functools.lru_cache(maxsize=1)
def _corpus_allowed_en() -> frozenset[str]:
    """English tells that show up in corpus_en/ — human academic prose uses some
    of these on purpose, and charging for them would push the rewriter away from
    a register real writers actually occupy."""
    joined = " ".join(b.lower() for _, b in style.articles_en())
    return frozenset(t for t in ENGLISH_TELLS if t.split("…")[0].strip() in joined)


def local_score(text: str, lang: str | None = None) -> dict:
    """0 = reads like the author's corpus, 100 = reads like a chatbot.

    Metrics are compared against the p05 of the author's own prose measured over
    windows of comparable size — being *inside* the range he actually writes in
    costs nothing, which is the only way the score means anything.
    """
    fp = style.fingerprint(text)
    lang = lang or LANG
    ref = style.corpus_fingerprint(lang=lang)
    rng = ref.get("ranges") or {}
    tells = LLM_TELLS if lang == "es" else ENGLISH_TELLS
    # Whitelist per language: tells the reference corpus itself uses are tics of
    # real human prose, not machines.
    allowed = _corpus_allowed() if lang == "es" else _corpus_allowed_en()
    penalties: list[tuple[float, str]] = []
    # Messages that only say «this number is below the corpus floor», as opposed
    # to a concrete defect quoted with its location. They are separated because
    # over the seven dossier-grounded sections of the 20260824 run corr(local
    # score, worst judge) = -0.85: the section the floors like least (sec03,
    # local 22.2) is the one the judges like most (62), and the one that
    # satisfies them (sec08, local 0.3) reads 92. `issues_texto` drops them, and
    # is what to feed the rewriter if that correlation is ever confirmed — the
    # one direct A/B run so far (`bench_reescritura.py`) did not confirm it.
    floors: list[str] = []

    def floor(metric: str, fallback: float) -> float:
        # p25, not p05: the bottom of the author's own range still reads human,
        # and calibration showed generated prose sits just under it.
        return rng.get(metric, {}).get("p25", fallback)

    # Burstiness is the single strongest public-detector feature.
    b_floor = floor("burstiness", 0.45)
    if fp["burstiness"] < b_floor:
        penalties.append((min(40, (b_floor - fp["burstiness"]) * 220),
                          f"burstiness {fp['burstiness']} por debajo del piso humano "
                          f"{b_floor:.3f}: oraciones demasiado parejas, hay que alternar "
                          "períodos muy largos con oraciones brevísimas"))
        floors.append(penalties[-1][1])
    lo = floor("sent_len_mean", 14.0)
    hi = rng.get("sent_len_mean", {}).get("p95", 42.0)
    if not lo <= fp["sent_len_mean"] <= hi:
        direction = "cortas" if fp["sent_len_mean"] < lo else "largas"
        gap = lo - fp["sent_len_mean"] if fp["sent_len_mean"] < lo else fp["sent_len_mean"] - hi
        penalties.append((min(20, gap * 1.6),
                          f"oraciones {direction} de más ({fp['sent_len_mean']} palabras; "
                          f"el autor se mueve entre {lo:.0f} y {hi:.0f})"))
        floors.append(penalties[-1][1])
    l_floor = floor("long_sent_ratio", 0.04)
    if fp["long_sent_ratio"] < l_floor:
        penalties.append((12, "faltan períodos largos con subordinación encadenada "
                              f"(>45 palabras): {fp['long_sent_ratio']:.2f} vs {l_floor:.2f}"))
        floors.append(penalties[-1][1])
    t_floor = floor("ttr", 0.25)
    if fp["ttr"] < t_floor:
        penalties.append((8, f"léxico repetitivo (ttr {fp['ttr']} vs piso {t_floor:.3f})"))
        floors.append(penalties[-1][1])

    words = max(len(text.split()), 1)
    hits = [t for t in tells if t not in allowed and t.split("…")[0].strip() in text.lower()]
    if hits:
        penalties.append((min(35, 7 * len(hits)),
                          "frases-cliché de IA presentes: " + "; ".join(hits[:8])))
    for pattern, per_word_max, label in (STRUCTURAL_TELLS if lang == "es"
                                         else EN_STRUCTURAL_TELLS):
        n = len(re.findall(pattern, text))
        if n / words > per_word_max:
            penalties.append((min(12, n * 1.5), label))

    if (repes := repeated_phrases(text)):
        penalties.append((min(18, 3 * len(repes)),
                          "frases repetidas casi textualmente en distintos párrafos o "
                          "secciones (si no son terminología del tema, reescribí o borrá "
                          "la segunda aparición): "
                          + "; ".join(f"«{r}»" for r in repes[:8])))

    # Uniform paragraph lengths are a strong generated-text signal.
    paras = [len(p.split()) for p in text.split("\n\n") if len(p.split()) > 25]
    if len(paras) > 4:
        cv = statistics.pstdev(paras) / statistics.fmean(paras)
        cv_floor = floor("para_cv", 0.30)
        if cv < cv_floor:
            penalties.append((14, f"párrafos de longitud demasiado uniforme "
                                  f"(cv={cv:.2f}, piso humano {cv_floor:.2f})"))
            floors.append(penalties[-1][1])

    # Colon density: the strongest measured AI tell in this corpus. Corpus sits
    # at 6-7 per 1k words; drafts routinely hit 12-14. Every extra colon above
    # the corpus ceiling is a setup:elaboration pair that screams machine.
    colon_count = len(re.findall(r"[a-záéíóúñ)]\s*:\s+[a-záéíóúñ«]", text))
    colon_per_1k = colon_count / max(words, 1) * 1000
    colon_ceiling = 7.5  # just above corpus p75 (~7)
    if lang == "es" and colon_per_1k > colon_ceiling:
        gap = colon_per_1k - colon_ceiling
        penalties.append((min(20, gap * 2.5),
                          f"dos puntos explicativos en exceso ({colon_per_1k:.1f}/1k "
                          f"palabras; techo del corpus ~{colon_ceiling:.0f}/1k). "
                          "Reemplazá la mayoría por punto y seguido, coma, inciso "
                          "entre paréntesis o reformulación sin pausa"))

    # The antithesis frame is THE English AI tell («It's not X, it's Y», «not just
    # X but Y») and a strong one in Spanish («no X sino Y»). Both sit at ~1/1k in
    # human prose; past ~2.5/1k it reads as a rhetorical tic rather than an argument.
    if lang == "es":
        sino_pat = (r"\bno\b[^.!?;\n]{0,70}?\bsino\b"
                    r"|\bno (?:es|son|está|están)\b[^.!?;\n]{0,60}?"
                    r",\s*(?:es|son|está|están|se trata)\b")
    else:
        sino_pat = (r"\b(?:is|are|was|it'?s)\s+not\b[^.!?;\n]{0,50}?[,;—]\s*"
                    r"(?:it'?s|it is|they are|rather)\b"
                    r"|\bnot (?:just|only|merely|simply)\b[^.!?;\n]{0,60}?\bbut\b")
    sino = len(re.findall(sino_pat, text, re.I))
    sino_1k = sino / max(words, 1) * 1000
    if sino_1k > 2.5:
        penalties.append((min(10, (sino_1k - 2.5) * 4),
                          f"demasiadas antítesis «no X sino Y» ({sino_1k:.1f}/1k palabras; "
                          "el autor usa ~1/1k): convertí varias en afirmaciones directas"
                          if lang == "es" else
                          f"too many «not X, but Y» antitheses ({sino_1k:.1f}/1k words; "
                          "human prose sits near 1/1k): rewrite most as direct assertions"))

    score = min(100.0, sum(p for p, _ in penalties))
    msgs = [m for _, m in penalties]
    return {"score": round(score, 1), "issues": msgs, "metrics": fp,
            "issues_texto": [m for m in msgs if m not in floors]}


JUDGE_PROMPT = """Sos un detector forense de texto generado por IA. Analizá el siguiente texto \
en español y estimá la probabilidad (0-100) de que haya sido escrito por un modelo de lenguaje.

Fijate en: uniformidad rítmica, transiciones prefabricadas, simetría argumental sospechosa, \
adjetivación genérica, ausencia de idiosincrasia, cierres explicativos redundantes, \
falta de riesgo retórico, SOBREUSO DE DOS PUNTOS EXPLICATIVOS (el patrón «X : Y» donde X \
prepara y Y elabora es la señal más fuerte de texto generado en español; un autor humano \
usa dos puntos para citas o enumeraciones, no como conector entre cláusulas), \
construcciones anafóricas mecánicas («Ese/Esa X que Y», «Lo que X es Y»), preguntas \
retóricas seguidas de respuesta inmediata, la ANTÍTESIS DE MANUAL («no es X, es Y», \
«no X sino Y», «la pregunta no es A, es B») repetida como fórmula, y párrafos-sentencia \
negativos o aforismos sin dato («La forma piensa.» como párrafo suelto).

Devolvé JSON: {{"ai_probability": <0-100>, "veredicto": "humano"|"ia"|"dudoso", \
"señales": ["...", "..."], "fragmentos_sospechosos": ["cita literal", "..."]}}

TEXTO:
{text}"""


JUDGE_PROMPT_EN = """You are a forensic detector of AI-generated text. Read the following \
English text and estimate the probability (0-100) that a language model wrote it.

Look for: uniform sentence rhythm, prefab transitions, suspicious argument symmetry, \
generic adjective padding, absence of idiosyncrasy, redundant explanatory closures, \
lack of rhetorical risk, OVERUSE OF EXPLANATORY COLONS, em-dash overuse, the rule-of-three \
reflex (every idea delivered as triads), mechanical anaphora ("This is not X. It is Y."), \
the TEXTBOOK ANTITHESIS repeated as a formula («not X but Y», «It's not about X, it's about Y»), \
negation-verdict or aphorism paragraphs with no concrete datum («The form thinks.»), stock \
phrases (delve, tapestry, ever-evolving landscape, serves as a reminder, testament to).

Return JSON: {{"ai_probability": <0-100>, "verdict": "human"|"ai"|"unsure", \
"señales": ["...", "..."], "fragmentos_sospechosos": ["literal quote", "..."]}}

TEXT:
{text}"""


JUDGE_WINDOW = 24000
# The judges never saw past the first 24k characters, and glm read that opening
# of a 10.8k-word draft as human (18%) while scoring the same draft's mid-sections
# at 82%. From `largo` up the gate scored the opening and a machine-sounding tail
# slipped past under one judge. Three windows bound the cost at 3x per judge.
JUDGE_SPAN = 3


def _judge_windows(text: str) -> list[str]:
    """Opening, middle and tail of a long text; short texts pass whole."""
    if len(text) <= JUDGE_WINDOW:
        return [text]
    step = len(text) // JUDGE_SPAN
    mids = [text[i * step:min(i * step + JUDGE_WINDOW, len(text))]
            for i in range(1, JUDGE_SPAN - 1)]
    return [text[:JUDGE_WINDOW], *mids, text[-JUDGE_WINDOW:]]


def llm_judges(text: str, models: list[str] | None = None, log=ui.log,
               lang: str | None = None) -> list[dict]:
    """Ask every judge at once; they are different providers and don't queue together.

    Long texts are judged on opening, middle and tail; each judge reports its
    WORST window, because a clean opening must not launder a machine-sounding tail.
    """
    windows = _judge_windows(text)
    if len(windows) > 1:
        log(f"  · texto largo ({len(text)}c): juzgando {len(windows)} ventanas "
            "(inicio, medio, final)")

    def ask(m: str) -> dict | None:
        try:
            worst: dict | None = None
            for w in windows:
                prompt = (JUDGE_PROMPT if (lang or LANG) == "es" else JUDGE_PROMPT_EN)
                r = llm.chat_json(m, prompt.format(text=w), temperature=0.2)
                r["ai_probability"] = _probability(r["ai_probability"])
                if worst is None or float(r.get("ai_probability", 50)) > \
                        float(worst.get("ai_probability", 50)):
                    worst = r
            worst["model"] = m
            return worst
        except Exception as e:  # noqa: BLE001 - a judge being down must not stop the run
            log(f"  · juez {m} no disponible: {type(e).__name__}")
            return None

    models = models or llm.JUDGES
    with futures.ThreadPoolExecutor(max_workers=max(1, len(models))) as pool:
        results = list(pool.map(ask, models))
    out = [r for r in results if r]
    for r in out:
        log(f"  · juez {r['model']}: {r.get('ai_probability')}% IA "
            f"({r.get('veredicto', r.get('verdict'))})")
    return out


def _probability(value) -> float:
    """Reject missing, nonfinite and out-of-range scores instead of passing them."""
    if isinstance(value, bool):
        raise ValueError("boolean detector score")
    score = float(value)
    if not math.isfinite(score) or not 0 <= score <= 100:
        raise ValueError("detector score must be between 0 and 100")
    return score


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# name, env var holding the key, request builder, 0-100 score extractor.
# Each service has its own auth style, payload shape, length cap and score
# polarity (Winston reports *humanity*), so the table carries all four.
KEYED_DETECTORS = [
    ("gptzero", "GPTZERO_API_KEY",
     lambda k, t: dict(url="https://api.gptzero.me/v2/predict/text",
                       headers={"x-api-key": k}, json={"document": t[:50000]}, timeout=90),
     lambda d: d["documents"][0]["class_probabilities"]["ai"] * 100),
    ("sapling", "SAPLING_API_KEY",
     lambda k, t: dict(url="https://api.sapling.ai/api/v1/aidetect",
                       json={"key": k, "text": t[:8000]}, timeout=90),
     lambda d: d["score"] * 100),
    ("winston", "WINSTON_API_KEY",
     lambda k, t: dict(url="https://api.gowinston.ai/v2/ai-content-detection",
                       headers={"Authorization": f"Bearer {k}"},
                       json={"text": t[:60000]}, timeout=120),
     lambda d: 100 - d.get("score", 50)),
]


# Local learned detector, English-only. Optional: only loads if `transformers`
# is installed; the model downloads once (~500 MB) and then runs offline. A real
# trained classifier is sharper than any LLM judge, so in --english mode it is
# worth its weight — but the pipeline degrades silently without it.
@functools.lru_cache(maxsize=1)
def _hf_detector():
    try:
        from transformers import pipeline as hf_pipeline
        return hf_pipeline("text-classification",
                           model="Hello-SimpleAI/chatgpt-detector-roberta",
                           truncation=True, max_length=512)
    except Exception:
        return None


def external_detectors(text: str, log=ui.log, lang: str | None = None) -> list[dict]:
    """Real detector services. Keyed ones only fire when the key is set."""
    out = []
    for name, env, request, score in KEYED_DETECTORS:
        key = os.environ.get(env)
        if not key:
            continue
        try:
            r = requests.post(**request(key, text))
            r.raise_for_status()
            out.append({"name": name, "ai_probability": round(_probability(score(r.json())), 1)})
        except Exception as e:  # noqa: BLE001 - a dead detector must not stop the round
            log(f"  · {name} error: {e}")
    try:  # keyless, undocumented, frequently changes shape
        r = requests.post("https://api.zerogpt.com/api/detect/detectText",
                          json={"input_text": text[:14000]},
                          headers={"Content-Type": "application/json"}, timeout=60)
        r.raise_for_status()
        fake = r.json().get("data", {}).get("fakePercentage")
        if fake is not None:
            out.append({"name": "zerogpt", "ai_probability": round(_probability(fake), 1)})
    except Exception:
        pass
    if (lang or LANG) == "en" and (clf := _hf_detector()):
        try:
            # 512 tokens ≈ 1800 characters; report the WORST window, like the judges,
            # because a clean opening must not launder a machine-sounding tail.
            probs = [clf(text[i:i + 1800])[0] for i in range(0, min(len(text), 60000), 1700)]
            worst = max((r["score"] if r["label"].endswith("1") else 1 - r["score"]
                         for r in probs), default=0.0)
            out.append({"name": "roberta-local", "ai_probability": round(100 * worst, 1)})
        except Exception as e:  # noqa: BLE001 - a dead detector must not stop the round
            log(f"  · roberta-local error: {e}")
    for d in out:
        log(f"  · {d['name']}: {d['ai_probability']}% IA")
    return out


REWRITE_PROMPT = """Revisá este bloque como editor del autor. Mejorá la claridad, la
continuidad del argumento y la naturalidad del español. Si una frase ya funciona,
conservala. No cambies la tesis ni conviertas la revisión en otro ensayo.

REGISTRO:
{registro}

CRITERIOS EDITORIALES:
{writing_rules}

Reglas duras:
- Conservá TODOS los argumentos, datos, nombres y citas, incluidas sus páginas y años.
- Conservá literalmente las citas textuales y las intervenciones editoriales.
- Conservá la extensión del bloque (±10%); no agregues contenido para rellenar.
- Conservá encabezados, notas al pie, sus números y sus llamadas.
- No agregues experiencias personales, polémica, sarcasmo ni información.
- No agregues viñetas, subtítulos, negritas ni comentarios sobre tu tarea.
- Las señales de abajo son sugerencias para revisar en contexto, no órdenes de
  alterar una frase correcta. Las métricas de estilo no prueban quién escribió un texto.

PROBLEMAS A REVISAR:
{issues}

{style_block}

CONTEXTO ANTERIOR (solo para enlazar; no lo copies):
{previous}

CONTEXTO SIGUIENTE (solo para enlazar; no lo copies):
{following}

=== BLOQUE A REESCRIBIR ===
{text}
=== FIN DEL BLOQUE ===

Devolvé únicamente el bloque reescrito."""


REWRITE_PROMPT_EN = """Edit this block for clarity, continuity and idiomatic English.
Preserve sentences that already work. Keep the author's argument and degree of
certainty; a style edit must not turn it into a different essay.

REGISTER:
{registro}

EDITORIAL PRIORITIES:
{writing_rules}

Hard rules:
- Keep EVERY argument, fact, name and citation, including its year and page numbers.
- Preserve direct quotations and editorial interventions verbatim.
- Keep the block length within ±10%; do not add material to fill space.
- Preserve headings, footnotes, their numbers and their references.
- Do not invent personal experience, disagreement, sarcasm or information.
- No new bullets, headings, bold or commentary about the editing task.
- The observations below are suggestions to check in context, not commands to
  alter a correct sentence. Style metrics cannot establish who wrote a text.

PROBLEMS TO REVIEW:
{issues}

{style_block}

PREVIOUS CONTEXT (for continuity only; do not copy):
{previous}

FOLLOWING CONTEXT (for continuity only; do not copy):
{following}

=== BLOCK TO REWRITE ===
{text}
=== END OF BLOCK ===

Return only the rewritten block."""


def _protected(text: str) -> collections.Counter:
    """Conservative preservation check; semantic fidelity still needs editorial review."""
    spans = [m.group(0) for pat in (research.CITE_RE, research.NARRATIVE_RE)
             for m in pat.finditer(text)]
    # ponytail: quotation spans and numeric tokens, not a semantic fact checker;
    # editorial review must still verify claims and unquoted names against sources.
    spans += re.findall(r'«[^»]+»|“[^”]+”|"[^"\n]+"|(?m:^>[^\n]*)', text)
    spans += re.findall(r"\[[^\]\n]+\]|\b\d+(?:[.,:/–-]\d+)*(?:[a-z])?\b", text)
    return collections.Counter(" ".join(s.split()) for s in spans)


def _rewrite(text: str, issues: list[str], style_block: str, log,
             batch_words: int = 700, register: str = "", lang: str | None = None) -> str:
    """Rewrite in blocks instead of all at once.

    Neighbouring context keeps independently edited blocks connected.
    """
    blocks, buf, n = [], [], 0
    for para in text.split("\n\n"):
        if para.startswith("#"):          # headings pass through untouched
            if buf:
                blocks.append(("text", "\n\n".join(buf)))
                buf, n = [], 0
            blocks.append(("head", para))
            continue
        buf.append(para)
        n += len(para.split())
        if n >= batch_words:
            blocks.append(("text", "\n\n".join(buf)))
            buf, n = [], 0
    if buf:
        blocks.append(("text", "\n\n".join(buf)))

    lang = lang or LANG
    issue_text = "\n".join(f"- {x}" for x in issues[:24]) or "—"

    def one(index: int, block: str) -> str:
        w = len(block.split())
        en = lang == "en"
        prompt = REWRITE_PROMPT_EN if en else REWRITE_PROMPT
        try:
            fmt = dict(
                writing_rules=style.writing_rules(lang),
                previous=" ".join("\n\n".join(b for _, b in blocks[max(0, index - 2):index]).split()[-100:]),
                following=" ".join("\n\n".join(b for _, b in blocks[index + 1:index + 3]).split()[:100]),
                registro=register or ("A sober theoretical essay: argued with data and "
                                      "citations, not with adjectives or irony." if en else
                                      "Ensayo teórico sobrio: se argumenta con datos y "
                                      "citas, no con adjetivos ni con ironía."),
                issues=issue_text, style_block=style_block, text=block)
            out = llm.chat(llm.FLASH, prompt.format(**fmt), temperature=0.6).strip()
            # Enforce the prompt's length bound before accepting a rewrite.
            if not 0.9 * w <= len(out.split()) <= 1.1 * w:
                log("  · bloque reescrito volvió recortado o ampliado; lo dejo como está")
                return block
            if _protected(block) != _protected(out):
                log("  · reescritura alteró citas, cifras o notas; conservo el bloque")
                return block
            if style.detect_language(out) not in (None, lang):
                log("  · reescritura cambió de idioma; conservo el bloque")
                return block
            if re.search(r"(?m)^\s*(?:#{1,6}\s|```)", out):
                log("  · reescritura agregó encabezados o código; conservo el bloque")
                return block
            return out or block
        except Exception as e:  # noqa: BLE001 - keep the original block on failure
            log(f"  · reescritura de bloque falló ({type(e).__name__}); lo dejo como está")
            return block

    # Blocks are independent, so a book-length rewrite costs one block's latency
    # rather than the sum of them.
    with futures.ThreadPoolExecutor(max_workers=4) as pool:
        rewritten = {i: pool.submit(one, i, b)
                     for i, (kind, b) in enumerate(blocks) if kind == "text"}
        return "\n\n".join(rewritten[i].result() if i in rewritten else b
                           for i, (kind, b) in enumerate(blocks))


def humanize(text: str, *, rounds: int = 3, threshold: float = 25.0,
             style_block: str | None = None, register: str = "",
             log=ui.log, lang: str | None = None) -> tuple[str, dict]:
    """Edit and measure; local style diagnostics are not an authorship verdict."""
    lang = lang or LANG
    if lang not in ("en", "es"):
        raise ValueError("lang must be 'en' or 'es'")
    if not text.strip():
        raise ValueError("cannot evaluate empty text")
    if not math.isfinite(threshold) or not 0 <= threshold <= 101:
        raise ValueError("threshold must be finite and between 0 and 101")
    if rounds <= 0:
        return text, {"rounds": [], "passed": None, "final_score": None, "skipped": True,
                      "text_hash": text_hash(text), "lang": lang}
    # The English corpus carries real samples; fall back to a register note only
    # while corpus_en/ has not been built yet.
    sb = style_block or ((style.style_block_compact(lang="en") if style.articles_en()
                          else "Academic essay English, plain and concrete: short "
                               "declarative sentences next to long subordinate ones, "
                               "first person allowed, no filler adjectives.")
                         if lang == "en" else style.style_block_compact())
    report: dict = {"rounds": [], "lang": lang}
    best, best_score = text, 1e9
    best_result = None
    for i in range(1, rounds + 1):
        local = local_score(text, lang=lang)
        judges = llm_judges(text, log=log, lang=lang)
        ext = external_detectors(text, log=log, lang=lang)
        probs = [_probability(d["ai_probability"]) for d in judges + ext]
        worst = max(probs) if probs else None
        result = {"round": i, "local": local["score"], "judges": judges,
                  "external": ext, "worst": worst, "text_hash": text_hash(text)}
        report["rounds"].append(result)
        log(f"[detector] ronda {i}: estilo={local['score']} detector={worst} (umbral {threshold})")
        rank = worst if worst is not None else 1000 + local["score"]
        if rank < best_score:
            best, best_score, best_result = text, rank, result
        if worst is not None and worst < threshold:
            report["passed"] = True
            report["final_score"] = worst
            report["text_hash"] = text_hash(text)
            report["selected_round"] = i
            return text, report
        if i == rounds or worst is None:
            break
        # Give the editor textual observations, not numerical rhythm quotas.
        issues = list(local["issues_texto"])
        for j in judges:
            issues += [f"[{j['model']}] {s}" for s in (j.get("señales") or [])[:4]]
            issues += [f"[{j['model']}] fragmento delator: «{f}»"
                       for f in (j.get("fragmentos_sospechosos") or [])[:3]]
        log(f"[detector] reescribiendo por bloques ({len(issues)} señales)…")
        rewritten = _rewrite(text, issues, sb, log, register=register, lang=lang)
        if rewritten == text:
            break
        text = rewritten
    report["final_score"] = best_result["worst"]
    report["passed"] = (best_result["worst"] < threshold
                        if best_result["worst"] is not None else None)
    report["text_hash"] = text_hash(best)
    report["selected_round"] = best_result["round"]
    return best, report


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()
    body = sys.stdin.read() if not sys.argv[1:] else open(sys.argv[1], encoding="utf-8").read()
    print(json.dumps(local_score(body), ensure_ascii=False, indent=2))
