"""Author style model, derived from the reference corpus.

Two complementary layers:
  1. A *quantitative fingerprint* (sentence-length distribution, burstiness,
     punctuation and connector frequencies, lexical richness). This is what the
     humanizer targets numerically — matching the author's rhythm is what most
     AI-detector heuristics actually measure.
  2. A *prose style guide* written by the PRO model after reading the corpus.
     This is what the drafting model reads.
"""

from __future__ import annotations

import functools
import json
import pathlib
import random
import re
import statistics
from collections import Counter

import llm
import ui

ROOT = pathlib.Path(__file__).parent
CORPUS = ROOT / "corpus"                 # published articles (Revista La Cueva)
CORPUS_FAC = ROOT / "corpus_facultad"    # university work, added by ingest_facultad.py
CORPUS_EN = ROOT / "corpus_en"           # human-written English prose (build_corpus_en.py)
GUIDE = ROOT / "style_guide.md"
GUIDE_EN = ROOT / "style_guide_en.md"
FINGERPRINT = ROOT / "style_fingerprint.json"
FINGERPRINT_EN = ROOT / "style_fingerprint_en.json"

_SENT = re.compile(r"[^.!?…]+[.!?…]+[\])»\"']*|\S+$")


def _read_dir(d: pathlib.Path) -> list[tuple[str, str]]:
    out = []
    if not d.exists():
        return out
    for p in sorted(d.glob("*.md")):
        body = p.read_text(encoding="utf-8").split("\n\n", 2)[-1]
        if len(body.split()) > 200:
            out.append((p.stem, body))
    return out


@functools.lru_cache(maxsize=2)
def articles(include_facultad: bool = True) -> list[tuple[str, str]]:
    """Every reference text, published or academic, as (slug, body) pairs.

    Cached: this re-reads ~200k words off disk, and ``excerpts`` alone calls it
    twice per invocation. Call ``articles.cache_clear()`` if the corpus changes
    inside a running process.
    """
    dirs = [CORPUS] + ([CORPUS_FAC] if include_facultad and CORPUS_FAC.exists() else [])
    return [x for d in dirs for x in _read_dir(d)]


@functools.lru_cache(maxsize=1)
def articles_en() -> list[tuple[str, str]]:
    """The English reference corpus: human-written prose from corpus_en/."""
    return _read_dir(CORPUS_EN)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.findall(text) if len(s.strip()) > 1]


def detect_language(text: str) -> str | None:
    """Conservative English/Spanish check; short or ambiguous passages stay unknown."""
    text = re.sub(r'«[^»]*»|“[^”]*”|"[^"\n]*"|(?m:^>[^\n]*)', "", text)
    words = re.findall(r"\b[^\W\d_]+\b", text.lower())
    if len(words) < 30:
        return None
    counts = Counter(words)
    en = sum(counts[w] for w in "the and of to is was with this that which from were".split())
    es = sum(counts[w] for w in "el la los las del que una un por para con como pero".split())
    # ponytail: function-word evidence only; abstain on mixed or quotation-heavy prose.
    if max(en, es) < len(words) * 0.08 or max(en, es) < 2 * min(en, es):
        return None
    return "en" if en > es else "es"


def fingerprint(text: str) -> dict:
    """Numeric signature of a text. Compared draft-vs-corpus by the humanizer."""
    sents = sentences(text)
    lens = [len(s.split()) for s in sents] or [0]
    words = re.findall(r"\b[\wáéíóúüñÁÉÍÓÚÑ]+\b", text.lower())
    paras = [p for p in text.split("\n\n") if len(p.split()) > 15]
    mean = statistics.fmean(lens)
    sd = statistics.pstdev(lens) if len(lens) > 1 else 0.0
    return {
        "sentences": len(sents),
        "words": len(words),
        "sent_len_mean": round(mean, 2),
        "sent_len_sd": round(sd, 2),
        # Burstiness: human prose varies sentence length far more than LLM prose.
        "burstiness": round(sd / mean, 3) if mean else 0.0,
        "sent_len_p10": round(_pct(lens, 10), 1),
        "sent_len_p90": round(_pct(lens, 90), 1),
        "long_sent_ratio": round(sum(1 for x in lens if x > 45) / len(lens), 3),
        "short_sent_ratio": round(sum(1 for x in lens if x < 10) / len(lens), 3),
        "para_len_mean": round(statistics.fmean([len(p.split()) for p in paras]), 1) if paras else 0,
        "ttr": round(len(set(words)) / len(words), 4) if words else 0,
        "per_1k": {ch: round(text.count(ch) / max(len(words), 1) * 1000, 2)
                   for ch in ("«", "“", ";", ":", "—", "(", "¿", "!", "[")},
    }


def _pct(values: list[float], p: float) -> float:
    v = sorted(values)
    if not v:
        return 0.0
    k = (len(v) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def windows(text: str, size: int = 1200) -> list[str]:
    """Split into ~``size``-word chunks on paragraph boundaries."""
    out, buf, n = [], [], 0
    for para in text.split("\n\n"):
        w = len(para.split())
        if n + w > size and buf:
            out.append("\n\n".join(buf))
            buf, n = [], 0
        buf.append(para)
        n += w
    if n > size // 3:
        out.append("\n\n".join(buf))
    return out


def corpus_ranges(size: int = 1200, lang: str = "es") -> dict:
    """Percentiles of each metric across same-sized windows of the author's prose.

    Comparing a 1.800-word draft against the 194k-word aggregate is a category
    error: variance grows with length, so the author's own articles scored 49-58
    on a scale where 0 was supposed to mean "reads human". Measuring windows of
    the size actually being judged fixes that.
    """
    per: dict[str, list[float]] = {}
    for _, body in (articles() if lang == "es" else articles_en()):
        for chunk in windows(body, size):
            fp = fingerprint(chunk)
            # Skip degenerate windows — bibliographies, formula sheets, heading
            # runs. They drag the low percentiles to nonsense (burstiness 0.06).
            if fp["sentences"] < 15 or fp["sent_len_mean"] < 12:
                continue
            for k in ("burstiness", "sent_len_mean", "long_sent_ratio",
                      "short_sent_ratio", "ttr"):
                per.setdefault(k, []).append(fp[k])
            paras = [len(p.split()) for p in chunk.split("\n\n") if len(p.split()) > 25]
            if len(paras) > 3:
                per.setdefault("para_cv", []).append(
                    statistics.pstdev(paras) / statistics.fmean(paras))
    return {k: {"p05": round(_pct(v, 5), 4), "p25": round(_pct(v, 25), 4),
                "p50": round(_pct(v, 50), 4), "p95": round(_pct(v, 95), 4),
                "n": len(v)}
            for k, v in per.items()}


def corpus_fingerprint(refresh: bool = False, lang: str = "es") -> dict:
    path = FINGERPRINT if lang == "es" else FINGERPRINT_EN
    arts = articles() if lang == "es" else articles_en()
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    if not arts:
        # Degrade instead of dying: humanize.local_score(lang="en") must work
        # (on built-in floors) even before build_corpus_en.py ever ran.
        ui.log(f"[style] sin corpus para lang={lang!r}; uso pisos por defecto")
        return {"signature_phrases": [], "ranges": {}}
    joined = "\n\n".join(body for _, body in arts)
    fp = fingerprint(joined)
    fp["signature_phrases"] = signature_phrases(joined)
    fp["ranges"] = corpus_ranges(lang=lang)
    path.write_text(json.dumps(fp, ensure_ascii=False, indent=2), encoding="utf-8")
    return fp


def signature_phrases(text: str, top: int = 40) -> list[str]:
    """Recurring 3-4 word sequences: the author's actual verbal tics."""
    words = re.findall(r"\b[\wáéíóúüñ]+\b", text.lower())
    grams = Counter()
    for n in (3, 4):
        for i in range(len(words) - n):
            g = " ".join(words[i:i + n])
            if len(g) > 12:
                grams[g] += 1
    return [g for g, c in grams.most_common(top * 3) if c >= 4][:top]


def excerpts(n: int = 3, chars: int = 3500, seed: int | None = None,
             lang: str = "es") -> str:
    """Verbatim passages for few-shot grounding. Nothing beats real samples."""
    rng = random.Random(seed)
    arts = articles() if lang == "es" else articles_en()
    picks = rng.sample(arts, min(n, len(arts)))
    out = []
    for slug, body in picks:
        start = rng.randint(0, max(0, len(body) - chars))
        out.append(f"--- fragmento de «{slug}» ---\n{body[start:start + chars]}")
    return "\n\n".join(out)


BUILD_PROMPT = """Sos un analista literario y estilométrico. A continuación tenés fragmentos \
extensos escritos por UN MISMO autor: artículos publicados en una revista de teoría social \
y trabajos universitarios de sociología (monografías, parciales domiciliarios, informes), \
todo en español rioplatense.

Tu tarea: escribir una GUÍA DE ESTILO operativa, en español, tan precisa que otro escritor \
pueda producir texto indistinguible del original. No describas en abstracto: citá ejemplos \
literales del corpus para cada rasgo.

Cubrí obligatoriamente:
1. Registro y persona (uso de primera persona, impersonal, «se» reflejo, futuro de intención).
2. Sintaxis: longitud y arquitectura de las oraciones, subordinación, incisos, uso del \
punto y coma y los dos puntos, aposiciones.
3. Rasgos arcaizantes o marcados (enclisis verbal tipo «débese», «llégase», «tómese»; \
inversiones; latinismos; conectores como «empero», «asimismo», «por ende», «tal es así que»).
4. Puntuación y tipografía: comillas angulares «» vs inglesas, corchetes para términos en \
otro idioma, cursivas, notas al pie.
5. Aparato de citas: formato exacto autor-año-página, citas en bloque, citas dejadas SIN \
traducir en inglés/alemán/francés, uso de «s/n» y «et. al.».
6. Arquitectura del texto: títulos (patrones como «De la…», «De lo que es…», series entre \
corchetes), apertura, hipótesis explícita, partes numeradas, palabras clave, conclusión, \
bibliografía.
7. Estrategia argumental: cómo introduce un autor, cómo lo contrasta con otro, cómo marca \
desacuerdo, cómo usa preguntas retóricas, cómo cierra polémicamente.
8. Campo léxico y marco teórico recurrente (categorías, autores citados una y otra vez).
9. Diferencias entre los tres registros: el largo-académico (monografías, papers), el \
corto-polémico (notas de intervención) y el universitario (parciales, informes de cátedra).
10. QUÉ NO HACE NUNCA este autor (listá antipatrones concretos).

Devolvé solo la guía en markdown, sin preámbulo. Extensión: 1200-2000 palabras.

=== CORPUS ===
{corpus}
=== FIN DEL CORPUS ===

Métricas cuantitativas medidas sobre el corpus completo (usalas en la guía):
{metrics}
"""


BUILD_PROMPT_EN = """You are a literary analyst and stylometrician. Below are extended samples \
written by DIFFERENT human authors: recent research papers and classic social-theory books, \
all in English. This is NOT one author's voice — it is the register the pipeline must imitate: \
human academic English.

Your task: write an operational STYLE GUIDE, in English, precise enough that a writer could \
produce prose indistinguishable from these samples on the dimensions they share. Quote \
literal examples for every trait. Do not describe what only one author does; describe what \
the human side of this corpus does that AI text does not.

Cover mandatorily:
1. Sentence architecture: length distribution, subordination habits, fragments, semicolons.
2. Paragraph behaviour: asymmetry, digressions, openings that are not topic sentences.
3. Hedging and stance: how human academics qualify, concede, disagree, self-correct.
4. Punctuation and typography: dashes, parentheses, quotation conventions, citation formats.
5. Argumentative moves: how sources are introduced and contrasted; rhetorical questions; \
first person; footnotes asides.
6. Vocabulary: concrete vs abstract balance, field jargon, idiomatic texture.
7. What NONE of these writers do (list concrete anti-patterns that mark AI prose).

Return only the guide in markdown, no preamble. Length: 1000-1800 words.

=== CORPUS ===
{corpus}
=== END OF CORPUS ===

Quantitative metrics measured over the whole corpus (use them in the guide):
{metrics}
"""


def build_guide(refresh: bool = False, sample_chars: int = 90000,
                lang: str = "es") -> str:
    guide_path = GUIDE if lang == "es" else GUIDE_EN
    if guide_path.exists() and not refresh:
        return guide_path.read_text(encoding="utf-8")
    arts = articles() if lang == "es" else articles_en()
    if not arts:
        raise SystemExit("corpus/ vacío — corré primero: python scrape_corpus.py"
                         if lang == "es" else
                         "corpus_en/ vacío — corré primero: python build_corpus_en.py")
    # Sample from every article so both the long-academic and short-polemic
    # registers are represented, not just the longest pieces.
    per = max(2000, sample_chars // len(arts))
    sample = "\n\n".join(f"### {slug}\n{body[:per]}" for slug, body in arts)
    fp = corpus_fingerprint(refresh=True, lang=lang)
    ui.log(f"[style] construyendo guía ({lang}) con {llm.PRO} sobre "
           f"{len(sample.split()):,} palabras…")
    prompt = BUILD_PROMPT if lang == "es" else BUILD_PROMPT_EN
    guide = llm.chat(llm.PRO, prompt.format(
        corpus=sample, metrics=json.dumps(fp, ensure_ascii=False, indent=2)),
        temperature=0.4)
    guide_path.write_text(guide, encoding="utf-8")
    return guide


def style_block_compact(seed: int | None = None, lang: str = "es") -> str:
    """Short style context for per-block rewriting.

    The full block is ~35 KB and gets resent for every block of every round,
    which is what made the rewrite stage crawl. The rewriter already carries
    editing instructions, so it needs samples of the voice, not the treatise.
    """
    if lang == "en":
        return ("# VERBATIM SAMPLES OF HUMAN ACADEMIC ENGLISH (imitate this voice)\n"
                "Voice reference only: any citations in these samples are NOT usable "
                "bibliography.\n\n" + excerpts(2, 1800, seed, lang="en"))
    return ("# MUESTRAS LITERALES DEL AUTOR (imitá esta voz)\n"
            "Referencia de voz solamente: las citas que aparezcan acá NO son "
            "bibliografía utilizable.\n\n" + excerpts(2, 1800, seed))


_guide_en_dead = False    # one failed build attempt per process, not one per draft call


def writing_rules(lang: str = "es") -> str:
    """Shared editorial priorities for drafting and rewriting in each language."""
    if lang == "en":
        return """Write idiomatic English for the intended reader and register.
Use precise verbs, concrete subjects and transitions that express the actual connection
between ideas. Keep technical terms consistent; explain them when the reader needs it.
Let the argument determine sentence length and paragraph shape. Do not manufacture
short sentences, long subordinate clauses, asides, rhetorical questions or punctuation
to meet a quota. Remove repetitive framing and empty conclusions; keep useful signposting.
Give each paragraph a purpose and connect it to its neighbours. End on a supported
consequence or a stated limit, without forcing a slogan or an unanswered question.
Prefer contemporary English syntax over literal translations of Spanish constructions.
Keep spelling, register and quotation conventions consistent. Preserve original-language
source quotations and titles exactly. Never invent experience, mistakes, quotations,
evidence or disagreements to sound human. Style samples guide the voice; their wording,
facts and citations must not be borrowed. These priorities override stylistic quotas
in a reference guide; the requested format and factual fidelity take precedence."""
    return """Escribí en español rioplatense natural, adecuado al lector y al registro.
Usá verbos precisos, sujetos concretos y transiciones que expresen la relación real
entre las ideas. Mantené estables los términos técnicos y explicalos cuando haga falta.
El argumento decide la longitud de las oraciones y la forma de cada párrafo. No fuerces
frases breves, subordinadas largas, incisos, preguntas retóricas ni signos de puntuación
para cumplir cuotas. Quitá la presentación repetitiva y los cierres vacíos; conservá
las indicaciones que orientan al lector. Cada párrafo cumple una función y enlaza con
los vecinos. Cerrá con una consecuencia fundada o un límite, sin forzar consignas.
Evitá calcos del inglés, sinónimos decorativos y enclisis arcaicas como «débese» o
«trátase» impuestas para imitar una voz. Usá tildes y signos de apertura correctamente;
si te dirigís al lector, mantené el voseo. No agregues coloquialismos a un texto académico.
Conservá literalmente las citas textuales y los títulos en su idioma original. Nunca
inventes experiencias, errores, citas, datos ni desacuerdos para parecer humano.
Las muestras orientan la voz; no se toman prestadas sus frases, datos ni referencias.
Estas prioridades prevalecen sobre las cuotas de una guía de estilo; mandan el formato
pedido y la fidelidad a los hechos."""


def style_block(seed: int | None = None, lang: str = "es") -> str:
    """The full style context injected into every drafting call."""
    global _guide_en_dead
    fp = corpus_fingerprint(lang=lang)
    if lang == "en":
        # The guide needs one PRO call over the corpus; until it has been built
        # (or while every provider is down) draft anyway without it.
        guide = "(guía de estilo no construida todavía — usá las muestras de abajo)"
        if not _guide_en_dead and (GUIDE_EN.exists() or articles_en()):
            try:
                guide = build_guide(lang="en")
            except Exception as e:  # noqa: BLE001 - a missing guide must not kill a run
                _guide_en_dead = True
                ui.log(f"[style] guía inglesa no disponible ({type(e).__name__}); "
                       "sigo con muestras y huella")
        return (f"# STYLE GUIDE (human academic English)\n{guide}\n\n"
                f"# DESCRIPTIVE FINGERPRINT (reference, not quotas)\n{json.dumps(fp, ensure_ascii=False)}\n\n"
                "# VERBATIM SAMPLES\n"
                "These are voice reference ONLY: syntax, rhythm, punctuation, ways of "
                "arguing. Citations appearing here are NOT available bibliography; the "
                "only valid bibliography is the dossier below.\n\n"
                f"{excerpts(3, 3000, seed, lang='en')}")
    return (f"# GUÍA DE ESTILO DEL AUTOR\n{build_guide()}\n\n"
            f"# HUELLA DESCRIPTIVA (referencia, no cuotas)\n{json.dumps(fp, ensure_ascii=False)}\n\n"
            "# MUESTRAS LITERALES DEL AUTOR\n"
            "Son solo referencia de VOZ: sintaxis, ritmo, puntuación, modo de argumentar. "
            "Las citas y referencias que aparezcan en estas muestras NO son bibliografía "
            "disponible y no se pueden reutilizar; la única bibliografía válida es el "
            "dossier de fuentes que viene más abajo.\n\n"
            f"{excerpts(3, 3000, seed)}")


if __name__ == "__main__":
    import sys
    print(build_guide(refresh="--refresh" in sys.argv))
