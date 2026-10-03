"""The generation pipeline: topic → research → outline → draft → humanize → review → approval.

Model routing follows the brief: PRO decides and judges (topic, review, final
approval), FLASH does the volume work (queries, outline, drafting, rewriting).

Every stage writes its output to ``output/<slug>/`` so a run can be resumed or
inspected, and so a failure at hour three does not throw away the research.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import pathlib
import random
import re
import textwrap
import unicodedata
import zlib

import humanize
import llm
import research
import style
import ui

ROOT = pathlib.Path(__file__).parent
OUTPUT = ROOT / "output"

# words: (mínimo, máximo) — un rango, no una cifra exacta: el texto está bien en
# cualquier punto del intervalo y no hay que rellenar para llegar al techo.
# sections: (mínimo, máximo) top-level parts — también un rango. El modelo decide
# cuántas y de qué tamaño (500 a 10.000 palabras cada una): a veces un capítulo es
# una parte entera y a veces un interludio de dos páginas, y forzar n secciones
# iguales es lo que producía libros de diez bloques idénticos.
# No hace falta que words[1] / sections[1] entre en una sola llamada: `_chunks()`
# parte por subsección cualquier sección de más de MAX_CALL_WORDS palabras.
# chapters: two-stage outline (arquitectura → detalle por lotes) y subsecciones
# obligatorias; hace falta arriba de ~15k palabras.
# register: the tone each format actually demands; it overrides the generic style
# rules, because a paper and an artículo de intervención are not the same voice.

# Ceiling for a single drafting call. Measured: asked for more, FLASH closes the
# section at two or three thousand words no matter what the prompt says, so past
# this a section is written subsection by subsection (`_chunks`).
MAX_CALL_WORDS = 2500

FORMATS: dict[str, dict] = {
    "corto":     {"words": (1200, 1600),   "sections": (2, 4),  "apparatus": False, "chapters": False,
                  "kind": "short intervention piece",
                  "register": "Short piece: a single thesis, direct prose, no digressions "
                              "and no apparatus. Informed-opinion register, not a paper's. "
                              "The position is clear from the start and is carried by a "
                              "concrete example, not by adjectives."},
    "medio":     {"words": (3000, 4000),   "sections": (4, 6),  "apparatus": True,  "chapters": False,
                  "kind": "theory-for-general-readers article",
                  "register": "Theory for general readers: explain every technical term the "
                              "first time it appears, one concrete example per section, "
                              "cultural-magazine prose. The reader is not a colleague: take "
                              "no school and no acronym for granted."},
    "largo":     {"words": (7000, 9000),   "sections": (5, 9),  "apparatus": True,  "chapters": False,
                  "kind": "long essay with critical apparatus",
                  "register": "Essay with critical apparatus: essayistic register, digression "
                              "is allowed if it returns to the argument, the author's voice is "
                              "visible but measured. The apparatus lives in the body of the "
                              "text, not in notes."},
    "paper":     {"words": (10000, 13000), "sections": (6, 10),  "apparatus": True,  "chapters": False,
                  "kind": "academic paper with abstract, keywords, hypothesis and conclusion",
                  "register": "Academic paper: impersonal, sober register. No irony, no "
                              "punchlines, no emphatic first person (first person only for "
                              "methodological decisions: «I surveyed», «I classified»). "
                              "Method, evidence and limits made explicit. Every strong "
                              "claim carries its citation."},
    "tesis":     {"words": (40000, 80000), "sections": (8, 24), "apparatus": True,  "chapters": True,
                  "kind": "thesis: chapters with literature review, theoretical framework, "
                          "development, discussion and conclusions",
                  "register": "Thesis: expository academic register. Each chapter states what "
                              "role it plays in the overall argument, defines its terms and "
                              "closes with what it has established. No intervention rhetoric "
                              "and no irony."},
    "libro":     {"words": (70000, 120000), "sections": (10, 40), "apparatus": True, "chapters": True,
                  "kind": "book-length theoretical essay divided into chapters",
                  "register": "Book-length essay: broad register, may narrate and give long "
                              "examples. The author's voice is visible, never topical polemic. "
                              "Each chapter stands on its own and also moves the book forward."},
    "discusion": {"words": (3500, 4500),   "sections": (4, 6),  "apparatus": True,  "chapters": False,
                  "kind": "debate: an argued confrontation between rival theoretical positions",
                  "register": "Theoretical debate: set out each rival position in its best "
                              "terms before objecting to it, and attribute to each only what it "
                              "actually holds, with its citation. Disagreement is the text's "
                              "subject, so it has to be argued, never emphatic or personal."},
}

# strftime("%B") follows the system locale, which is English on this machine;
# the author dates his pieces "Marzo de 2018".
MESES = ["", "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio",
         "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]

# The articles go out under a pen name, not the author's own. The style is his;
# the signature is not. Override with AW_FIRMA in .env.
BYLINE = os.environ.get("AW_FIRMA", "Solaris")

# Which role writes the sections. FLASH by default because a long format is
# dozens of calls, but the drafting model is the largest measured lever on how
# machine-written the text reads: over the 2026-08-24 bench, worst-judge went
# 68 (opus) / 78-82 (sonnet) / 88-93 (the hyper trio) on the same prompt and the
# same topic. Nothing in the prompt is worth 25 points; the model is.
# `AW_BORRADOR=pro` or `--drafter pro` buys that at PRO prices.
DRAFT_ROLE = os.environ.get("AW_BORRADOR", "flash").strip().lower()

# The author's interests live in a plain-text file, one area per line, so a fresh
# clone can set its own at `--setup`: interests.txt is personal (gitignored), the
# example ships with the repo and is the fallback.
INTERESTS_FILE = ROOT / "interests.txt"
INTERESTS_EXAMPLE = ROOT / "interests.example.txt"


def interest_areas() -> list[str]:
    """The author's areas: interests.txt, else the shipped example. `#` = comment."""
    for f in (INTERESTS_FILE, INTERESTS_EXAMPLE):
        if f.exists():
            areas = [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines()
                     if ln.strip() and not ln.lstrip().startswith("#")]
            if areas:
                return areas
    return []


def interests_text(areas: list[str] | None = None) -> str:
    return "; ".join(areas if areas is not None else interest_areas())


TOPIC_AREAS = 4  # areas sampled per topic round


def _area_query(area: str) -> str:
    """The news query for an area: its first term («astronáutica, astrofísica…» ->
    «astronáutica»; «filosofía de la mente: conciencia…» -> «filosofía de la mente»)."""
    return re.split(r"[,:;(]", area, 1)[0].strip()[:60]

# Article language. The prompts are written in English; `_lang()` appends the
# output-language directive. "es" (--spanish) is the author's own voice and the
# one his corpus, style guide and detector floors are measured on; "en" (the
# CLI default) is calibrated against corpus_en/. humanize swaps its tell lists,
# judge and rewrite prompts per language.
LANG = "es"


def _lang() -> str:
    if LANG == "es":
        return ("\n\nOUTPUT LANGUAGE: Write all article prose, titles, headings, abstract and "
                "keywords in Rioplatense Spanish (voseo), regardless of the language of these "
                "instructions. Keep JSON field names and enum values as specified. For APA "
                "citations use «y» between two authors, «s/f» for an undated work and «s.p.» "
                "for a missing page. Use «» as the primary quotation marks. Preserve "
                "original-language quotations and source titles.\n"
                + style.writing_rules("es"))
    return ("\n\nOUTPUT LANGUAGE: Write all article prose, titles, headings, abstract and "
            "keywords in English. Keep JSON field names and enum values as specified, even "
            "where they are Spanish. For APA citations use & between two authors in "
            "parentheses, and 'and' in narrative citations; use n.d. for an undated work. "
            "Preserve original-language quotations and source titles.\n"
            + style.writing_rules("en"))


APA_RULES = """- Cite in APA style (7th edition), with the conventions of the output language:
  · in the text, (Surname, year), or (Surname, year, p. 302) for a verbatim quote;
  · two authors, (Surname & Surname, year); three or more, (Surname et al., year);
  · for two works by the same author and year, use the key's suffix (2016a, 2016b);
  · narrative citation when the author is the subject: «Postone (2006) argues that…»;
  · never invent a page: without one, write plain (Surname, year);
  · dossier texts that come from a PDF carry interleaved «[p. N]» markers: those are
    page numbers. When you quote or paraphrase a specific passage, give the page of the
    [p. N] marker immediately before it: (Surname, year, p. N). The markers are NOT
    copied into the article, only read;
  · if a dossier key has no year, cite it exactly as it comes (Surname, n.d.): do not
    invent a year, and never put «n.d.» next to one.
- Use the dossier keys EXACTLY. One invented citation ruins the whole piece.
- EVERY mention of a study, report, index, ranking, survey or statistic carries its
  citation in the same sentence, with author and year. No exceptions.
- If something is not in the dossier, do not mention it: not the institution, not the
  consultancy, not the report's name, not its findings. Unreferenced formulas are
  forbidden: «a recent study», «according to consultancy reports», «the literature
  shows», «it is estimated that», «several studies agree». Either cited, or out.
- The same goes for figures: a percentage, an amount or a rate without (Surname, year)
  beside it is not written. If evidence is missing, omit or narrow the claim; never leave
  a placeholder in finished prose.
- Substantive author/work attributions, quotations and historical claims also need a
  supporting citation at the relevant passage. Key membership is NOT proof of support.
  Add citations only when the supplied evidence supports the claim; do not inflate counts
  or invent references, pages, dates, editions, quotations or source access."""


def wmid(spec: dict) -> int:
    """Middle of a format's word range — used for internal budgeting (context
    size, source budget), never as a target handed to the model."""
    lo, hi = spec["words"]
    return (lo + hi) // 2


def wrange(spec_or_pair) -> str:
    """«3,000 and 4,000» — how a length is stated in every prompt."""
    lo, hi = spec_or_pair["words"] if isinstance(spec_or_pair, dict) else spec_or_pair
    return f"{lo:,} and {hi:,}"


MONTHS = ("January February March April May June July August September October "
          "November December").split()


def hoy() -> str:
    """«August 21, 2026». The models have a training cutoff and, left alone,
    write about the present as if it were two years ago; every prompt that touches
    current events gets the real date."""
    d = dt.date.today()
    return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"


def slugify(text: str, maxlen: int = 60) -> str:
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    t = re.sub(r"[^a-zA-Z0-9]+", "-", t).strip("-").lower()
    if len(t) > maxlen:
        t = t[:maxlen].rsplit("-", 1)[0]
    return t or "articulo"


@dataclasses.dataclass
class Run:
    fmt: str = "medio"
    mode: str = "auto"           # auto | asistido
    brief: str = ""              # user-provided topic or thematic hint
    exact_topic: bool = False    # use `brief` verbatim as the topic, no alternatives
    ask_library: bool = True     # pause to let the user drop missing books in library/
    dir: pathlib.Path = OUTPUT
    state: dict = dataclasses.field(default_factory=dict)
    log: object = ui.log

    def save(self, name: str, data) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self.dir / name
        research.write_atomic(p, data if isinstance(data, str)
                              else json.dumps(data, ensure_ascii=False, indent=2))

    def load(self, name: str):
        p = self.dir / name
        if not p.exists():
            return None
        raw = p.read_text(encoding="utf-8")
        return json.loads(raw) if name.endswith(".json") else raw

    def ask(self, prompt: str, default: str = "") -> str:
        """Interactive gate. In auto mode it always returns the default."""
        if self.mode == "auto":
            return default
        try:
            got = input(f"\n{prompt} ").strip()
        except EOFError:
            return default
        return got or default


# --------------------------------------------------------------------------- #
# 1. Topic
# --------------------------------------------------------------------------- #

TOPIC_PROMPT = """TODAY IS {hoy}. Everything you say about current affairs must refer \
to this present, not to your training data's: if you do not see it in the headlines \
below, do not take it as true.

You are the editor of a curious, wide-ranging essay magazine. \
The author has many interests; for THIS issue, work on these areas \
(picked at random from theirs, so the magazine does not always circle the same ground):

{interests}

Their latest articles were (most recent first):
{recent}

Other titles they have already written: {written}.

{variety}

Headlines from recent weeks (to anchor in current affairs; the date comes first):
{news}

Propose 5 topics. Each must: (a) be arguable, not a descriptive overview; \
(b) have a strong, risky hypothesis that could be falsified; \
(c) rest on academic literature that actually exists; (d) have a reason why \
NOW (a headline, an anniversary, a finding, a publication; it need not be economic news).

Return JSON:
{{"temas": [{{"titulo": "...", "pregunta": "...?", "hipotesis": "...", \
"por_que_ahora": "...", "tension_teorica": "what debate it opens and against whom", \
"autores_clave": ["..."], "obras_necesarias": ["Author, Title (year)"], \
"riesgo": "why it could fail"}}]}}"""


DEVELOP_PROMPT = """TODAY IS {hoy}. Current affairs are those of this date, not of your \
training data.

The author has already chosen the topic and wants no alternatives. Your job is to \
turn it into a working object: sharpen it, give it a hypothesis and find the debate.

TOPIC REQUESTED BY THE AUTHOR (respect it, do not change the subject): «{brief}»

The author's field: {interests}

Recent headlines, in case the topic has a current-affairs anchor:
{news}

Return JSON:
{{"titulo": "title in the author's style, true to the requested topic",
  "pregunta": "...?", "hipotesis": "strong, falsifiable hypothesis",
  "por_que_ahora": "...", "tension_teorica": "what debate it opens and against whom",
  "autores_clave": ["..."], "obras_necesarias": ["Author, Title (year)"],
  "riesgo": "why it could fail"}}"""


VARIETY_RULES = """VARIETY RULES (mandatory):
- No topic may return to the subject, thesis or angle of an article already written, \
even under a new title.
- The 5 topics must come from different areas; each area above may appear in at most \
one topic.
- Political economy, value critique and Marxism may be, at most, a lens in ONE topic, \
not the axis of all of them. An article on dinosaurs, aircraft, languages or video \
games stands on its own: it does not need to become a critique of capital.
- Be creative: prefer odd, precise, surprising questions to predictable overviews."""


def already_written(root: pathlib.Path | None = None, limit: int = 60) -> list[str]:
    """Titles of previous runs, newest first, so the picker does not repeat itself.

    Only matters in continuous mode, where nobody is watching the topic list.
    """
    root = root if root is not None else OUTPUT
    if not root.exists():
        return []
    out = []
    for d in sorted(root.iterdir(), reverse=True):
        f = d / "01_topic.json" if d.is_dir() else None
        if f is None or not f.exists():
            continue
        try:
            out.append(json.loads(f.read_text(encoding="utf-8"))["titulo"])
        except (json.JSONDecodeError, KeyError, OSError):
            continue
        if len(out) >= limit:
            break
    return out


def pick_topic(run: Run) -> dict:
    if (cached := run.load("01_topic.json")):
        run.log("[topic] reusing 01_topic.json")
        return cached
    if run.exact_topic and run.brief:
        return _develop_topic(run)
    # A few areas per run, not the whole list: handed everything, the model gravitates
    # to the same political-economy corner every time.
    # A --topic brief replaces the sampled areas and the variety rules: those rules cap
    # value critique at one lens, which silently vetoed briefs on value critique.
    areas = interest_areas()
    focus = [run.brief] if run.brief else random.sample(areas, min(TOPIC_AREAS, len(areas)))
    run.log(f"[topic] areas for this issue: {'; '.join(_area_query(a) for a in focus)}")
    run.log("[topic] scanning the news…")
    news = []
    # The generic Argentina query only suits free picks: under a brief its football and
    # politics headlines pulled the topic off the requested subject.
    for q in [_area_query(a) for a in focus] + ([] if run.brief else ["Argentina"]):
        news += research.google_news(q, 5, "es", days=30)
        news += research.google_news(q, 4, "en", days=30)
    news.sort(key=lambda n: n.date, reverse=True)
    headlines = "\n".join(f"- {n.date or 'n.d.'} · {n.title} ({n.venue})"
                          for n in news[:45]) or "(no data)"
    # Own past runs first: in continuous mode the acute risk is repeating itself.
    # The most recent ones are listed apart so a truncated `written` cannot drop them.
    past = already_written()
    recent = "\n".join(f"- {t}" for t in past[:12]) or "(none)"
    written = ", ".join(past[12:] + [p.stem.replace("-", " ") for p in style.CORPUS.glob("*.md")])
    variety = (f"TOPIC REQUESTED BY THE AUTHOR (respect it, do not change the subject): "
               f"«{run.brief}». All 5 topics must develop THAT request, each from a "
               "different angle, and none may repeat an article already written."
               ) if run.brief else VARIETY_RULES
    data = llm.chat_json(llm.PRO, TOPIC_PROMPT.format(
        hoy=hoy(), interests="\n".join(f"- {a}" for a in focus), recent=recent,
        written=written[:3000], news=headlines,
        variety=variety) + _lang(), temperature=1.0)
    temas = data.get("temas", [])
    # In auto mode nobody picks, so listing them is noise; and it takes one at random,
    # since "always the first" is always the model's most predictable idea.
    for i, t in enumerate(temas if run.mode != "auto" else [], 1):
        run.log(f"\n  [{i}] {t['titulo']}\n      hypothesis: {t['hipotesis']}\n"
                f"      why now: {t.get('por_que_ahora','')}")
    default = str(random.randint(1, len(temas))) if run.mode == "auto" and temas else "1"
    choice = run.ask(f"Pick a topic [1-{len(temas)}] (Enter = 1, or type your own title):", default)
    if choice.isdigit() and 1 <= int(choice) <= len(temas):
        topic = temas[int(choice) - 1]
    else:
        # A typed title still gets a hypothesis and a bibliography plan built for it.
        run.brief = choice
        return _develop_topic(run)
    return _settle_topic(run, topic)


def _develop_topic(run: Run) -> dict:
    """Turn the user's own topic into a workable object without proposing others."""
    run.log(f"[topic] developing the requested topic: «{run.brief}»")
    run.log("[topic] looking for recent news…")
    news = (research.google_news(run.brief, 8, "es", days=45)
            + research.google_news(run.brief, 5, "en", days=45))
    topic = llm.chat_json(llm.PRO, DEVELOP_PROMPT.format(
        hoy=hoy(), brief=run.brief, interests=interests_text(),
        news="\n".join(f"- {n.date or 'n.d.'} · {n.title} ({n.venue})"
                       for n in news[:25]) or "(no data)") + _lang(),
        temperature=0.85)
    run.log(f"  hypothesis: {topic.get('hipotesis','')}")
    if run.mode != "auto":
        # Loop: every piece of feedback regenerates the topic, and the new
        # hypothesis is shown again for validation until Enter accepts it.
        while tweak := run.ask("Adjust the hypothesis or the title? (Enter = it is fine):", ""):
            topic = llm.chat_json(llm.PRO,
                f"Adjust this topic as requested and return the same JSON, corrected.\n"
                f"REQUEST: {tweak}\n\n{json.dumps(topic, ensure_ascii=False)}"
                f"{_lang()}", temperature=0.7)
            run.log(f"\n[topic] title: «{topic.get('titulo', '')}»\n"
                    f"  hypothesis: {topic.get('hipotesis', '')}")
    return _settle_topic(run, topic)


def _settle_topic(run: Run, topic: dict) -> dict:
    run.dir = OUTPUT / f"{dt.date.today():%Y%m%d}-{slugify(topic['titulo'])}"
    run.save("00_corrida.json", {"fmt": run.fmt, "lang": LANG})
    run.save("01_topic.json", topic)
    run.log(f"\n[topic] «{topic['titulo']}»\n[topic] folder: {run.dir}")
    return topic


# --------------------------------------------------------------------------- #
# 2. Research
# --------------------------------------------------------------------------- #

QUERIES_PROMPT = """Topic: {titulo}
Question: {pregunta}
Hypothesis: {hipotesis}
Key authors: {autores}

Write search queries to gather real literature on this.

IMPORTANT: the academic databases (OpenAlex, Crossref, DOAJ, Semantic Scholar) AND \
every term together. A nine-word query returns nothing. Write queries of 2 to 5 \
words, using the field's technical terms, with no connectives and no full sentences. \
Bad: «achievable postwork utopias without abolishing value». Good: «postwork utopia», \
«abolición del trabajo», «value-form critique».

Return JSON:
{{"academicas_es": ["6 queries in Spanish, 2-5 words each"],
  "academicas_en": ["8 queries in English, 2-5 words, technical terminology"],
  "prensa": ["4 current-affairs queries in Spanish"],
  "prensa_en": ["3 current-affairs queries in English"],
  "libros": ["6 specific works: «Author, Title» that must be obtained"],
  "contraargumentos": ["3 short queries for literature that REFUTES the hypothesis"]}}"""


def do_research(run: Run, topic: dict) -> tuple[list[research.Source], list[dict]]:
    """Research in three checkpointed substeps: gather, full texts, books.

    Each one takes minutes of network and is written to disk as soon as it ends,
    so a Ctrl-C (or a dead mirror) resumes where it stopped instead of paying for
    the whole stage again. ``02_estado.json`` says which substeps already ran —
    ``02_dossier.json`` alone cannot, since it is now saved half-built.
    """
    dossier = run.dir / "02_dossier.json"
    done = set(run.load("02_estado.json") or [])
    sources = research.load_dossier(dossier) if dossier.exists() else []
    if "libros" in done:
        run.log("[research] reusing 02_dossier.json")
        return sources, run.load("02_faltantes.json") or []

    def keep(step: str) -> None:
        research.save_dossier(sources, dossier)
        done.add(step)
        run.save("02_estado.json", sorted(done))

    plan = run.load("02_plan.json") or llm.chat_json(llm.FLASH, QUERIES_PROMPT.format(
        titulo=topic["titulo"], pregunta=topic.get("pregunta", ""),
        hipotesis=topic.get("hipotesis", ""),
        autores=", ".join(topic.get("autores_clave", []))) + _lang(), temperature=0.6)
    run.save("02_plan.json", plan)
    queries = (plan.get("academicas_es", []) + plan.get("academicas_en", [])
               + plan.get("contraargumentos", []))
    wanted = plan.get("libros", []) + topic.get("obras_necesarias", [])
    run.log(f"[research] {len(queries)} academic queries + news + "
            f"{len(wanted)} works searched by title")
    if "fuentes" in done:
        run.log(f"[research] reusing {len(sources)} sources already gathered")
    else:
        sources = research.gather(queries, news_queries=plan.get("prensa", []),
                                  news_queries_en=plan.get("prensa_en", []),
                                  book_titles=wanted, log=run.log)
        keep("fuentes")
    if "textos" in done:
        run.log("[research] sources: full texts already downloaded")
    else:
        run.log(f"[research] sources: downloading the full text of the "
                f"{len(sources)} dossier sources…")
        budget = 20 if wmid(FORMATS[run.fmt]) > 8000 else 10
        research.enrich_fulltext(sources, budget=budget, log=run.log)
        keep("textos")
    gaps = research.missing_books(sources, wanted, log=run.log)
    run.save("02_faltantes.json", gaps)
    keep("libros")
    if gaps:
        sources = _library_gate(run, sources, gaps)
        keep("libros")
    return sources, gaps


def _library_gate(run: Run, sources: list[research.Source],
                  gaps: list[dict]) -> list[research.Source]:
    """Show what could not be downloaded and let the user supply it by hand.

    Runs up to three times: each round rescans ``library/`` and reports which
    works are still missing. In auto mode it only prints the report and moves on
    — an unattended run must never block on a prompt.
    """
    research.LIBRARY.mkdir(exist_ok=True)
    if not run.ask_library:
        run.log(f"[bibliography] {len(gaps)} work(s) missing; carrying on without asking.")
        return sources
    for attempt in range(3):
        run.log(f"\n[bibliography] full text missing for {len(gaps)} work(s):")
        for g in gaps:
            run.log(f"\n  · {g['wanted']}")
            for f in g.get("fallbacks", [])[:4]:
                run.log(f"      {f['title']:<26} {f['url']}")
        if run.mode == "auto":
            run.log("\n[bibliography] auto mode: carrying on with what there is "
                    "(the missing works will not be cited).")
            return sources
        run.log(f"\n  Folder: {research.LIBRARY}")
        answer = run.ask("Drop the files there (epub, pdf, docx, txt, html) and press Enter. "
                         "Type 'skip' to go on without them, or paste a URL:", "skip")
        if answer.startswith("http"):
            body = research.fetch_text(answer, 200000)
            if len(body) > 1000:
                sources.append(research.Source(
                    title=answer.rsplit("/", 1)[-1][:120] or "supplied source",
                    authors=[], year="", kind="local", venue="supplied by the author",
                    url=answer, abstract=body[:2000], fulltext=body, origin="usuario"))
                run.log(f"  · downloaded: {len(body.split())} words")
            else:
                run.log("  · could not extract text from that URL")
        extra = [s for s in research.scan_library()
                 if s.url not in {x.url for x in sources}]
        if extra:
            sources += extra
            run.log(f"[bibliography] local library: +{len(extra)} works "
                    f"({sum(len(s.fulltext.split()) for s in extra)} words)")
        sources = research.assign_keys(research.dedupe(sources))
        research.save_dossier(sources, run.dir / "02_dossier.json")
        gaps = [g for g in gaps if not research._have_title(sources, g["wanted"])]
        run.save("02_faltantes.json", gaps)
        if not gaps:
            run.log("[bibliography] complete.")
            return sources
        if answer.lower().startswith(("skip", "seguir")) and not extra and attempt:
            break
    run.log(f"[bibliography] carrying on without {len(gaps)} work(s); they will not be cited.")
    return sources


# --------------------------------------------------------------------------- #
# 3. Outline
# --------------------------------------------------------------------------- #

OUTLINE_PROMPT = """TODAY IS {hoy}: the text's present is this date.

You are planning a {kind} of between {words} words.
That is a RANGE, not a target: any length inside it is fine.

THE FORMAT'S REGISTER (it overrides every other consideration of tone):
{registro}

TITLE: {titulo}
QUESTION: {pregunta}
HYPOTHESIS TO DEFEND: {hipotesis}
THEORETICAL TENSION: {tension}

AVAILABLE SOURCES (cite ONLY with these keys; no other source exists):
{catalogo}

Design between {nmin} and {nmax} sections — how many, and how long each one is, is your \
call, according to what the argument needs. A section can run from 500 to 10,000 words: \
some need long development and others are a short stretch that turns the argument. Do \
not make them all the same length. Each section must move the argument forward, not \
repeat it. Give each one the keys of its sources and a word budget; the budgets must \
add up to a total inside the range {words}.
{chapters_note}

Return JSON:
{{"titulo_final": "final title, in the author's style",
  "resumen": "120-word abstract",
  "palabras_clave": ["6 terms"],
  "secciones": [{{"n": 1, "titulo": "...", "tesis": "what this section argues",
                  "contenido": "what it develops, in 3-5 lines",
                  "fuentes": ["Key, year", "..."], "palabras": 000,
                  "subsecciones": ["..."]}}]}}"""


# Long formats get the outline in two passes. One call cannot plan 40 units in
# detail: the JSON runs past the output limit and comes back truncated, and what
# does arrive is forty interchangeable blocks. First the architecture (titles,
# function, word budget), then the detail in batches, each batch seeing the whole
# architecture so it knows what the other units already cover.
PLAN_PROMPT = """TODAY IS {hoy}: the text's present is this date.

You are designing the ARCHITECTURE of a {kind} of between {words} words.
That is a RANGE, not a target.

THE FORMAT'S REGISTER (it overrides every other consideration of tone):
{registro}

TITLE: {titulo}
QUESTION: {pregunta}
HYPOTHESIS TO DEFEND: {hipotesis}
THEORETICAL TENSION: {tension}

AVAILABLE SOURCES (everything that can be cited comes from here):
{catalogo}

The structure is your call: between {nmin} and {nmax} units, each of 500 to 10,000 \
words. A unit can be a long chapter, a part, a short section, an excursus or an \
interlude: use the form the argument asks for and do NOT split the words into equal \
parts. You may group consecutive units under one part or larger chapter (say so in \
«parte»). The budgets must add up to a total inside the range {words}.

Return JSON:
{{"titulo_final": "final title, in the author's style",
  "resumen": "120-word abstract",
  "palabras_clave": ["6 terms"],
  "estructura": [{{"n": 1, "parte": "chapter or part it belongs to, or \\"\\"",
                   "titulo": "...", "funcion": "what it does in the overall argument, 1-2 lines",
                   "palabras": 000}}]}}"""

DETAIL_PROMPT = """You are detailing the outline of a {kind} titled «{titulo}».

OVERALL HYPOTHESIS: {hipotesis}
REGISTER: {registro}

FULL ARCHITECTURE (so you know what each unit covers and do not overlap them):
{estructura}

AVAILABLE SOURCES (cite ONLY with these keys; no other source exists):
{catalogo}

Detail ONLY units number {ns}, keeping their title, function and word budget. If a \
unit runs past {maxcall} words, give it subsections with their own titles (one every \
1,000-2,500 words); if it is short, leave "subsecciones": [].

Return JSON:
{{"secciones": [{{"n": 1, "titulo": "...", "tesis": "what this unit argues",
                  "contenido": "what it develops, in 3-5 lines",
                  "fuentes": ["Key, year", "..."], "palabras": 000,
                  "subsecciones": ["..."]}}]}}"""


def _plan_outline(run: Run, spec: dict, topic: dict, catalogo: str) -> dict:
    """Architecture first, then detail in batches of 6 units."""
    nmin, nmax = spec["sections"]
    plan = run.load("03_plan.json") or llm.chat_json(llm.FLASH, PLAN_PROMPT.format(
        hoy=hoy(), kind=spec["kind"], words=wrange(spec), registro=spec["register"],
        titulo=topic["titulo"], pregunta=topic.get("pregunta", ""),
        hipotesis=topic.get("hipotesis", ""), tension=topic.get("tension_teorica", ""),
        catalogo=catalogo, nmin=nmin, nmax=nmax) + _lang(), temperature=0.7)
    run.save("03_plan.json", plan)
    units = plan["estructura"]
    for i, u in enumerate(units, 1):
        u["n"] = i
    run.log(f"[outline] architecture: {len(units)} units, "
            f"{sum(int(u.get('palabras') or 0) for u in units)}w planned")
    esquema = "\n".join(
        f"{u['n']}. {('[' + u['parte'] + '] ') if u.get('parte') else ''}{u['titulo']} "
        f"({u.get('palabras')}w) — {u.get('funcion', '')}" for u in units)

    secciones: list[dict] = []
    for i in range(0, len(units), 6):
        batch = units[i:i + 6]
        run.log(f"[outline] detailing units {batch[0]['n']}-{batch[-1]['n']}…")
        got = llm.chat_json(llm.FLASH, DETAIL_PROMPT.format(
            kind=spec["kind"], titulo=plan.get("titulo_final", topic["titulo"]),
            hipotesis=topic.get("hipotesis", ""), registro=spec["register"],
            estructura=esquema, catalogo=catalogo, maxcall=MAX_CALL_WORDS,
            ns=", ".join(str(u["n"]) for u in batch)) + _lang(), temperature=0.6)
        by_n = {int(s["n"]): s for s in got.get("secciones", []) if s.get("n")}
        for u in batch:  # a unit the model skipped still gets written, from the plan
            sec = by_n.get(u["n"], {"titulo": u["titulo"], "tesis": u.get("funcion", ""),
                                    "contenido": u.get("funcion", ""), "fuentes": [],
                                    "subsecciones": []})
            sec["n"], sec["palabras"] = u["n"], sec.get("palabras") or u.get("palabras") or 1500
            secciones.append(sec)
    return {"titulo_final": plan.get("titulo_final", topic["titulo"]),
            "resumen": plan.get("resumen", ""),
            "palabras_clave": plan.get("palabras_clave", []),
            "secciones": secciones}


def make_outline(run: Run, topic: dict, sources: list[research.Source]) -> dict:
    if (cached := run.load("03_outline.json")):
        run.log("[outline] reusing 03_outline.json")
        return cached
    spec = FORMATS[run.fmt]
    catalogo = "\n".join(
        f"- [{s.key}] {s.title[:110]} ({s.kind}{', ' + s.date if s.date else ''})"
        f" — {(s.abstract or '')[:170]}"
        for s in sources[:160])
    if spec["chapters"]:
        outline = _plan_outline(run, spec, topic, catalogo)
    else:
        nmin, nmax = spec["sections"]
        run.log(f"[outline] asking for the full outline ({nmin}-{nmax} sections)…")
        outline = llm.chat_json(llm.FLASH, OUTLINE_PROMPT.format(
            hoy=hoy(), kind=spec["kind"], words=wrange(spec), registro=spec["register"],
            nmin=nmin, nmax=nmax,
            titulo=topic["titulo"], pregunta=topic.get("pregunta", ""),
            hipotesis=topic.get("hipotesis", ""), tension=topic.get("tension_teorica", ""),
            catalogo=catalogo,
            chapters_note=f"If a section runs past {MAX_CALL_WORDS} words, give it "
                          "subsections with their own titles; otherwise leave «subsecciones»: []."
            ) + _lang(), temperature=0.7)

    def show(o: dict) -> None:
        run.log(f"\n[outline] {o.get('titulo_final')}")
        for s in o["secciones"]:
            run.log(f"  {s['n']}. {s['titulo']} ({s['palabras']}w)"
                    f" — {len(s.get('fuentes', []))} sources")

    show(outline)
    # PRO audits the plan before any words get written; a bad outline is the
    # most expensive thing to discover after 30k words of drafting.
    run.log("[outline] PRO audits the outline…")
    critique = llm.chat_json(llm.PRO, f"""Audit this outline of a {spec['kind']}.
Does the argument progress or repeat itself? Is the hypothesis supported or merely stated?
Are there hollow sections? Is the counterargument missing? Is the allocation of sources plausible?

{json.dumps(outline, ensure_ascii=False, indent=2)}

Return JSON: {{"veredicto": "aprobado"|"corregir", "problemas": ["..."], \
"esquema_corregido": <the same outline JSON object, corrected>}}""" + _lang(),
temperature=0.4)
    # PRO sometimes answers a bare list: an unusable audit is no audit, keep the outline
    if not isinstance(critique, dict):
        critique = {}
    fixed = critique.get("esquema_corregido") or {}
    # A long outline handed back whole comes back truncated — half the sections
    # silently dropped. Take the correction only if it still has them all.
    if critique.get("veredicto") == "corregir" and \
            len(fixed.get("secciones") or []) >= len(outline["secciones"]):
        run.log("[outline] PRO corrected: " + "; ".join(critique.get("problemas", [])[:3]))
        outline = fixed
        show(outline)
    elif critique.get("veredicto") == "corregir":
        run.log("[outline] PRO objected but returned an incomplete outline; keeping mine: "
                + "; ".join(critique.get("problemas", [])[:3]))
    while run.mode != "auto":
        edit = run.ask("Changes to the outline? (Enter = continue, or describe what to change):", "")
        if not edit:
            break
        outline = llm.chat_json(llm.FLASH,
            f"Apply these changes to the outline and return the full corrected JSON.\n"
            f"CHANGES: {edit}\n\nOUTLINE:\n{json.dumps(outline, ensure_ascii=False)}"
            f"{_lang()}", temperature=0.5)
        show(outline)
    run.save("03_outline.json", outline)
    return outline


# --------------------------------------------------------------------------- #
# 4. Draft
# --------------------------------------------------------------------------- #

SECTION_PROMPT = """{style_block}

# ASSIGNMENT
You are writing section {n} of {total} of a {kind} titled «{titulo}».
OVERALL HYPOTHESIS OF THE WORK: {hipotesis}

## Section to write
Title: {sec_titulo}
Section thesis: {sec_tesis}
It must develop: {sec_contenido}
{subsecciones}
Length: between {pmin} and {pmax} words. That is a range, not a target: anywhere in it
is fine. Do not pad to reach the ceiling, and do not stop short of the floor.

## THE FORMAT'S REGISTER — it overrides everything else
{registro}
If any style rule below clashes with this register, the register wins.
Indicative citation density: {min_citas} distinct citations, only if there are enough
relevant sources. Each citation supports a concrete claim; do not cite a source that
barely touches the subject, and do not repeat the reference after every sentence of
one passage.
DOCUMENTARY DENSITY: at least {min_datos} facts about the world —exact dates
(«September 15, 1970», not «in the early seventies»), figures with their unit, proper
names of people WITH THEIR POST OR ROLE («Richard Helms, director of the CIA»),
institutions, places, exact titles of works, page numbers— all taken from the dossier.
A citation is not a fact: «(Postone, 1993)» does not count, «the 4.7 million tonnes
Postone (1993, p. 287) takes from the 1968 census» does. This is what separates a
documented text from one that merely administers a bibliography, and it is the first
thing a reader notices. If the dossier has no hard facts on a point, write less about it.

## What has already been said (do not repeat it; pick it up if needed)
{previo}

## Formulas already used — FORBIDDEN to write them again
{evitar}
No image, comparison, anecdote, rhetorical question or punchline may appear twice in
the whole work. If it has been used, use something else or nothing. The same goes for
uncommon words: if one has already carried a sentence, find another.

## SOURCES — this is EVERYTHING that exists. You cannot cite anything not here.
{fuentes}

# RULES
- Write in the requested language, in the voice of the matching samples.
{apa}
- Verbatim quotations over 40 words go in a separate block, without quotation marks,
  and stay in their original language if the source is in English, German or French.
- No bullet lists, no bold, no summaries at the end of the section.
- Each section moves the argument forward from what has been established. Choose the
  opening by its function: a documented case, a precise question or a thesis. Do not
  repeat one opening-development-close template in every section.
- Quote verbatim only when the exact wording matters and the passage is available in
  the sources. Never invent a quotation to meet a quota.
- Sentence and paragraph length follow the idea. Keep useful transitions and cut the
  sentences that only announce or repeat what the reader already knows.
- TAKE A POSITION, NOT A FIGHT. The text holds a thesis and shows it, but does not go
  looking for antagonists:
  · name the position you dispute precisely (a school, a hypothesis, a methodological
    assumption) and attribute it to whoever actually holds it, with the citation; no
    «a certain discourse» or «the system» in the abstract, and no portraits of authors
    as naive, complicit or snake-oil sellers;
  · no irony at third parties, no sarcasm, nicknames or punchlines («selling smoke»,
    «statistical astrology», «with the elegance of someone crossing a bridge without
    looking at the river»). A precise image is worth more than a sharp one, and if it
    adds nothing, it goes;
  · take sides with arguments, not adjectives: the force comes from the fact and the
    consequence that follows, not from the tone;
  · when your own argument has a limit, say so once, calmly, and move on. The
    concession is part of the argument, not a confession;
  · a brief critical remark now and then, not in every paragraph. More than one
    objection to an author on a page is too many.
- Use punctuation and syntax natural to the requested language. Do not force a quota
  of long or short sentences, and do not avoid a mark that clarifies the argument.
- TODAY IS {hoy}. You write from this date: current affairs are today's, not those of
  the year the model was trained. When the argument touches politics, economics, war
  or technology, use the MOST RECENT figure in the sources (they carry the date in
  brackets), cite it with its date —«on August 14», «the July figures»— and never cite
  an old piece when there is a newer one on the same point. No bare «recently».
  Do not invent what happened after the latest source either.
- If a fact is not in the sources, do not assert it. You may mark a gap with [to verify].
{aparato}
- Start directly with the section's text, with no heading or title.

Write it now."""


# Measured 2026-08-25 against six real windows of the author's corpus: what a
# published article has and a generated one never does is not rhythm, it is
# *machinery*. His prose carries footnotes with digressions and bibliographic
# strings, editorial brackets inside quotations, an English gloss in
# parentheses, two quote styles doing two different jobs, exact times of day.
# The draft comes out immaculate — one dash policy, one quote style, no notes,
# no visible seams — and immaculate reads as machine-written. None of this is
# invented: every note has to be dossier-backed like any other claim.
APPARATUS_RULES = """- FOOTNOTE APPARATUS. This section may carry up to {notas_max} footnotes, only
  when they add a necessary precision that would interrupt the main argument. They are
  marked in the body with the number in brackets, attached to the word or punctuation
  mark —like this[{nota_1}]— and this section's numbering STARTS AT {nota_1} and goes
  on from there. At the very end of the section's text, after a line that says exactly
  NOTES:, write one note per paragraph, in the form «[{nota_1}] text of the note.». A
  footnote is NOT a stray citation: it is what does not fit the argument and is still
  needed —the precision of a date, the history of a translation, the side fact, why a
  source says what it says, a minor disagreement, a document's full reference. They
  carry their citations like everything else.
- Keep quotations verbatim. Use editorial brackets only when an omission or
  clarification is necessary and faithful to the passage, never to fake documentary work.
- Use the typographic conventions of the article's language. Figures, technical terms
  and titles keep their precision; do not add ornaments for variety.
- FINISHED PROSE ONLY. Keep source-access limitations, dossier/excerpt availability,
  copyright-policy refusals and editorial instructions in private review reports, never
  in the article or its footnotes. Legitimate scholarly discussion of copyright, editions
  and methodological limitations is allowed when supported. Do not describe which sources
  the writing process could access. Never add process notes such as «only the first chapter of Kurz (1998) is to
  hand, and what follows rests on it»."""


def _sources_block(sources: list[research.Source], keys: list[str], chars: int) -> str:
    by_key = {s.key: s for s in sources}
    picked = [by_key[k] for k in keys if k in by_key]
    if len(picked) < 4:  # outline under-assigned; top up with the richest sources
        extra = sorted((s for s in sources if s.key not in {p.key for p in picked}),
                       key=lambda s: -len(s.fulltext or s.abstract))
        picked += extra[:6 - len(picked)]
    budget = max(600, chars // max(len(picked), 1))
    return "\n\n".join(p.brief(budget) for p in picked)


def _banned(parts: list[str], limit: int = 30) -> str:
    """Formulas the next section is not allowed to serve again.

    Short sentences are the ones that come back verbatim three sections later —
    they are the "sentence that lands" the style asks for, and the model reuses
    its own best one. Listing them is cheaper than hoping it remembers.
    """
    if not parts:
        return "(nothing written yet)"
    text = "\n".join(l for l in "\n\n".join(parts).split("\n") if not l.startswith("#"))
    short = [s.strip(" \n«»") for s in re.split(r"(?<=[.?!])\s+", text)
             if 3 <= len(s.split()) <= 12]
    phrases = list(dict.fromkeys(humanize.repeated_phrases(text) + short))
    return "\n".join(f"- «{p}»" for p in phrases[:limit])


def _chunks(sec: dict) -> list[tuple[str, str, int, str]]:
    """One drafting call each: (cache suffix, subtitle, words, heading).

    A section over MAX_CALL_WORDS is written subsection by subsection, and a
    subsection still over it, in parts. Asking for 8.000 words in one call does
    not return 8.000 words: the model closes at two or three thousand and the
    section arrives half written, so a book budgeted in long chapters came out at
    a third of its length.
    """
    subs = [str(s).strip() for s in sec.get("subsecciones") or [] if str(s).strip()]
    words = int(sec.get("palabras") or 1500)
    if words <= MAX_CALL_WORDS or not subs:
        return [(f"{int(sec['n']):02d}", "", words, "")]
    each = max(400, round(words / len(subs)))
    out: list[tuple[str, str, int, str]] = []
    for i, sub in enumerate(subs, 1):
        n_parts = -(-each // MAX_CALL_WORDS)          # ceil
        for j in range(1, n_parts + 1):
            suffix = f"{int(sec['n']):02d}_{i:02d}" + ("" if n_parts == 1 else chr(96 + j))
            label = sub if n_parts == 1 else f"{sub} (part {j} of {n_parts})"
            out.append((suffix, label, round(each / n_parts),
                        f"### {sub}" if j == 1 else ""))
    return out


def draft(run: Run, topic: dict, outline: dict, sources: list[research.Source]) -> str:
    spec = FORMATS[run.fmt]
    # crc32, not hash(): str hashing is salted per process, so a resumed run would
    # draft against different style excerpts than the run it is resuming.
    sb = style.style_block(seed=zlib.crc32(topic["titulo"].encode()) % 1000,
                           lang=LANG)
    parts: list[str] = []
    synopsis = ""
    secciones = outline["secciones"]
    ctx_chars = 24000 if wmid(spec) > 8000 else 14000
    # The rolling synopsis costs one FLASH call per section and only ever feeds a
    # section that still has to be written. On a resumed run every chunk is on
    # disk, so those calls bought nothing — and the last section always paid for
    # a summary no one reads. `pending` is what is still missing ahead.
    pending = [not (run.dir / f"04_sec{sfx}.md").exists()
               for sec in secciones for sfx, *_ in _chunks(sec)]
    # The blanket pass in do_research walks the dossier in gather order and
    # stops after budget*2 candidates; the sources an outline ends up citing
    # usually sit past that cut and reached the draft as abstracts only. Here,
    # with the outline's per-section `fuentes` in hand, each one gets a last
    # targeted try before any section is written against it.
    if any(pending):
        wanted_keys: set[str] = set()
        i = 0
        for sec in secciones:
            n = len(_chunks(sec))
            if any(pending[i:i + n]):
                wanted_keys.update(sec.get("fuentes", []))
            i += n
        todo = [s for s in sources if s.key in wanted_keys
                and not s.fulltext and (s.doi or s.url)]
        if todo:
            run.log(f"[draft] downloading the full text of {len(todo)} "
                    "cited source(s) that arrived without one…")
            research.enrich_fulltext(sources, log=run.log, keys=wanted_keys)
            research.save_dossier(sources, run.dir / "02_dossier.json")
    seen = 0
    notas: list[str] = []          # footnote bodies, in the order they were written
    for sec in secciones:
        chunks = _chunks(sec)
        seen += len(chunks)
        bodies: list[str] = []
        for suffix, sub, words, head in chunks:
            cache = run.dir / f"04_sec{suffix}.md"
            label = f"{sec['titulo']} — {sub}" if sub else sec["titulo"]
            if cache.exists():
                body = cache.read_text(encoding="utf-8")
                run.log(f"[draft] {label} cached ({len(body.split())}w)")
            else:
                run.log(f"[draft] {ui.progress_bar(int(sec['n']), len(secciones))} {label} "
                        f"({words}w)…")
                if sub:  # writing one subsection: show the others so it does not cover them
                    subs = ("Subsections of this section — write ONLY the one marked →; "
                            "someone else writes the others, do not anticipate them:\n"
                            + "\n".join(f"  {'→' if x == sub else ' '} {x}"
                                        for _, x, _, _ in chunks))
                else:
                    subs = ("Subsections (with their titles in the text):\n"
                            + "\n".join(f"  - {x}" for x in sec["subsecciones"])
                            ) if sec.get("subsecciones") else ""
                # Inside a section the rolling synopsis is still one entry behind,
                # so the tail of the previous chunk is what makes the seam invisible.
                previo = synopsis or "(this is the first section)"
                if bodies:
                    previo += ("\n\n(The last lines written in this same section, to "
                               f"splice onto: …{' '.join(bodies[-1].split()[-120:])})")
                prompt = (SECTION_PROMPT.format(
                    style_block=sb, hoy=hoy(), n=sec["n"], total=len(secciones),
                    kind=spec["kind"],
                    titulo=outline.get("titulo_final", topic["titulo"]),
                    hipotesis=topic.get("hipotesis", ""), sec_titulo=label,
                    sec_tesis=sec.get("tesis", ""), sec_contenido=sec.get("contenido", ""),
                    subsecciones=subs, registro=spec["register"],
                    pmin=int(words * 0.9), pmax=int(words * 1.1),
                    apa=APA_RULES, min_citas=max(2, words // 180),
                    min_datos=max(4, words // 150),
                    aparato=_apparatus_rules(spec, words, len(notas) + 1),
                    previo=previo, evitar=_banned(parts + bodies),
                    fuentes=_sources_block(sources, sec.get("fuentes", []), ctx_chars))
                    + _lang())
                body = _draft_chunk(run, prompt, words, label)
                research.write_atomic(cache, body)
            # Cached chunks are split too: the numbering of the *next* section
            # depends on how many notes came before it, so a resume has to count
            # them the same way the original run did.
            body, mine = _split_notes(body)
            notas += mine
            bodies.append(f"{head}\n\n{body}" if head else body)
        parts.append(f"## {sec['titulo']}\n\n" + "\n\n".join(bodies))
        if any(pending[seen:]):
            synopsis = _update_synopsis(synopsis, sec, "\n\n".join(bodies))
    if notas:
        parts.append(("## Notes" if LANG == "en" else "## Notas") + "\n\n" + "\n\n".join(notas))
    return "\n\n".join(parts)


DRAFT_TRIES = 3   # replies asked of each link before the chunk moves to the next one


def _draft_chunk(run: Run, prompt: str, words: int, label: str) -> str:
    """One drafted chunk, retried until it is usable.

    A stub (a free-tier model answering 12 words, twice) is a bad link, not a
    bad section: after DRAFT_TRIES replies the chunk is handed to each backup in
    the chain, head first. Only when every link has failed does it raise —
    there is nothing left to cache, and an amputated section must never be.
    """
    best = None
    chain = list(llm.CHAIN)
    try:
        for k in range(max(1, len(chain))):
            rotated = chain[k:] + chain[:k]
            llm.CHAIN = rotated
            if k:
                run.log(f"[draft] {label}: handing the section to {rotated[0]}")
            ask = prompt
            for attempt in range(DRAFT_TRIES):
                body = _strip_preamble(llm.chat(
                    llm.PRO if DRAFT_ROLE == "pro" else llm.FLASH, ask, temperature=0.9))
                count = len(body.split())
                right_lang = style.detect_language(body) in (None, LANG)
                if 0.9 * words <= count <= 1.1 * words and right_lang:
                    return body
                if right_lang and (best is None or abs(count - words) < abs(len(best.split()) - words)):
                    best = body
                # The ±10% asked for is a target, not a gate: killing an
                # hours-long run because a section came back at 1.2x is a
                # worse outcome than keeping it. Only a section that is
                # plainly amputated or in the wrong language keeps retrying.
                if attempt and best is not None and 0.7 * words <= len(best.split()) <= 2 * words:
                    run.log(f"[draft] keeping {len(best.split())} words ({words} asked)")
                    return best
                run.log(f"[draft] {count} words or wrong language "
                        f"(«{' '.join(body.split()[:15])}»); retrying")
                ask = prompt + (
                    f"\n\nThe previous answer had {count} words. Write a complete version "
                    f"of {int(words * 0.9)} to {int(words * 1.1)} words in "
                    f"{'English' if LANG == 'en' else 'Spanish'}. Develop the implications "
                    "of the sources without repeating or inventing facts; check the length.")
    finally:
        # _recover() inside chat may have repointed the chain; keep that choice.
        if llm.CHAIN == rotated:
            llm.CHAIN = chain
    raise RuntimeError(f"Section {label}: no provider returned a complete section "
                       f"in {LANG}. Nothing was cached.")


def _apparatus_rules(spec: dict, words: int, first: int) -> str:
    """The footnote and typography block, sized for this chunk.

    `corto` declares `apparatus: False` and means it — a 1.400-word intervention
    piece with six footnotes is not the author's register, it is a parody of it.
    """
    if not spec["apparatus"]:
        return ""
    return APPARATUS_RULES.format(notas_max=max(2, words // 300),
                                  nota_1=first)


_NOTES_MARK = re.compile(r"^\s*(?:NOTAS?|NOTES?)\s*:\s*$", re.M | re.I)


def _split_notes(body: str) -> tuple[str, list[str]]:
    """Peel the footnote block off a drafted chunk.

    The model writes its notes after a bare `NOTES:` line; they are collected
    across the whole draft and printed once, before the reference list, the way
    the author's own articles print them. A chunk with no notes is returned
    untouched, which is also every chunk cached by a run older than this.
    """
    parts = _NOTES_MARK.split(body, maxsplit=1)
    if len(parts) < 2:
        return body.strip(), []
    notes = [n.strip() for n in parts[1].strip().split("\n") if n.strip()]
    return parts[0].strip(), notes


def _strip_preamble(text: str) -> str:
    """Drop a leading '## Title' or meta line the model sometimes prepends.

    Also drops any '[p. N]' page marker copied over from a dossier PDF: the
    markers are there to be read, never to be published. The APA form
    '(Kurz, 1998, p. 12)' is parenthetical and untouched.

    The heading pattern is not the only shape a preamble takes. Measured on the
    20260824 run, sec08 shipped with «Two-front section, dossier tight (Scholz
    x2, Barcelona, Varela, Cruz). Writing now.» followed by a '---' rule: the
    model's own note to itself, published as the section's first sentence. Any
    short block closed by a horizontal rule is that note — a real section does
    not open with two lines and a divider.
    """
    text = re.sub(r"\[p+\.\s*\d+\]", "", text)
    head, rule, rest = text.strip().partition("\n---")
    if rule and len(head.split()) <= 60 and rest.strip():
        text = rest.lstrip("-\n").strip()
    lines = text.strip().split("\n")
    while lines and (re.match(r"^#{1,4}\s", lines[0]) or
                     re.match(r"^\**(Sección|Section)\b", lines[0])):
        lines.pop(0)
    return "\n".join(lines).strip()


_TAIL_MARK = "\n\n(Last lines written, to splice onto: …"


def _update_synopsis(previous: str, sec: dict, body: str) -> str:
    """Rolling memory. Keeps long-form runs coherent without resending everything.

    Only the *latest* section's tail is carried forward; older tails are dropped
    so the synopsis grows by one summary per section instead of ballooning.
    """
    tail = " ".join(body.split()[-120:])
    summary = llm.chat(llm.FLASH,
        f"Summarize in 90 words what this section argues and what conclusion it "
        f"reaches. Dry prose, no ornament.\n\n{body[:12000]}", temperature=0.3)
    summaries = previous.split(_TAIL_MARK)[0].strip()
    entry = f"[{sec['n']}] {sec['titulo']}: {summary}"
    return f"{summaries}\n{entry}".strip() + _TAIL_MARK + f"{tail})"


# --------------------------------------------------------------------------- #
# 5. Review & approval
# --------------------------------------------------------------------------- #

REVIEW_PROMPT = """You are a severe peer reviewer, a specialist in critical social \
theory and the critique of political economy. Review the following {kind}.

HYPOTHESIS IT HAD TO DEFEND: {hipotesis}

Citations the text uses that DO exist in the dossier: {ok}
Citations that do NOT exist in the dossier (possible inventions): {bad}

AVAILABLE SOURCES (check attributions against these passages; where there is only an
abstract or an excerpt, do not assume you verified the rest of the work):
{fuentes}

Assess without mercy:
- Is the hypothesis defended with arguments, or merely declaimed?
- Are there conceptual errors about Marx, value critique, political economy?
- Do the citations support what they are made to say, or are they decorative?
  Check substantive author/work attributions, quotations and historical claims even when
  uncited. Matching a dossier key is not evidence of support. Require a supported citation
  at the passage or narrow/remove the claim; never invent missing evidence or source access.
  Flag dossier/excerpt-access notes, copyright-policy refusals, editorial instructions and
  placeholders as high severity. Legitimate copyright scholarship and evidenced discussion
  of editions or methodological limitations are allowed, not production leakage.
  Review the abstract, notes and reference list too when present. Cite only the edition
  supported by the evidence; a copyright date is not automatically a publication date.
- Does the citation style follow APA 7 in the article's language —(Surname, year),
  two authors, (Surname et al., year), (Surname, year, p. 302) for verbatim quotes— or
  are formats mixed?
- Is there any mention of a study, report, index, survey or figure WITHOUT its citation
  beside it? Do institutions or consultancies appear that are not in the dossier? Are
  there unreferenced formulas left («a recent study», «according to reports», «it is
  estimated that»)? List them one by one: each is a high-severity problem.
- Is the strong counterargument missing? Is the position under dispute attributed to
  someone specific, with the citation, or does it fight abstractions?
- DOCUMENTARY DENSITY: count the facts about the world per thousand words —exact dates,
  figures with their unit, proper names with their post, institutions, places, titles
  and pages. A whole paragraph with none, made only of theorists' names and positions,
  is a medium-severity problem: ask for it to be anchored in the fact the source
  provides, and if the source provides none, for the paragraph to be shortened. A
  citation does not count as a fact.
- Are there sections that say the same thing in other words?
- VERBATIM REPETITIONS already found by counting: {repes}
  Discard those that are the topic's terminology or proper names (the object of study
  is always called the same, and that is fine). The rest —formulas, images, comparisons,
  anecdotes, punchlines, rhetorical questions— is a high-severity problem: rewrite or
  delete the second occurrence, never keep it. Also look for those the count missed.
- REGISTER REQUIRED BY THE FORMAT: {registro}
  Does the text respect it, or slide into another tone?
- TONE: is the text picking a fight? Mark as high severity every dismissal of an author
  («I disagree with X», «X stays on the surface», «I distrust X», «stops halfway»),
  every irony at third parties, every punchline and every emphatic personal confession.
  A sober critical remark now and then is fine; one objection per paragraph is not.
  Disagreement is argued with facts, never with adjectives or judgements of the person.
- TODAY IS {hoy}: does the text discuss current affairs with the most recent figures
  and their dates, or treat the present as if it were a year ago? Does it assert facts
  later than the sources?

Return JSON (keep these field names and enum values exactly):
{{"veredicto": "aprobado"|"revisar"|"rechazado",
  "puntaje": 0-100,
  "fortalezas": ["..."],
  "problemas": [{{"seccion": "title or no.", "gravedad": "alta"|"media"|"baja",
                  "problema": "...", "correccion": "exactly what to do"}}],
  "citas_a_eliminar": ["those that do not hold"],
  "falta_desarrollar": ["..."]}}

=== TEXT ===
{text}"""


def review(run: Run, text: str, topic: dict, sources: list[research.Source]) -> dict:
    used, unknown = research.verify_citations(text, sources)
    run.log(f"[review] citations checked: {len(used)} valid, {len(unknown)} unbacked")
    if unknown:
        run.log("  · unbacked: " + ", ".join(unknown[:10]))
    repes = humanize.repeated_phrases(text)
    if repes:
        run.log(f"[review] {len(repes)} repeated phrase(s): «{repes[0]}»…")
    run.log("[review] PRO report on the draft…")
    rep = llm.chat_json(llm.PRO, REVIEW_PROMPT.format(
        kind=FORMATS[run.fmt]["kind"], registro=FORMATS[run.fmt]["register"],
        repes="; ".join(f"«{r}»" for r in repes) or "(none)",
        hoy=hoy(), hipotesis=topic.get("hipotesis", ""),
        ok=", ".join(used[:60]) or "(none)",
        fuentes=_sources_block(sources, used, 30000),
        bad=", ".join(unknown[:30]) or "(none)", text=text[:120000]) + _lang(),
temperature=0.35)
    rep["text_hash"] = humanize.text_hash(text)
    rep["citas_validas"] = used
    if len(text) > 120000 or editorial_issues(text):
        rep.setdefault("problemas", []).append({"gravedad": "alta", "seccion": "final",
            "problema": "Production residue or text exceeds complete-review context.",
            "correccion": "Remove production residue; do not approve an unread tail."})
    rep["citas_sin_respaldo"] = unknown
    run.log(f"[review] verdict: {rep.get('veredicto')} ({rep.get('puntaje')}/100), "
            f"{len(rep.get('problemas', []))} problems")
    return rep


REVISE_PROMPT = """Correct the text according to the review report, with surgical patches: \
do not rewrite it whole, return only the fragments that change. Change ONLY what was \
flagged; do not touch what works, or the style.

REPORT:
{report}

NON-NEGOTIABLE RULE ON CITATIONS: these keys exist in the bibliography —{ok}—, but key \
membership does not establish claim support. Preserve them EXCEPT keys explicitly named \
in the report's citas_a_eliminar. For those, narrow/remove the unsupported attribution or \
replace it with a claim and citation supported by the supplied passages. All other citation \
removals and all invented keys are rejected.

Citations without bibliographic backing — those, yes: remove them or replace them with \
one of the verified keys above.

Citation rules (APA 7) the corrected text must follow:
{apa}

AVAILABLE SOURCES (the only bibliography that exists):
{fuentes}

=== TEXT ===
{text}
=== END ===

Return JSON: {{"parches": [{{"buscar": "…", "reemplazar": "…"}}]}}
- `buscar` is a LITERAL copy of the text above —same words, same accents, same \
punctuation— long enough to occur ONLY once: the whole sentence, or the paragraph if the \
sentence repeats. If it does not occur exactly, or occurs twice, the patch is lost.
- `reemplazar` is that same fragment, corrected; "" deletes it.
- One patch per problem, and nothing beyond the report's problems."""


def _apply_patch(text: str, find: str, repl: str) -> str | None:
    """Swap the single occurrence of `find` for `repl`. None if it is not unique.

    The model retypes the fragment, so exact match fails on a line break it turned
    into a space or a double space it collapsed; the fallback matches on the words
    and lets any whitespace sit between them.
    """
    if text.count(find) == 1:
        return text.replace(find, repl)
    if find in text:
        return None                                   # ambiguous: several copies
    loose = re.compile(r"\s+".join(re.escape(w) for w in find.split()))
    return (loose.sub(lambda _m: repl, text, count=1)
            if len(loose.findall(text)) == 1 else None)


def revise(run: Run, text: str, report: dict, sources: list[research.Source]) -> str:
    problems = [p for p in report.get("problemas", []) if p.get("gravedad") != "baja"]
    if not problems and not report.get("citas_sin_respaldo") and not report.get("citas_a_eliminar"):
        return text
    # The reviewer writes citations however it likes («Smith, 2020», «(Smith, 2020)»,
    # «Smith (2020)»); resolve them the way the text's own citations are resolved.
    authorized = set(research.verify_citations(
        " ".join(f"({c})" for c in report.get("citas_a_eliminar", [])), sources)[0])
    run.log(f"[review] applying {len(problems)} corrections…")
    valid = report.get("citas_validas", [])
    fuentes = _sources_block(sources, valid, 30000)
    # PRO, not FLASH: applying the report is the same judgement that wrote it. FLASH
    # answered a report about invented citations by inventing more of them, and left
    # production notes ("Now the final section text:") in the prose.
    #
    # Patches, not the whole text: a 10k-word article rewritten in one call costs a
    # 4.000-word completion and was discarded whole when a single citation slipped
    # ("bajó de 14 a 13 citas válidas") — three minutes of PRO for nothing. Each patch
    # is now checked and kept or dropped on its own.
    answer = llm.chat_json(llm.PRO, REVISE_PROMPT.format(
        report=json.dumps({"problemas": problems,
                           "citas_a_eliminar": report.get("citas_a_eliminar", []),
                           "citas_sin_respaldo": report.get("citas_sin_respaldo", []),
                           "falta_desarrollar": report.get("falta_desarrollar", [])},
                          ensure_ascii=False, indent=2),
        ok=", ".join(valid[:60]) or "(none yet)", apa=APA_RULES,
        fuentes=fuentes, text=text) + _lang(), temperature=0.5)
    patches = answer.get("parches") or []
    revised, applied, lost, missed = text, 0, 0, 0
    for p in patches:
        find, repl = (p.get("buscar") or ""), (p.get("reemplazar") or "")
        if not find.strip():
            continue
        # Only explicit reviewer authorization can relax citation preservation.
        had, _ = research.verify_citations(find, sources)
        keeps, unknown = research.verify_citations(repl, sources)
        if set(had) - set(keeps) - authorized or unknown:
            lost += 1
            continue
        out = _apply_patch(revised, find, repl)
        if out is None:
            missed += 1
            continue
        revised, applied = out, applied + 1
    detail = "".join([f", {lost} dropped over citations" if lost else "",
                      f", {missed} without an exact match" if missed else ""])
    run.log(f"[review] {applied}/{len(patches)} patches applied{detail}")
    return revised


def editorial_issues(text: str) -> list[str]:
    """High-confidence production residue only; this is not a factuality detector."""
    patterns = (
        r"\[(?:dato a verificar|to verify|data to verify|citation needed|insert (?:citation|source)|cita requerida)\]",
        r"(?:now the final section text|writing now|ahora el texto final de la sección)\s*[:.]",
        r"(?:dossier.{0,80}(?:excerpt|available here|contents|source access|bylines?)|(?:excerpt|extracto).{0,40}(?:del |the )?dossier)",
        r"\[(?:verificar|cita|fuente|todo|citation|source|tbd)\]",
        r"(?:supplied|provided) (?:excerpt|extract)|(?:extracto|fragmento) (?:suministrado|proporcionado)",
        r"\bonly [^.]{0,60}\b(?:is|are) to hand\b",
        r"(?:I (?:cannot|can't)|no puedo).{0,100}(?:copyright|derechos de autor)",
    )
    return [m.group(0) for pattern in patterns for m in re.finditer(pattern, text, re.I)]


def review_passed(report: dict) -> bool:
    """No rejection, no high-severity finding, no citation the reviewer or the
    verifier refused. Medium findings are the revise loop's job, not a veto: a
    PRO report nearly always carries some, and vetoing on them would block every run."""
    return (report.get("veredicto") in ("aprobado", "revisar")
            and not report.get("citas_sin_respaldo") and not report.get("citas_a_eliminar")
            and not any(p.get("gravedad") == "alta" for p in report.get("problemas", [])))


def _final_review(run: Run, text: str, topic: dict, sources: list[research.Source]) -> dict:
    """A review of exactly the text going to approval. The review rounds read the
    draft before humanizing and correction, so their report says nothing about
    what is actually published. Cached by hash, so a resume does not pay again."""
    cached = run.load("06_review_final.json")
    if cached and cached.get("text_hash") == humanize.text_hash(text):
        run.log("[review] reusing 06_review_final.json")
        return cached
    rep = {**review(run, text, topic, sources), "text_hash": humanize.text_hash(text)}
    run.save("06_review_final.json", rep)
    return rep


def require_publishable(text: str, approval: dict | None) -> None:
    """Shared, offline live-publication boundary. Draft uploads need no approval."""
    if (not isinstance(approval, dict) or approval.get("publicable") is not True
            or approval.get("semantic_review_passed") is not True
            or approval.get("text_hash") != humanize.text_hash(text)
            or editorial_issues(text)):
        raise ValueError("Live publication requires a clean substantive approval for the exact final Markdown.")


def final_approval(run: Run, text: str, topic: dict, det: dict,
                   sources: list[research.Source] | None = None, report: dict | None = None,
                   abstract: str = "") -> dict:
    sources, report = sources or [], report or {}
    used, unknown = research.verify_citations(text, sources)
    # ponytail: bounded model context; oversized finals fail closed rather than certify an unread tail.
    complete = len(text) <= 120000
    semantic_ok = (bool(sources) and review_passed(report) and complete
                   and report.get("text_hash") == humanize.text_hash(text)
                   and not unknown and not editorial_issues(text))
    run.log("[approval] PRO's final verdict…")
    verdict = llm.chat_json(llm.PRO, f"""Final approval. Decide whether this text is published.

Criteria: soundness of the argument, bibliographic honesty, internal coherence, \
prose that sounds like a human author with a voice of their own and not like a language model.

AI detector score (0 = human, 100 = machine): {det.get('final_score')}

SOURCE EVIDENCE (key membership alone does not prove support):
{_sources_block(sources, used, 30000)}

LATEST SUBSTANTIVE REVIEW (unresolved material findings forbid approval):
{json.dumps(report, ensure_ascii=False)}

Check the whole text, its footnotes and the abstract below. Reject
unsupported attributions, quotes, figures and historical claims, mismatched editions,
production/source-access commentary and placeholders. Do not reject legitimate scholarship
about copyright or methodological limitations. Never invent support absent from evidence.

Return JSON: {{"publicable": true|false, "puntaje": 0-100, \
"dictamen": "3-6 lines", "ajustes_menores": ["..."]}}

TITLE: {topic['titulo']}

ABSTRACT: {abstract or '(none)'}

{text[:120000]}""" + _lang(), temperature=0.3)
    if not isinstance(verdict, dict):
        verdict = {"dictamen": f"unreadable answer: {str(verdict)[:200]}"}
    # «"publicable": "false"» is a truthy string: read as approval it skipped the
    # edit gate and let `--publish auto` go live. Only an explicit yes counts.
    said = verdict.get("publicable")
    verdict["publicable"] = said is True or str(said).strip().lower() in ("true", "sí", "si", "yes")
    verdict.update(publicable=verdict["publicable"] and semantic_ok,
                   semantic_review_passed=semantic_ok, text_hash=humanize.text_hash(text))
    if not semantic_ok:
        verdict["dictamen"] = "Substantive review missing, stale, incomplete or unresolved; publication blocked."
    run.log(f"[approval] publishable={verdict.get('publicable')} "
            f"({verdict.get('puntaje')}/100)")
    return verdict


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

def assemble(run: Run, topic: dict, outline: dict, text: str,
             sources: list[research.Source], report: dict, det: dict,
             verdict: dict) -> str:
    spec = FORMATS[run.fmt]
    cited_text = text + ("\n" + outline.get("resumen", "") if spec["apparatus"] else "")
    used, _ = research.verify_citations(cited_text, sources)
    today = dt.date.today()
    en = LANG == "en"
    month = MONTHS[today.month - 1]
    head = [f"# {outline.get('titulo_final', topic['titulo'])}", "",
            f"{'By' if en else 'Por'} {BYLINE}.", "",
            f"{month if en else MESES[today.month]}{' ' if en else ' de '}{today.year}.", ""]
    if spec["apparatus"]:
        head += [outline.get("resumen", ""), "",
                 ("Keywords: " if en else "Palabras clave: ") + ", ".join(outline.get("palabras_clave", [])), "", ""]
    body = [*head, text.strip()]
    # Abstract and keywords belong to the academic formats; the reference list is
    # not apparatus, it is what the in-text citations point at. A «corto» that
    # cites (Kurz, 1998) and prints no bibliography leaves the reader with a
    # surname and a year and no way to reach the book.
    if refs := research.bibliography(sources, used, lang=LANG).strip():
        body += ["", "", "## References" if en else "## Referencias", "", refs]
    doc = "\n".join(body)
    # The approval was given to the body; what gets published is the assembled page
    # (title, abstract, references). Bind it to that exact file, and let residue in
    # the abstract veto it like residue in the body would.
    if editorial_issues(doc):
        verdict.update(publicable=False, semantic_review_passed=False)
    verdict["text_hash"] = humanize.text_hash(doc)
    run.save("05_final.md", doc)
    run.save("06_review.json", report)
    run.save("07_detector.json", det)
    run.save("08_approval.json", verdict)
    return doc


def run_pipeline(run: Run, *, rounds: int = 2, detector_rounds: int = 3,
                 threshold: float = 25.0) -> pathlib.Path:
    topic = pick_topic(run)
    sources, _gaps = do_research(run, topic)
    if not sources:
        # RuntimeError, not SystemExit: --continuous catches Exception, and one topic
        # with an empty dossier (a network blip) used to end the whole loop.
        raise RuntimeError("No source could be found. Check the connection.")
    outline = make_outline(run, topic, sources)
    text = draft(run, topic, outline, sources)
    run.save("04_draft.md", text)
    run.log(f"[draft] {len(text.split())} words")

    report: dict = {}
    for i in range(1, rounds + 1):
        # Both halves of a round are cached: the report costs a PRO call over the whole
        # draft, and a resume that re-derives it pays it again for the same verdict.
        # The report is saved before the corrections are applied, so a crash mid-revise
        # resumes on the corrections instead of re-reviewing.
        # Reports saved before the hash existed are trusted: re-reviewing them costs PRO.
        if ((cached := run.load(f"06_review_r{i}.json"))
                and cached.get("text_hash", humanize.text_hash(text)) == humanize.text_hash(text)):
            run.log(f"[review] reusing 06_review_r{i}.json (round {i})")
            report = cached
        else:
            report = review(run, text, topic, sources)
            run.save(f"06_review_r{i}.json", report)
        if (report.get("veredicto") == "aprobado" and not report.get("citas_sin_respaldo")
                and not any(p.get("gravedad") in ("alta", "media")
                            for p in report.get("problemas", []))):
            break
        if i == rounds:
            break
        if (done := run.load(f"04_draft_rev{i}.md")):
            run.log(f"[review] reusing 04_draft_rev{i}.md")
            text = done
            continue
        if run.mode != "auto":
            for p in report.get("problemas", [])[:8]:
                run.log(f"  · [{p.get('gravedad')}] {p.get('seccion')}: {p.get('problema')}")
            if run.ask("Apply the corrections? [Y/n]", "y").lower().startswith("n"):
                break
        text = revise(run, text, report, sources)
        run.save(f"04_draft_rev{i}.md", text)

    cached_det = run.load("07_detector_humanizado.json") or run.load("07_detector.json")
    if ((cached := run.load("04_draft_humanizado.md")) and cached_det
            and cached_det.get("lang") == LANG
            and cached_det.get("text_hash") == humanize.text_hash(cached)):
        run.log("[detector] reusing 04_draft_humanizado.md")
        text, det = cached, cached_det
        score = det.get("final_score")
        det["passed"] = score < threshold if score is not None else None
        used, unknown = research.verify_citations(text, sources)
    else:
        run.log("[detector] running the AI detectors…")
        before, before_unknown = research.verify_citations(text, sources)
        humanized, det = humanize.humanize(text, rounds=detector_rounds, threshold=threshold,
                                           register=FORMATS[run.fmt]["register"], log=run.log,
                                           lang=LANG)
        # The humanizer rewrites prose freely, so citations must be re-verified. Losing
        # references to a stylistic rewrite is not a trade worth making.
        used, unknown = research.verify_citations(humanized, sources)
        # Only what the rewrite itself broke counts against it: an unbacked
        # citation the draft already carried is the correction step's job below,
        # and used to throw away every rewrite of such a draft.
        if set(before) - set(used) or set(unknown) - set(before_unknown):
            run.log("[detector] the rewrite lost or invented citations; "
                    "keeping the revised draft")
            used, unknown = research.verify_citations(text, sources)
            # The score saved below must describe the draft saved beside it, or
            # the hash check above fails and every resume humanizes again. Round 1
            # already scored exactly this text; reuse it instead of paying again.
            first = next((r for r in det.get("rounds", [])
                          if r.get("text_hash") == humanize.text_hash(text)), None)
            if first:
                det = {**det, "final_score": first["worst"], "text_hash": first["text_hash"],
                       "selected_round": first["round"],
                       "passed": first["worst"] < threshold if first["worst"] is not None else None}
            else:
                _, det = humanize.humanize(text, rounds=1 if detector_rounds > 0 else 0,
                                           threshold=threshold, log=run.log, lang=LANG)
        else:
            text = humanized
        run.save("04_draft_humanizado.md", text)
        # Saved here, not only in assemble(): without it the humanized draft is on disk
        # with no detector score beside it, and the whole detector stage runs again.
        run.save("07_detector_humanizado.json", det)
    report["citas_validas"], report["citas_sin_respaldo"] = used, unknown
    # The post-humanizing correction is a whole-document PRO call and it was the one
    # stage with no cache file: a --resume paid for it again on every cycle. Hand edits
    # made at the approval gate land in the same file, so they survive a Ctrl-C too.
    if (fixed := run.load("04_draft_corregido.md")):
        run.log("[review] reusing 04_draft_corregido.md")
        text = fixed
    elif unknown:
        run.log(f"[detector] the rewrite left {len(unknown)} unbacked citations; correcting")
        text = revise(run, text, report, sources)
        run.save("04_draft_corregido.md", text)
    if fixed or unknown:
        used, unknown = research.verify_citations(text, sources)
        report["citas_validas"], report["citas_sin_respaldo"] = used, unknown

    # Review what will actually be published; one round of patches if it fails.
    report = _final_review(run, text, topic, sources)
    if not review_passed(report) and not fixed:
        patched = revise(run, text, report, sources)
        # Saved even when unchanged: it marks the attempt, so a resume does not pay again.
        run.save("04_draft_corregido.md", patched)
        if patched != text:
            text = patched
            report = _final_review(run, text, topic, sources)
    used, unknown = research.verify_citations(text, sources)
    report["citas_validas"], report["citas_sin_respaldo"] = used, unknown

    if det.get("text_hash") != humanize.text_hash(text):
        _, det = humanize.humanize(text, rounds=1 if detector_rounds > 0 else 0,
                                   threshold=threshold, log=run.log, lang=LANG)
    run.save("07_detector.json", det)
    verdict = final_approval(run, text, topic, det, sources, report,
                             outline.get("resumen", "") if FORMATS[run.fmt]["apparatus"] else "")
    if unknown:
        verdict.update(publicable=False, dictamen="Unbacked citations remain.")
    # PRO's rejections are usually minutes of hand editing ("[dato a verificar]" left in,
    # a year that reads 1993a in one section and 1993 in the next). Let the user fix them
    # in place and ask PRO again instead of throwing the run away or publishing anyway.
    while not verdict.get("publicable") and run.mode != "auto":
        run.log("[approval] " + str(verdict.get("dictamen", "")))
        for adj in (verdict.get("ajustes_menores") or [])[:8]:
            run.log(f"  · {adj}")
        # The edited file is the corrected body, not the humanized draft: it is what the
        # resume branch above reads back, so an edit outlives a Ctrl-C at this prompt.
        run.save("04_draft_corregido.md", text)
        choice = run.ask(f"Not approved. [e]dit {run.dir / '04_draft_corregido.md'} "
                         "and review again, [p]ublish anyway (as a draft if live needs approval), [n]ot publish [e/p/N]",
                         "n").strip().lower()[:1]
        if choice != "e":
            if choice != "p":
                run.log("[approval] saved as a draft, not approved.")
            break
        run.ask("Edit the file and press Enter when ready…", "")
        text = run.load("04_draft_corregido.md") or text
        used, unknown = research.verify_citations(text, sources)
        report["citas_validas"], report["citas_sin_respaldo"] = used, unknown
        if unknown:
            run.log(f"[approval] your edit left {len(unknown)} unbacked citations: "
                    + ", ".join(unknown[:6]))
        if det.get("text_hash") != humanize.text_hash(text):
            _, det = humanize.humanize(text, rounds=1 if detector_rounds > 0 else 0,
                                       threshold=threshold, log=run.log, lang=LANG)
        report = _final_review(run, text, topic, sources)
        verdict = final_approval(run, text, topic, det, sources, report,
                             outline.get("resumen", "") if FORMATS[run.fmt]["apparatus"] else "")
        if unknown:
            verdict.update(publicable=False, dictamen="Unbacked citations remain.")
    doc = assemble(run, topic, outline, text, sources, report, det, verdict)
    out = run.dir / "05_final.md"
    # What the publisher needs to decide draft-vs-live, without re-reading the run.
    run.state.update(topic=topic, verdict=verdict, detector=det, final=out)
    run.log(textwrap.dedent(f"""
        ─────────────────────────────────────────
        Done: {out}
        Words: {len(doc.split())}
        Sources cited: {len(report.get('citas_validas', []))} of {len(sources)} gathered
        AI detector: {det.get('final_score')} (threshold {threshold})
        Approval: {verdict.get('puntaje')}/100
        ─────────────────────────────────────────"""))
    return out
