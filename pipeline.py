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
                  "kind": "artículo breve de intervención",
                  "register": "Artículo breve: una sola tesis, prosa directa, sin digresiones "
                              "ni aparato. Registro de opinión informada, no de paper. La "
                              "posición es clara desde el principio y se sostiene con un "
                              "ejemplo concreto, no con adjetivos."},
    "medio":     {"words": (3000, 4000),   "sections": (4, 6),  "apparatus": True,  "chapters": False,
                  "kind": "artículo de divulgación teórica",
                  "register": "Divulgación teórica: explicá cada término técnico la primera vez "
                              "que aparece, un ejemplo concreto por sección, prosa de revista "
                              "cultural. El lector no es colega: no des por sabida ninguna "
                              "escuela ni ninguna sigla."},
    "largo":     {"words": (7000, 9000),   "sections": (5, 9),  "apparatus": True,  "chapters": False,
                  "kind": "ensayo extenso con aparato crítico",
                  "register": "Ensayo con aparato crítico: registro ensayístico, la digresión "
                              "está permitida si vuelve al argumento, la voz del autor es "
                              "visible pero mesurada. El aparato va en el cuerpo del texto, "
                              "no en notas."},
    "paper":     {"words": (10000, 13000), "sections": (6, 10),  "apparatus": True,  "chapters": False,
                  "kind": "paper académico con resumen, palabras clave, hipótesis y conclusión",
                  "register": "Paper académico: registro impersonal y sobrio. Sin ironía, sin "
                              "frases de efecto, sin primera persona enfática (la primera "
                              "persona sólo para decisiones metodológicas: «relevé», "
                              "«clasifiqué»). Método, evidencia y límites explícitos. Toda "
                              "afirmación fuerte va con su cita."},
    "tesis":     {"words": (40000, 80000), "sections": (8, 24), "apparatus": True,  "chapters": True,
                  "kind": "tesis: capítulos con estado de la cuestión, marco teórico, "
                          "desarrollo, discusión y conclusiones",
                  "register": "Tesis: registro académico expositivo. Cada capítulo declara qué "
                              "función cumple en el argumento general, define sus términos y "
                              "cierra con lo que deja establecido. Sin retórica de intervención "
                              "y sin ironía."},
    "libro":     {"words": (70000, 120000), "sections": (10, 40), "apparatus": True, "chapters": True,
                  "kind": "libro de ensayo teórico dividido en capítulos",
                  "register": "Libro de ensayo: registro amplio, puede narrar y ejemplificar "
                              "largo. La voz del autor es visible, nunca polémica de coyuntura. "
                              "Cada capítulo se sostiene solo y a la vez avanza el libro."},
    "discusion": {"words": (3500, 4500),   "sections": (4, 6),  "apparatus": True,  "chapters": False,
                  "kind": "discusión: confrontación argumentada entre posiciones teóricas rivales",
                  "register": "Discusión teórica: exponé cada posición rival en sus mejores "
                              "términos antes de objetarla, y atribuile a cada una lo que "
                              "efectivamente sostiene, con su cita. El desacuerdo es el objeto "
                              "del texto, y por eso tiene que ser argumentado, nunca enfático "
                              "ni personal."},
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

# Every drafting and revision prompt inherits this; APA 7 is a hard requirement.
# Article language. "es" is the default and the only calibrated one: the corpus,
# the style guide and every detector floor are measured on the author's Spanish
# prose. "en" (--english) keeps the Spanish instructions but orders English
# output; humanize swaps to its English tell lists and rewrite prompt.
LANG = "es"


def _lang() -> str:
    if LANG == "es":
        return "\n\n" + style.writing_rules("es")
    return ("\n\nOUTPUT LANGUAGE: Write all article prose, titles, headings, abstract and "
            "keywords in English, regardless of the language of these instructions. "
            "Keep JSON field names as specified. For APA citations use & between two "
            "authors in parentheses, and 'and' in narrative citations; use n.d. for "
            "an undated work. Preserve original-language quotations and source titles.\n"
            + style.writing_rules("en"))


APA_RULES = """- Citás en NORMAS APA (7ª edición), en español:
  · en el texto, (Apellido, año) o (Apellido, año, p. 302) cuando citás textual;
  · dos autores, (Apellido y Apellido, año); tres o más, (Apellido et al., año);
  · si citás dos obras del mismo autor y año, usá el sufijo de la clave (2016a, 2016b);
  · cita narrativa cuando el autor es sujeto de la oración: «Postone (2006) sostiene que…»;
  · sin página no inventes una: poné (Apellido, año) a secas o (Apellido, año, s.p.);
  · los textos del dossier que vienen de un PDF traen marcas «[p. N]» intercaladas:
    son el número de página. Si citás textual o parafraseás un pasaje puntual, poné
    la página de la marca [p. N] inmediatamente anterior al pasaje: (Apellido, año,
    p. N). Las marcas NO se copian al artículo, sólo se leen;
  · si la clave del dossier no trae año, citala tal cual viene (Apellido, s/f): no le
    inventes un año ni le pongas «s/f» al lado de uno.
- Usá EXACTAMENTE las claves del dossier. Inventar una cita arruina el trabajo entero.
- TODA mención de un estudio, informe, índice, ranking, encuesta, relevamiento o dato
  estadístico lleva su cita en el mismo enunciado, con autor y año. Sin excepción.
- Si algo no está en el dossier, no lo menciones: ni la institución, ni la consultora,
  ni el nombre del informe, ni sus resultados. Prohibidas las fórmulas sin referencia:
  «un estudio reciente», «según informes de consultoras», «la literatura muestra»,
  «se estima que», «distintos trabajos coinciden». O va con cita, o no va.
- Lo mismo vale para las cifras: un porcentaje, un monto o una tasa sin (Apellido, año)
  al lado no se escribe. Si el dato hace falta y no está, marcá [dato a verificar]."""


def wmid(spec: dict) -> int:
    """Middle of a format's word range — used for internal budgeting (context
    size, source budget), never as a target handed to the model."""
    lo, hi = spec["words"]
    return (lo + hi) // 2


def wrange(spec_or_pair) -> str:
    """«3.000 y 4.000» — how a length is stated in every prompt."""
    lo, hi = spec_or_pair["words"] if isinstance(spec_or_pair, dict) else spec_or_pair
    return f"{lo:,} y {hi:,}".replace(",", ".")


def hoy() -> str:
    """«21 de agosto de 2026». The models have a training cutoff and, left alone,
    write about the present as if it were two years ago; every prompt that touches
    current events gets the real date."""
    d = dt.date.today()
    return f"{d.day} de {MESES[d.month].lower()} de {d.year}"


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

TOPIC_PROMPT = """HOY ES {hoy}. Todo lo que digas sobre la coyuntura tiene que \
referirse a este presente, no al de tu entrenamiento: si no lo ves en los titulares \
de abajo, no lo des por cierto.

Sos el editor de una revista de ensayo en español, curiosa y variada. \
El autor tiene muchos intereses; para ESTE número, trabajá sobre estas áreas \
(elegidas al azar entre las suyas, para que la revista no gire siempre sobre lo mismo):

{interests}

Sus últimos artículos fueron (el primero es el más reciente):
{recent}

Otros títulos que ya escribió: {written}.

REGLAS DE VARIEDAD (obligatorias):
- Ningún tema puede retomar el asunto, la tesis o el ángulo de un artículo ya escrito, \
aunque cambie el título.
- Los 5 temas tienen que ser de áreas distintas entre sí; cada área de arriba puede \
aparecer como mucho en un tema.
- La economía política, la crítica del valor y el marxismo pueden ser, como mucho, \
una lente en UN tema, no el eje de todos. Un artículo sobre dinosaurios, aviones, \
lenguas o videojuegos vale por sí mismo: no hace falta convertirlo en crítica del capital.
- Sé creativo: preferí preguntas raras, precisas y sorprendentes a los panoramas \
previsibles.

Titulares de las últimas semanas (para anclar en la coyuntura; la fecha va delante):
{news}

{brief_block}

Proponé 5 temas. Cada uno debe: (a) ser discutible, no un panorama descriptivo; \
(b) tener una hipótesis fuerte y arriesgada, susceptible de ser falsada; \
(c) apoyarse en literatura académica realmente existente; (d) tener una razón de por qué \
AHORA (un titular, un aniversario, un hallazgo, una publicación; no tiene que ser coyuntura \
económica).

Devolvé JSON:
{{"temas": [{{"titulo": "...", "pregunta": "¿...?", "hipotesis": "...", \
"por_que_ahora": "...", "tension_teorica": "qué discusión abre y contra quién", \
"autores_clave": ["..."], "obras_necesarias": ["Autor, Título (año)"], \
"riesgo": "por qué podría fallar"}}]}}"""


DEVELOP_PROMPT = """HOY ES {hoy}. La coyuntura es la de esta fecha, no la de tu \
entrenamiento.

El autor ya eligió su tema y no quiere alternativas. Tu tarea es \
convertirlo en un objeto de trabajo: afilarlo, darle hipótesis y encontrar la discusión.

TEMA PEDIDO POR EL AUTOR (respetalo, no lo cambies de asunto): «{brief}»

Campo del autor: {interests}

Titulares recientes, por si el tema tiene anclaje en la coyuntura:
{news}

Devolvé JSON:
{{"titulo": "título en el estilo del autor, que respete el tema pedido",
  "pregunta": "¿...?", "hipotesis": "hipótesis fuerte y falsable",
  "por_que_ahora": "...", "tension_teorica": "qué discusión abre y contra quién",
  "autores_clave": ["..."], "obras_necesarias": ["Autor, Título (año)"],
  "riesgo": "por qué podría fallar"}}"""


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
        run.log("[tema] reutilizando 01_topic.json")
        return cached
    if run.exact_topic and run.brief:
        return _develop_topic(run)
    # A few areas per run, not the whole list: handed everything, the model gravitates
    # to the same political-economy corner every time.
    areas = interest_areas()
    focus = random.sample(areas, min(TOPIC_AREAS, len(areas)))
    run.log(f"[tema] áreas de este número: {'; '.join(_area_query(a) for a in focus)}")
    run.log("[tema] sondeando novedades…")
    news = []
    for q in [_area_query(a) for a in focus] + ["Argentina"]:
        news += research.google_news(q, 5, "es", days=30)
        news += research.google_news(q, 4, "en", days=30)
    news.sort(key=lambda n: n.date, reverse=True)
    headlines = "\n".join(f"- {n.date or 's/f'} · {n.title} ({n.venue})"
                          for n in news[:45]) or "(sin datos)"
    # Own past runs first: in continuous mode the acute risk is repeating itself.
    # The most recent ones are listed apart so a truncated `written` cannot drop them.
    past = already_written()
    recent = "\n".join(f"- {t}" for t in past[:12]) or "(ninguno)"
    written = ", ".join(past[12:] + [p.stem.replace("-", " ") for p in style.CORPUS.glob("*.md")])
    brief_block = (f"El autor pidió específicamente trabajar sobre: «{run.brief}». "
                   "Todos los temas propuestos deben desarrollar ESE pedido desde ángulos distintos."
                   ) if run.brief else ""
    data = llm.chat_json(llm.PRO, TOPIC_PROMPT.format(
        hoy=hoy(), interests="\n".join(f"- {a}" for a in focus), recent=recent,
        written=written[:3000], news=headlines,
        brief_block=brief_block) + _lang(), temperature=1.0)
    temas = data.get("temas", [])
    # In auto mode nobody picks, so listing them is noise; and it takes one at random,
    # since "always the first" is always the model's most predictable idea.
    for i, t in enumerate(temas if run.mode != "auto" else [], 1):
        run.log(f"\n  [{i}] {t['titulo']}\n      hipótesis: {t['hipotesis']}\n"
                f"      ahora: {t.get('por_que_ahora','')}")
    default = str(random.randint(1, len(temas))) if run.mode == "auto" and temas else "1"
    choice = run.ask(f"Elegí tema [1-{len(temas)}] (Enter = 1, o escribí tu propio título):", default)
    if choice.isdigit() and 1 <= int(choice) <= len(temas):
        topic = temas[int(choice) - 1]
    else:
        # A typed title still gets a hypothesis and a bibliography plan built for it.
        run.brief = choice
        return _develop_topic(run)
    return _settle_topic(run, topic)


def _develop_topic(run: Run) -> dict:
    """Turn the user's own topic into a workable object without proposing others."""
    run.log(f"[tema] desarrollando el tema pedido: «{run.brief}»")
    run.log("[tema] buscando novedades recientes…")
    news = (research.google_news(run.brief, 8, "es", days=45)
            + research.google_news(run.brief, 5, "en", days=45))
    topic = llm.chat_json(llm.PRO, DEVELOP_PROMPT.format(
        hoy=hoy(), brief=run.brief, interests=interests_text(),
        news="\n".join(f"- {n.date or 's/f'} · {n.title} ({n.venue})"
                       for n in news[:25]) or "(sin datos)") + _lang(),
        temperature=0.85)
    run.log(f"  hipótesis: {topic.get('hipotesis','')}")
    if run.mode != "auto":
        # Loop: every piece of feedback regenerates the topic, and the new
        # hypothesis is shown again for validation until Enter accepts it.
        while tweak := run.ask("¿Ajusto la hipótesis o el título? (Enter = así está bien):", ""):
            topic = llm.chat_json(llm.PRO,
                f"Ajustá este tema según el pedido y devolvé el mismo JSON corregido.\n"
                f"PEDIDO: {tweak}\n\n{json.dumps(topic, ensure_ascii=False)}"
                f"{_lang()}", temperature=0.7)
            run.log(f"\n[tema] título: «{topic.get('titulo', '')}»\n"
                    f"  hipótesis: {topic.get('hipotesis', '')}")
    return _settle_topic(run, topic)


def _settle_topic(run: Run, topic: dict) -> dict:
    run.dir = OUTPUT / f"{dt.date.today():%Y%m%d}-{slugify(topic['titulo'])}"
    run.save("00_corrida.json", {"fmt": run.fmt, "lang": LANG})
    run.save("01_topic.json", topic)
    run.log(f"\n[tema] «{topic['titulo']}»\n[tema] carpeta: {run.dir}")
    return topic


# --------------------------------------------------------------------------- #
# 2. Research
# --------------------------------------------------------------------------- #

QUERIES_PROMPT = """Tema: {titulo}
Pregunta: {pregunta}
Hipótesis: {hipotesis}
Autores clave: {autores}

Generá consultas de búsqueda para reunir bibliografía real sobre esto.

IMPORTANTE: las bases académicas (OpenAlex, Crossref, DOAJ, Semantic Scholar) hacen \
AND entre todos los términos. Una consulta de nueve palabras no devuelve nada. \
Escribí consultas de 2 a 5 palabras, con los términos técnicos del campo, sin \
conectores ni frases completas. Mal: «utopías postrabajo realizables sin abolir el \
valor». Bien: «postwork utopia», «abolición del trabajo», «value-form critique».

Devolvé JSON:
{{"academicas_es": ["6 consultas en español, 2-5 palabras cada una"],
  "academicas_en": ["8 consultas en inglés, 2-5 palabras, terminología técnica"],
  "prensa": ["4 consultas de actualidad en español"],
  "prensa_en": ["3 consultas de actualidad en inglés"],
  "libros": ["6 obras concretas: «Autor, Título» que habría que conseguir sí o sí"],
  "contraargumentos": ["3 consultas cortas para literatura que REFUTE la hipótesis"]}}"""


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
        run.log("[investigación] reutilizando 02_dossier.json")
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
    run.log(f"[investigación] {len(queries)} consultas académicas + prensa + "
            f"{len(wanted)} obras buscadas por título")
    if "fuentes" in done:
        run.log(f"[investigación] reutilizando {len(sources)} fuentes ya reunidas")
    else:
        sources = research.gather(queries, news_queries=plan.get("prensa", []),
                                  news_queries_en=plan.get("prensa_en", []),
                                  book_titles=wanted, log=run.log)
        keep("fuentes")
    if "textos" in done:
        run.log("[investigación] fuentes: textos completos ya bajados")
    else:
        run.log(f"[investigación] fuentes: bajando el texto completo de las "
                f"{len(sources)} fuentes del dossier…")
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
        run.log(f"[bibliografía] faltan {len(gaps)} obra(s); sigo sin pedirlas.")
        return sources
    for attempt in range(3):
        run.log(f"\n[bibliografía] falta el texto completo de {len(gaps)} obra(s):")
        for g in gaps:
            run.log(f"\n  · {g['wanted']}")
            for f in g.get("fallbacks", [])[:4]:
                run.log(f"      {f['title']:<26} {f['url']}")
        if run.mode == "auto":
            run.log("\n[bibliografía] modo auto: sigo con lo que hay "
                    "(las obras faltantes no se van a citar).")
            return sources
        run.log(f"\n  Carpeta: {research.LIBRARY}")
        answer = run.ask("Dejá ahí los archivos (epub, pdf, docx, txt, html) y apretá Enter. "
                         "Escribí 'seguir' para continuar sin ellos, o pegá una URL:", "seguir")
        if answer.startswith("http"):
            body = research.fetch_text(answer, 200000)
            if len(body) > 1000:
                sources.append(research.Source(
                    title=answer.rsplit("/", 1)[-1][:120] or "fuente aportada",
                    authors=[], year="", kind="local", venue="aportado por el autor",
                    url=answer, abstract=body[:2000], fulltext=body, origin="usuario"))
                run.log(f"  · descargado: {len(body.split())} palabras")
            else:
                run.log("  · no pude extraer texto de esa URL")
        extra = [s for s in research.scan_library()
                 if s.url not in {x.url for x in sources}]
        if extra:
            sources += extra
            run.log(f"[bibliografía] biblioteca local: +{len(extra)} obras "
                    f"({sum(len(s.fulltext.split()) for s in extra)} palabras)")
        sources = research.assign_keys(research.dedupe(sources))
        research.save_dossier(sources, run.dir / "02_dossier.json")
        gaps = [g for g in gaps if not research._have_title(sources, g["wanted"])]
        run.save("02_faltantes.json", gaps)
        if not gaps:
            run.log("[bibliografía] completa.")
            return sources
        if answer.lower().startswith("seguir") and not extra and attempt:
            break
    run.log(f"[bibliografía] sigo sin {len(gaps)} obra(s); no se van a citar.")
    return sources


# --------------------------------------------------------------------------- #
# 3. Outline
# --------------------------------------------------------------------------- #

OUTLINE_PROMPT = """HOY ES {hoy}: el presente del texto es esta fecha.

Vas a planificar un {kind} de entre {words} palabras en español.
Es un RANGO, no una meta: cualquier extensión dentro del intervalo está bien.

REGISTRO DEL FORMATO (manda sobre cualquier otra consideración de tono):
{registro}

TÍTULO: {titulo}
PREGUNTA: {pregunta}
HIPÓTESIS A DEFENDER: {hipotesis}
TENSIÓN TEÓRICA: {tension}

FUENTES DISPONIBLES (usá SOLO estas claves para citar; no existe ninguna otra):
{catalogo}

Diseñá entre {nmin} y {nmax} secciones — cuántas, y de qué tamaño cada una, lo decidís vos \
según lo que el argumento necesite. Una sección puede ir de 500 a 10.000 palabras: algunas \
piden desarrollo largo y otras son un tramo breve que gira el argumento. No las hagas todas \
del mismo largo. Cada sección debe hacer avanzar el argumento, no repetirlo. \
Asigná a cada una las claves de las fuentes que le corresponden y un presupuesto de palabras; \
la suma de los presupuestos tiene que caer dentro del rango {words}.
{chapters_note}

Devolvé JSON:
{{"titulo_final": "título definitivo, en el estilo del autor",
  "resumen": "resumen de 120 palabras (abstract)",
  "palabras_clave": ["6 términos"],
  "secciones": [{{"n": 1, "titulo": "...", "tesis": "qué sostiene esta sección",
                  "contenido": "qué desarrolla, en 3-5 renglones",
                  "fuentes": ["Clave, año", "..."], "palabras": 000,
                  "subsecciones": ["..."]}}]}}"""


# Long formats get the outline in two passes. One call cannot plan 40 units in
# detail: the JSON runs past the output limit and comes back truncated, and what
# does arrive is forty interchangeable blocks. First the architecture (titles,
# function, word budget), then the detail in batches, each batch seeing the whole
# architecture so it knows what the other units already cover.
PLAN_PROMPT = """HOY ES {hoy}: el presente del texto es esta fecha.

Vas a diseñar la ARQUITECTURA de un {kind} de entre {words} palabras en español.
Es un RANGO, no una meta.

REGISTRO DEL FORMATO (manda sobre cualquier otra consideración de tono):
{registro}

TÍTULO: {titulo}
PREGUNTA: {pregunta}
HIPÓTESIS A DEFENDER: {hipotesis}
TENSIÓN TEÓRICA: {tension}

FUENTES DISPONIBLES (de ahí sale todo lo que se puede citar):
{catalogo}

La estructura la decidís vos: entre {nmin} y {nmax} unidades, y cada una de 500 a 10.000 \
palabras. Una unidad puede ser un capítulo largo, una parte, una sección corta, un excursus \
o un interludio: usá la forma que pida el argumento y NO repartas las palabras en partes \
iguales. Podés agrupar unidades consecutivas bajo una misma parte o capítulo mayor \
(decilo en «parte»). La suma de los presupuestos tiene que caer dentro del rango {words}.

Devolvé JSON:
{{"titulo_final": "título definitivo, en el estilo del autor",
  "resumen": "resumen de 120 palabras (abstract)",
  "palabras_clave": ["6 términos"],
  "estructura": [{{"n": 1, "parte": "capítulo o parte al que pertenece, o \\"\\"",
                   "titulo": "...", "funcion": "qué hace en el argumento general, 1-2 renglones",
                   "palabras": 000}}]}}"""

DETAIL_PROMPT = """Estás detallando el esquema de un {kind} titulado «{titulo}».

HIPÓTESIS GENERAL: {hipotesis}
REGISTRO: {registro}

ARQUITECTURA COMPLETA (para saber qué cubre cada unidad y no pisarlas):
{estructura}

FUENTES DISPONIBLES (usá SOLO estas claves para citar; no existe ninguna otra):
{catalogo}

Detallá SOLAMENTE las unidades número {ns}, respetando su título, su función y su \
presupuesto de palabras. Si una unidad pasa las {maxcall} palabras, dale subsecciones con \
título propio (una cada 1.000-2.500 palabras); si es corta, dejá "subsecciones": [].

Devolvé JSON:
{{"secciones": [{{"n": 1, "titulo": "...", "tesis": "qué sostiene esta unidad",
                  "contenido": "qué desarrolla, en 3-5 renglones",
                  "fuentes": ["Clave, año", "..."], "palabras": 000,
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
    run.log(f"[esquema] arquitectura: {len(units)} unidades, "
            f"{sum(int(u.get('palabras') or 0) for u in units)}p previstas")
    esquema = "\n".join(
        f"{u['n']}. {('[' + u['parte'] + '] ') if u.get('parte') else ''}{u['titulo']} "
        f"({u.get('palabras')}p) — {u.get('funcion', '')}" for u in units)

    secciones: list[dict] = []
    for i in range(0, len(units), 6):
        batch = units[i:i + 6]
        run.log(f"[esquema] detallando unidades {batch[0]['n']}-{batch[-1]['n']}…")
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
        run.log("[esquema] reutilizando 03_outline.json")
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
        run.log(f"[esquema] pidiendo el esquema completo ({nmin}-{nmax} secciones)…")
        outline = llm.chat_json(llm.FLASH, OUTLINE_PROMPT.format(
            hoy=hoy(), kind=spec["kind"], words=wrange(spec), registro=spec["register"],
            nmin=nmin, nmax=nmax,
            titulo=topic["titulo"], pregunta=topic.get("pregunta", ""),
            hipotesis=topic.get("hipotesis", ""), tension=topic.get("tension_teorica", ""),
            catalogo=catalogo,
            chapters_note=f"Si una sección pasa las {MAX_CALL_WORDS} palabras, dale "
                          "subsecciones con título propio; si no, dejá «subsecciones»: []."
            ) + _lang(), temperature=0.7)

    def show(o: dict) -> None:
        run.log(f"\n[esquema] {o.get('titulo_final')}")
        for s in o["secciones"]:
            run.log(f"  {s['n']}. {s['titulo']} ({s['palabras']}p)"
                    f" — {len(s.get('fuentes', []))} fuentes")

    show(outline)
    # PRO audits the plan before any words get written; a bad outline is the
    # most expensive thing to discover after 30k words of drafting.
    run.log("[esquema] auditoría del esquema por PRO…")
    critique = llm.chat_json(llm.PRO, f"""Auditá este esquema de un {spec['kind']}.
¿El argumento progresa o se repite? ¿La hipótesis se sostiene o se enuncia y nada más?
¿Hay secciones huecas? ¿Falta el contraargumento? ¿El reparto de fuentes es plausible?

{json.dumps(outline, ensure_ascii=False, indent=2)}

Devolvé JSON: {{"veredicto": "aprobado"|"corregir", "problemas": ["..."], \
"esquema_corregido": <el mismo objeto JSON del esquema, ya corregido>}}""" + _lang(),
temperature=0.4)
    # PRO sometimes answers a bare list: an unusable audit is no audit, keep the outline
    if not isinstance(critique, dict):
        critique = {}
    fixed = critique.get("esquema_corregido") or {}
    # A long outline handed back whole comes back truncated — half the sections
    # silently dropped. Take the correction only if it still has them all.
    if critique.get("veredicto") == "corregir" and \
            len(fixed.get("secciones") or []) >= len(outline["secciones"]):
        run.log("[esquema] PRO corrigió: " + "; ".join(critique.get("problemas", [])[:3]))
        outline = fixed
        show(outline)
    elif critique.get("veredicto") == "corregir":
        run.log("[esquema] PRO objetó pero devolvió un esquema incompleto; sigo con el mío: "
                + "; ".join(critique.get("problemas", [])[:3]))
    while run.mode != "auto":
        edit = run.ask("¿Cambios al esquema? (Enter = seguir, o describí qué cambiar):", "")
        if not edit:
            break
        outline = llm.chat_json(llm.FLASH,
            f"Aplicá estos cambios al esquema y devolvé el JSON completo corregido.\n"
            f"CAMBIOS: {edit}\n\nESQUEMA:\n{json.dumps(outline, ensure_ascii=False)}"
            f"{_lang()}", temperature=0.5)
        show(outline)
    run.save("03_outline.json", outline)
    return outline


# --------------------------------------------------------------------------- #
# 4. Draft
# --------------------------------------------------------------------------- #

SECTION_PROMPT = """{style_block}

# ENCARGO
Escribís la sección {n} de {total} de un {kind} titulado «{titulo}».
HIPÓTESIS GENERAL DEL TRABAJO: {hipotesis}

## Sección a escribir
Título: {sec_titulo}
Tesis de la sección: {sec_tesis}
Debe desarrollar: {sec_contenido}
{subsecciones}
Extensión: entre {pmin} y {pmax} palabras. Es un rango, no una meta: cualquier punto del
intervalo está bien. No rellenes para llegar al techo ni cortes antes del piso.

## REGISTRO DEL FORMATO — manda sobre todo lo demás
{registro}
Si alguna regla de estilo de más abajo choca con este registro, gana el registro.
Densidad de citas orientativa: {min_citas} citas distintas, sólo si hay suficientes
fuentes pertinentes. Cada cita sostiene una afirmación concreta; no cites una fuente
porque apenas roce el tema ni repitas la referencia tras cada frase de un mismo pasaje.
DENSIDAD DOCUMENTAL: al menos {min_datos} elementos del mundo —fechas exactas («el 15 de
septiembre de 1970», no «a principios de los setenta»), cifras con su unidad, nombres
propios de personas CON SU CARGO O FUNCIÓN («Richard Helms, director de la CIA»),
instituciones, lugares, títulos exactos de obras, números de página— todos sacados del
dossier. Una cita no es un dato: «(Postone, 1993)» no cuenta, «los 4,7 millones de
toneladas que Postone (1993, p. 287) toma del censo de 1968» sí. Esto es lo que separa
un texto documentado de uno que sólo administra bibliografía, y es lo primero que se
nota. Si el dossier no trae datos duros sobre un punto, escribí menos sobre ese punto.

## Lo que ya se dijo (no lo repitas; retomalo si hace falta)
{previo}

## Fórmulas ya usadas — PROHIBIDO volver a escribirlas
{evitar}
Ninguna imagen, comparación, anécdota, pregunta retórica ni remate puede aparecer dos
veces en todo el trabajo. Si ya se usó, se usa otra cosa o no se usa nada. Lo mismo con
las palabras poco frecuentes: si una ya cargó una frase, buscá otra.

## FUENTES — es TODO lo que existe. No podés citar nada que no esté acá.
{fuentes}

# REGLAS
- Escribí en el idioma solicitado, con la voz de las muestras correspondientes.
{apa}
- Las citas textuales de más de 40 palabras van en bloque aparte, sin comillas, y se dejan
  en su idioma original si la fuente está en inglés, alemán o francés.
- Nada de listas con viñetas, nada de negritas, nada de resúmenes al final de la sección.
- Cada sección hace avanzar el argumento desde lo que ya se estableció. Elegí la
  apertura según su función: un caso documentado, una pregunta precisa o una tesis.
  No repitas una plantilla de apertura, desarrollo y cierre en todas las secciones.
- Citá textualmente sólo cuando importe la formulación exacta y el pasaje esté
  disponible en las fuentes. No inventes una cita para cumplir una cuota.
- La longitud de oraciones y párrafos responde a la idea. Conservá las transiciones
  útiles y eliminá las frases que sólo anuncian o repiten lo que el lector ya sabe.
- TENÉ POSICIÓN, NO PLEITO. El texto sostiene una tesis y se le nota, pero no sale a
  buscar antagonistas:
  · nombrá con precisión la posición que discutís (una corriente, una hipótesis, un
    supuesto metodológico) y atribuísela a quien efectivamente la sostiene, con su cita;
    nada de «cierto discurso» ni «el sistema» en abstracto, y nada de retratos de
    autores como ingenuos, cómplices o vendedores de humo;
  · nada de ironía contra terceros, sarcasmo, motes ni frases de efecto («venden humo»,
    «astrología estadística», «con la elegancia de quien cruza un puente sin mirar el
    río»). Una imagen precisa vale más que una filosa, y si no aporta, sobra;
  · tomá partido con argumentos, no con adjetivos: la fuerza viene del dato y de la
    consecuencia que se sigue, no del tono;
  · cuando el propio argumento tiene un límite, decilo una vez, con calma, y seguí. La
    concesión es parte del argumento, no una confesión.
  · un comentario crítico breve cada tanto, no en cada párrafo. Si en una página hay
    más de una objeción a un autor, sobran objeciones.
- Usá puntuación y sintaxis naturales en el idioma solicitado. No fuerces una cuota
  de oraciones largas o breves ni evites un signo que aclare el argumento.
- HOY ES {hoy}. Escribís desde esta fecha: la coyuntura es la de ahora, no la del año
  en que se entrenó el modelo. Cuando el argumento toque política, economía, guerra o
  tecnología, usá el dato MÁS RECIENTE que haya en las fuentes (traen la fecha entre
  corchetes), citalo con su fecha —«el 14 de agosto», «los datos de julio»— y no cites
  una nota vieja habiendo una nueva sobre lo mismo. Nada de «recientemente» a secas.
  Tampoco inventes lo que pasó después de la última fuente.
- Si un dato no está en las fuentes, no lo afirmes. Podés marcar un hueco con [dato a verificar].
{aparato}
- Empezá directamente con el texto de la sección, sin encabezado ni título.

Escribí ahora."""


# Measured 2026-08-25 against six real windows of the author's corpus: what a
# published article has and a generated one never does is not rhythm, it is
# *machinery*. His prose carries footnotes with digressions and bibliographic
# strings, editorial brackets inside quotations, an English gloss in
# parentheses, two quote styles doing two different jobs, exact times of day.
# The draft comes out immaculate — one dash policy, one quote style, no notes,
# no visible seams — and immaculate reads as machine-written. None of this is
# invented: every note has to be dossier-backed like any other claim.
APPARATUS_RULES = """- APARATO AL PIE. Esta sección puede llevar hasta {notas_max} notas al pie,
  sólo cuando aporten una precisión necesaria que interrumpiría el argumento principal.
  Se marcan en el cuerpo con el número entre corchetes pegado a la palabra o al signo
  de puntuación —así[{nota_1}]— y la numeración de esta sección EMPIEZA EN {nota_1} y
  sigue de ahí. Al final de todo el texto de la sección, después de una línea que diga
  exactamente NOTAS:, va una nota por párrafo, con la forma «[{nota_1}] texto de la
  nota.». Una nota al pie NO es una cita suelta: es lo que no entra en el argumento y
  sin embargo hace falta —la precisión de una fecha, la historia de una traducción, el
  dato lateral, la aclaración de por qué una fuente dice lo que dice, un desacuerdo
  menor, la referencia completa de un documento. Van con su cita como todo lo demás.
- Conservá las citas literalmente. Usá corchetes editoriales sólo cuando una omisión
  o aclaración sea necesaria y fiel al pasaje, nunca para simular trabajo documental.
- Usá las convenciones tipográficas del idioma del artículo. Las cifras, términos
  técnicos y títulos conservan su precisión; no agregues adornos por variedad.
- LA MANO QUE ESCRIBE. Una vez en toda la sección, no más, el texto puede admitir algo
  sobre su propia construcción: qué fuente se consiguió sólo en un extracto, qué edición
  se leyó, qué no se pudo verificar, dónde se corta el material. Es una constatación
  de trabajo, seca, sin confesión ni queja: «de Kurz (1998) sólo hay a mano el primer
  capítulo, y lo que sigue se apoya en eso»."""


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
        return "(todavía no se escribió nada)"
    text = "\n".join(l for l in "\n\n".join(parts).split("\n") if not l.startswith("#"))
    short = [s.strip(" \n«»") for s in re.split(r"(?<=[.?!])\s+", text)
             if 3 <= len(s.split()) <= 12]
    phrases = list(dict.fromkeys(humanize.repeated_phrases(text) + short))
    return "\n".join(f"- «{p}»" for p in phrases[:limit])


def _chunks(sec: dict) -> list[tuple[str, str, int, str]]:
    """One drafting call each: (sufijo de caché, subtítulo, palabras, encabezado).

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
            label = sub if n_parts == 1 else f"{sub} (parte {j} de {n_parts})"
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
            run.log(f"[borrador] bajando el texto completo de {len(todo)} "
                    "fuente(s) citadas que llegaron sin texto…")
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
                run.log(f"[borrador] {label} en caché ({len(body.split())}p)")
            else:
                run.log(f"[borrador] {ui.progress_bar(int(sec['n']), len(secciones))} {label} "
                        f"({words}p)…")
                if sub:  # writing one subsection: show the others so it does not cover them
                    subs = ("Subsecciones de esta sección — escribís SOLO la marcada con →; "
                            "las otras las escribe otro y no hay que adelantarlas:\n"
                            + "\n".join(f"  {'→' if x == sub else ' '} {x}"
                                        for _, x, _, _ in chunks))
                else:
                    subs = ("Subsecciones (con sus títulos en el texto):\n"
                            + "\n".join(f"  - {x}" for x in sec["subsecciones"])
                            ) if sec.get("subsecciones") else ""
                # Inside a section the rolling synopsis is still one entry behind,
                # so the tail of the previous chunk is what makes the seam invisible.
                previo = synopsis or "(es la primera sección)"
                if bodies:
                    previo += ("\n\n(Lo último escrito, en esta misma sección, para "
                               f"empalmar: …{' '.join(bodies[-1].split()[-120:])})")
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
                run.log(f"[borrador] {label}: paso la sección a {rotated[0]}")
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
                    run.log(f"[borrador] me quedo con {len(best.split())} palabras (pedidas {words})")
                    return best
                run.log(f"[borrador] {count} palabras o idioma incorrecto "
                        f"(«{' '.join(body.split()[:15])}»); reintento")
                ask = prompt + (
                    f"\n\nLa respuesta anterior tenía {count} palabras. Escribí una versión "
                    f"completa de {int(words * 0.9)} a {int(words * 1.1)} palabras en "
                    f"{'inglés' if LANG == 'en' else 'español'}. Desarrollá las implicaciones "
                    "de las fuentes sin repetir ni inventar datos; comprobá la extensión.")
    finally:
        # _recover() inside chat may have repointed the chain; keep that choice.
        if llm.CHAIN == rotated:
            llm.CHAIN = chain
    raise RuntimeError(f"Sección {label}: ningún proveedor devolvió una sección completa "
                       f"en {LANG}. No se guardó en caché.")


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

    The model writes its notes after a bare `NOTAS:` line; they are collected
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


_TAIL_MARK = "\n\n(Últimas líneas escritas, para empalmar: …"


def _update_synopsis(previous: str, sec: dict, body: str) -> str:
    """Rolling memory. Keeps long-form runs coherent without resending everything.

    Only the *latest* section's tail is carried forward; older tails are dropped
    so the synopsis grows by one summary per section instead of ballooning.
    """
    tail = " ".join(body.split()[-120:])
    summary = llm.chat(llm.FLASH,
        f"Resumí en 90 palabras qué argumenta esta sección y a qué conclusión llega. "
        f"Prosa seca, sin adornos.\n\n{body[:12000]}", temperature=0.3)
    summaries = previous.split(_TAIL_MARK)[0].strip()
    entry = f"[{sec['n']}] {sec['titulo']}: {summary}"
    return f"{summaries}\n{entry}".strip() + _TAIL_MARK + f"{tail})"


# --------------------------------------------------------------------------- #
# 5. Review & approval
# --------------------------------------------------------------------------- #

REVIEW_PROMPT = """Sos un revisor de pares severo, especialista en teoría social crítica \
y crítica de la economía política. Revisá el siguiente {kind}.

HIPÓTESIS QUE DEBÍA DEFENDER: {hipotesis}

Citas que el texto usa y que SÍ existen en el dossier: {ok}
Citas que NO existen en el dossier (posibles invenciones): {bad}

FUENTES DISPONIBLES (comprobá las atribuciones contra estos pasajes; si sólo hay
resumen o un extracto, no supongas que verificaste el resto de la obra):
{fuentes}

Evaluá sin piedad:
- ¿La hipótesis se defiende con argumentos o solo se declama?
- ¿Hay errores conceptuales sobre Marx, la crítica del valor, la economía política?
- ¿Las citas sostienen lo que se les hace decir, o son decorativas?
- ¿El citado respeta APA 7 en español —(Apellido, año), (Apellido y Apellido, año),
  (Apellido et al., año), (Apellido, año, p. 302) para lo textual— o hay formatos mezclados?
- ¿Hay alguna mención de un estudio, informe, índice, encuesta o cifra SIN su cita al
  lado? ¿Aparecen instituciones o consultoras que no están en el dossier? ¿Quedan
  fórmulas sin referencia («un estudio reciente», «según informes», «se estima que»)?
  Listalas una por una: cada una es un problema de gravedad alta.
- ¿Falta el contraargumento fuerte? ¿La posición que discute está atribuida a alguien
  concreto y con su cita, o pelea contra abstracciones?
- DENSIDAD DOCUMENTAL: contá los elementos del mundo por cada mil palabras —fechas
  exactas, cifras con su unidad, nombres propios con su cargo, instituciones, lugares,
  títulos y páginas. Un párrafo entero sin ninguno, hecho sólo de nombres de teóricos y
  de sus posiciones, es problema de gravedad media: pedí que se ancle en el dato que la
  fuente trae, y si la fuente no trae ninguno, que se acorte el párrafo. Una cita no
  cuenta como dato.
- ¿Hay secciones que repiten lo mismo con otras palabras?
- REPETICIONES LITERALES ya detectadas por conteo: {repes}
  Descartá las que sean terminología del tema o nombres propios (el objeto de estudio se
  llama siempre igual, y está bien). El resto —fórmulas, imágenes, comparaciones,
  anécdotas, remates, preguntas retóricas— es problema de gravedad alta: reescribir o
  borrar la segunda aparición, nunca dejarla. Buscá también las que el conteo no vio.
- REGISTRO EXIGIDO POR EL FORMATO: {registro}
  ¿El texto lo respeta, o se desliza a otro tono?
- TONO: ¿el texto busca pelea? Marcá como problema de gravedad alta cada descalificación
  de un autor («no estoy de acuerdo con X», «X se queda en la superficie», «desconfío de
  X», «se queda a mitad de camino»), cada ironía contra terceros, cada frase de efecto y
  cada confesión personal enfática. Un comentario crítico sobrio cada tanto está bien;
  una objeción por párrafo, no. El desacuerdo va argumentado con datos, nunca con
  adjetivos ni con juicios sobre la persona.
- HOY ES {hoy}: ¿el texto habla de la coyuntura con los datos más recientes y con su
  fecha, o la trata como si el presente fuera hace un año? ¿Afirma hechos posteriores
  a las fuentes?

Devolvé JSON:
{{"veredicto": "aprobado"|"revisar"|"rechazado",
  "puntaje": 0-100,
  "fortalezas": ["..."],
  "problemas": [{{"seccion": "título o nº", "gravedad": "alta"|"media"|"baja",
                  "problema": "...", "correccion": "qué hacer exactamente"}}],
  "citas_a_eliminar": ["las que no se sostienen"],
  "falta_desarrollar": ["..."]}}

=== TEXTO ===
{text}"""


def review(run: Run, text: str, topic: dict, sources: list[research.Source]) -> dict:
    used, unknown = research.verify_citations(text, sources)
    run.log(f"[revisión] citas verificadas: {len(used)} válidas, {len(unknown)} sin respaldo")
    if unknown:
        run.log("  · sin respaldo: " + ", ".join(unknown[:10]))
    repes = humanize.repeated_phrases(text)
    if repes:
        run.log(f"[revisión] {len(repes)} frase(s) repetidas: «{repes[0]}»…")
    run.log("[revisión] informe de PRO sobre el borrador…")
    rep = llm.chat_json(llm.PRO, REVIEW_PROMPT.format(
        kind=FORMATS[run.fmt]["kind"], registro=FORMATS[run.fmt]["register"],
        repes="; ".join(f"«{r}»" for r in repes) or "(ninguna)",
        hoy=hoy(), hipotesis=topic.get("hipotesis", ""),
        ok=", ".join(used[:60]) or "(ninguna)",
        fuentes=_sources_block(sources, used, 30000),
        bad=", ".join(unknown[:30]) or "(ninguna)", text=text[:120000]) + _lang(),
temperature=0.35)
    rep["citas_validas"] = used
    rep["citas_sin_respaldo"] = unknown
    run.log(f"[revisión] veredicto: {rep.get('veredicto')} ({rep.get('puntaje')}/100), "
            f"{len(rep.get('problemas', []))} problemas")
    return rep


REVISE_PROMPT = """Corregí el texto según el informe de revisión, con parches quirúrgicos: \
no lo reescribas entero, devolvé solo los fragmentos que cambian. Cambiá SOLO lo señalado; \
no toques lo que funciona ni el estilo.

INFORME:
{report}

REGLA INNEGOCIABLE SOBRE CITAS: estas citas están verificadas contra la bibliografía y \
TIENEN que seguir en el texto corregido —{ok}—. Si el informe dice que una cita es \
decorativa, la solución NO es borrarla: es hacerla trabajar, mostrando qué sostiene la \
fuente y por qué importa para el argumento. Un parche que se lleve puestas citas \
verificadas, o que introduzca una que no está en las fuentes de abajo, se descarta entero.

Citas sin respaldo bibliográfico — esas sí, eliminalas o reemplazalas por alguna de las \
claves verificadas de arriba.

Normas de citado (APA 7, español) que el texto corregido debe respetar:
{apa}

FUENTES DISPONIBLES (única bibliografía existente):
{fuentes}

=== TEXTO ===
{text}
=== FIN ===

Devolvé JSON: {{"parches": [{{"buscar": "…", "reemplazar": "…"}}]}}
- `buscar` es una copia LITERAL del texto de arriba —mismas palabras, mismos acentos, \
misma puntuación— y lo bastante larga para aparecer UNA sola vez: la oración entera, o \
el párrafo si la oración se repite. Si no aparece tal cual, o aparece dos veces, el \
parche se pierde.
- `reemplazar` es ese mismo fragmento ya corregido; "" lo borra.
- Un parche por problema, y nada más que los problemas del informe."""


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
    if not problems and not report.get("citas_sin_respaldo"):
        return text
    run.log(f"[revisión] aplicando {len(problems)} correcciones…")
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
                           "citas_a_eliminar": report.get("citas_sin_respaldo", []),
                           "falta_desarrollar": report.get("falta_desarrollar", [])},
                          ensure_ascii=False, indent=2),
        ok=", ".join(valid[:60]) or "(ninguna todavía)", apa=APA_RULES,
        fuentes=fuentes, text=text) + _lang(), temperature=0.5)
    patches = answer.get("parches") or []
    revised, applied, lost, missed = text, 0, 0, 0
    for p in patches:
        find, repl = (p.get("buscar") or ""), (p.get("reemplazar") or "")
        if not find.strip():
            continue
        # A revision that strips the bibliography is worse than no revision at all:
        # "this citation is decorative" must not be answered by deleting the citation,
        # and a correction must not invent a source to cite either.
        had, _ = research.verify_citations(find, sources)
        keeps, unknown = research.verify_citations(repl, sources)
        if set(had) - set(keeps) or unknown:
            lost += 1
            continue
        out = _apply_patch(revised, find, repl)
        if out is None:
            missed += 1
            continue
        revised, applied = out, applied + 1
    detail = "".join([f", {lost} descartado(s) por citas" if lost else "",
                      f", {missed} sin coincidencia exacta" if missed else ""])
    run.log(f"[revisión] {applied}/{len(patches)} parches aplicados{detail}")
    return revised


def final_approval(run: Run, text: str, topic: dict, det: dict) -> dict:
    run.log("[aprobación] dictamen final de PRO…")
    verdict = llm.chat_json(llm.PRO, f"""Aprobación final. Decidí si este texto se publica.

Criterios: solidez del argumento, honestidad bibliográfica, coherencia interna, \
que la prosa suene a un autor humano con voz propia y no a un modelo de lenguaje.

Puntaje del detector de IA (0 = humano, 100 = máquina): {det.get('final_score')}

Devolvé JSON: {{"publicable": true|false, "puntaje": 0-100, \
"dictamen": "3-6 renglones", "ajustes_menores": ["..."]}}

TÍTULO: {topic['titulo']}

{text[:100000]}""" + _lang(), temperature=0.3)
    if not isinstance(verdict, dict):
        verdict = {"dictamen": f"respuesta ilegible: {str(verdict)[:200]}"}
    # «"publicable": "false"» is a truthy string: read as approval it skipped the
    # edit gate and let `--publish auto` go live. Only an explicit yes counts.
    said = verdict.get("publicable")
    verdict["publicable"] = said is True or str(said).strip().lower() in ("true", "sí", "si", "yes")
    run.log(f"[aprobación] publicable={verdict.get('publicable')} "
            f"({verdict.get('puntaje')}/100)")
    return verdict


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

def assemble(run: Run, topic: dict, outline: dict, text: str,
             sources: list[research.Source], report: dict, det: dict,
             verdict: dict) -> str:
    spec = FORMATS[run.fmt]
    used, _ = research.verify_citations(text, sources)
    today = dt.date.today()
    en = LANG == "en"
    month = ("January February March April May June July August September October "
             "November December").split()[today.month - 1]
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
        raise RuntimeError("No se consiguió ninguna fuente. Revisá la conexión.")
    outline = make_outline(run, topic, sources)
    text = draft(run, topic, outline, sources)
    run.save("04_draft.md", text)
    run.log(f"[borrador] {len(text.split())} palabras")

    report: dict = {}
    for i in range(1, rounds + 1):
        # Both halves of a round are cached: the report costs a PRO call over the whole
        # draft, and a resume that re-derives it pays it again for the same verdict.
        # The report is saved before the corrections are applied, so a crash mid-revise
        # resumes on the corrections instead of re-reviewing.
        if (cached := run.load(f"06_review_r{i}.json")):
            run.log(f"[revisión] reutilizando 06_review_r{i}.json (ronda {i})")
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
            run.log(f"[revisión] reutilizando 04_draft_rev{i}.md")
            text = done
            continue
        if run.mode != "auto":
            for p in report.get("problemas", [])[:8]:
                run.log(f"  · [{p.get('gravedad')}] {p.get('seccion')}: {p.get('problema')}")
            if run.ask("¿Aplico las correcciones? [S/n]", "s").lower().startswith("n"):
                break
        text = revise(run, text, report, sources)
        run.save(f"04_draft_rev{i}.md", text)

    cached_det = run.load("07_detector_humanizado.json") or run.load("07_detector.json")
    if ((cached := run.load("04_draft_humanizado.md")) and cached_det
            and cached_det.get("lang") == LANG
            and cached_det.get("text_hash") == humanize.text_hash(cached)):
        run.log("[detector] reutilizando 04_draft_humanizado.md")
        text, det = cached, cached_det
        score = det.get("final_score")
        det["passed"] = score < threshold if score is not None else None
        used, unknown = research.verify_citations(text, sources)
    else:
        run.log("[detector] pasando por detectores de IA…")
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
            run.log("[detector] la reescritura perdió o inventó citas; "
                    "me quedo con el borrador revisado")
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
        run.log("[revisión] reutilizando 04_draft_corregido.md")
        text = fixed
    elif unknown:
        run.log(f"[detector] la reescritura dejó {len(unknown)} citas sin respaldo; corrigiendo")
        text = revise(run, text, report, sources)
        run.save("04_draft_corregido.md", text)
    if fixed or unknown:
        used, unknown = research.verify_citations(text, sources)
        report["citas_validas"], report["citas_sin_respaldo"] = used, unknown

    if det.get("text_hash") != humanize.text_hash(text):
        _, det = humanize.humanize(text, rounds=1 if detector_rounds > 0 else 0,
                                   threshold=threshold, log=run.log, lang=LANG)
    run.save("07_detector.json", det)
    verdict = final_approval(run, text, topic, det)
    if unknown:
        verdict.update(publicable=False, dictamen="Quedan citas sin respaldo bibliográfico.")
    # PRO's rejections are usually minutes of hand editing ("[dato a verificar]" left in,
    # a year that reads 1993a in one section and 1993 in the next). Let the user fix them
    # in place and ask PRO again instead of throwing the run away or publishing anyway.
    while not verdict.get("publicable") and run.mode != "auto":
        run.log("[aprobación] " + str(verdict.get("dictamen", "")))
        for adj in (verdict.get("ajustes_menores") or [])[:8]:
            run.log(f"  · {adj}")
        # The edited file is the corrected body, not the humanized draft: it is what the
        # resume branch above reads back, so an edit outlives a Ctrl-C at this prompt.
        run.save("04_draft_corregido.md", text)
        choice = run.ask(f"No aprobado. [e]ditar {run.dir / '04_draft_corregido.md'} "
                         "y volver a revisar, [p]ublicar igual, [n]o publicar [e/p/N]",
                         "n").strip().lower()[:1]
        if choice != "e":
            if choice != "p":
                run.log("[aprobación] guardado como borrador, sin aprobar.")
            break
        run.ask("Editá el archivo y dale Enter cuando esté listo…", "")
        text = run.load("04_draft_corregido.md") or text
        used, unknown = research.verify_citations(text, sources)
        report["citas_validas"], report["citas_sin_respaldo"] = used, unknown
        if unknown:
            run.log(f"[aprobación] tu edición dejó {len(unknown)} citas sin respaldo: "
                    + ", ".join(unknown[:6]))
        if det.get("text_hash") != humanize.text_hash(text):
            _, det = humanize.humanize(text, rounds=1 if detector_rounds > 0 else 0,
                                       threshold=threshold, log=run.log, lang=LANG)
        verdict = final_approval(run, text, topic, det)
        if unknown:
            verdict.update(publicable=False, dictamen="Quedan citas sin respaldo bibliográfico.")
    doc = assemble(run, topic, outline, text, sources, report, det, verdict)
    out = run.dir / "05_final.md"
    # What the publisher needs to decide draft-vs-live, without re-reading the run.
    run.state.update(topic=topic, verdict=verdict, detector=det, final=out)
    run.log(textwrap.dedent(f"""
        ─────────────────────────────────────────
        Listo: {out}
        Palabras: {len(doc.split())}
        Fuentes citadas: {len(report.get('citas_validas', []))} de {len(sources)} reunidas
        Detector IA: {det.get('final_score')} (umbral {threshold})
        Aprobación: {verdict.get('puntaje')}/100
        ─────────────────────────────────────────"""))
    return out
