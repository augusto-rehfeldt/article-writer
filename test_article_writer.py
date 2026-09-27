"""Offline self-check. No network, no API key, no models.

Covers the logic that would silently corrupt an article if it broke: citation
verification, key assignment, dedup, stylometry and the tolerant JSON parser.

    python test_article_writer.py
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import humanize
import llm
import pipeline
import publish
import research
import style

class _FakeService:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def generate_content(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _shared(replies: dict):
    """Swap book writer's AIService for fakes; yields [(provider, overrides, calls)]."""
    import contextlib
    from unittest.mock import patch

    @contextlib.contextmanager
    def ctx():
        seen = []

        def factory(provider, overrides):
            service = _FakeService(replies[provider])
            seen.append((provider, dict(overrides), service.calls))
            return service
        with patch.object(llm, "shared_service", side_effect=factory):
            yield seen
    return ctx()


def test_assign_keys_disambiguates() -> None:
    srcs = [research.Source(title="A", authors=["Robert Kurz"], year="1998"),
            research.Source(title="B", authors=["Robert Kurz"], year="1998"),
            research.Source(title="C", authors=["Moishe Postone"], year="2006"),
            research.Source(title="D", authors=[], year="")]
    research.assign_keys(srcs)
    assert [s.key for s in srcs] == ["Kurz, 1998a", "Kurz, 1998b", "Postone, 2006", "Anon, s/f"], \
        [s.key for s in srcs]


def test_dedupe_by_doi_then_url_then_title() -> None:
    srcs = [research.Source(title="X", doi="10.1/a"),
            research.Source(title="Otro titulo", doi="10.1/A"),   # same DOI, different case
            research.Source(title="X"),                            # same title as the first
            research.Source(title="Y", url="http://z")]
    assert len(research.dedupe(srcs)) == 3


def test_verify_citations_flags_hallucinations() -> None:
    srcs = research.assign_keys([
        research.Source(title="Tiempo, trabajo y dominación social",
                        authors=["Moishe Postone"], year="2006"),
        research.Source(title="El colapso de la modernización",
                        authors=["Robert Kurz"], year="2016")])
    text = ("Como sostiene Postone, el valor es histórico (Postone, 2006: 302), "
            "y en la misma línea (Kurz, 2016: 14). Pero (Habermas, 1981: 55) no "
            "está en el dossier, ni (Fulanez, 1999).")
    used, unknown = research.verify_citations(text, srcs)
    assert used == ["Kurz, 2016", "Postone, 2006"], used
    assert len(unknown) == 2 and any("Habermas" in u for u in unknown), unknown


def test_verify_citations_matches_on_surname_only() -> None:
    srcs = research.assign_keys([research.Source(
        title="T", authors=["Moishe Postone"], year="2006")])
    # A different year for a known author must NOT silently pass as the same work,
    # but the surname is known, so it resolves rather than being flagged as invented.
    used, unknown = research.verify_citations("segun (Moishe Postone, 2006: 12)", srcs)
    assert used == ["Postone, 2006"] and not unknown, (used, unknown)


def test_verify_citations_ignores_accents_and_case() -> None:
    srcs = research.assign_keys([research.Source(
        title="T", authors=["Slavoj Žižek"], year="2009")])
    used, unknown = research.verify_citations("como dice (Zizek, 2009) y ZIZEK (2009) repite", srcs)
    assert used == ["Žižek, 2009"] and not unknown, (used, unknown)


def test_a_section_near_its_budget_is_kept_not_fatal() -> None:
    """Two replies at 1.2x used to raise and end an hours-long run; only an
    amputated section or one in the wrong language still stops it."""
    sec = {"n": 1, "titulo": "Uno", "tesis": "t", "contenido": "c",
           "fuentes": [], "palabras": 400}
    outline = {"titulo_final": "T", "secciones": [sec]}
    saved = (llm.chat, style.style_block, llm.CHAIN)
    style.style_block = lambda *a, **kw: "ESTILO"
    try:
        # A link that keeps answering stubs hands the section to the backup.
        llm.CHAIN = ["hyper", "zen"]
        heads: list[str] = []
        def chat(*a, **kw):
            heads.append(llm.CHAIN[0])
            return "Doce palabras." if llm.CHAIN[0] == "hyper" else "Cuerpo escrito y breve. " * 100
        llm.chat = chat
        with tempfile.TemporaryDirectory() as d:
            pipeline.draft(pipeline.Run(dir=pathlib.Path(d), log=lambda *_: None),
                           {"titulo": "T"}, outline, [])
            assert (pathlib.Path(d) / "04_sec01.md").exists()
        assert heads == ["hyper"] * pipeline.DRAFT_TRIES + ["zen"], heads
        assert llm.CHAIN == ["hyper", "zen"], "the chain was left rotated"
        for reply, fatal in (("Cuerpo escrito y breve. " * 120, False),   # 480 of 400
                             ("Cuerpo escrito y breve. " * 40, True)):    # 160 of 400
            llm.chat = lambda m, p, *a, reply=reply, **kw: reply
            with tempfile.TemporaryDirectory() as d:
                run = pipeline.Run(dir=pathlib.Path(d), log=lambda *_: None)
                try:
                    pipeline.draft(run, {"titulo": "T"}, outline, [])
                    assert not fatal, "an amputated section was accepted"
                    assert (pathlib.Path(d) / "04_sec01.md").exists()
                except RuntimeError:
                    assert fatal, "a section 20% over budget ended the run"
                    assert not (pathlib.Path(d) / "04_sec01.md").exists()
    finally:
        llm.chat, style.style_block, llm.CHAIN = saved


def test_fingerprint_detects_uniform_prose() -> None:
    uniform = " ".join(["Esta oracion tiene exactamente ocho palabras aqui mismo."] * 30)
    varied = ("Corto. " + "Un periodo largo que se extiende con subordinadas encadenadas "
              "y que no termina nunca porque el autor prefiere acumular clausulas antes "
              "que cortar la respiracion del lector que lo sigue. " + "Nada mas. ") * 10
    assert style.fingerprint(uniform)["burstiness"] < 0.15
    assert style.fingerprint(varied)["burstiness"] > 0.5


def test_corpus_fingerprint_is_bursty() -> None:
    """The author's real prose must land well above the LLM-flat baseline."""
    if not style.FINGERPRINT.exists():
        print("  (saltado: falta style_fingerprint.json — corré --setup)")
        return
    fp = style.corpus_fingerprint()
    assert fp["burstiness"] > 0.5, fp["burstiness"]
    assert fp["words"] > 10000, fp["words"]


def test_json_candidates_survive_fences_and_prose() -> None:
    payloads = ['{"a": 1}',
                'Claro:\n```json\n{"a": 1}\n```\nEso es todo.',
                'Aqui va: {"a": 1} listo']
    for p in payloads:
        got = next(json.loads(c) for c in llm._json_candidates(p)
                   if _parses(c))
        assert got == {"a": 1}, p


def _parses(candidate: str) -> bool:
    try:
        json.loads(candidate)
        return True
    except json.JSONDecodeError:
        return False


def test_slugify() -> None:
    assert pipeline.slugify("¿Qué es la «crítica del valor»?") == "que-es-la-critica-del-valor"


def test_formats_cover_every_requested_length() -> None:
    for name in ("corto", "medio", "largo", "paper", "tesis", "libro", "discusion"):
        assert name in pipeline.FORMATS
        spec = pipeline.FORMATS[name]
        lo, hi = spec["words"]                      # a range, never a single target
        nlo, nhi = spec["sections"]                 # so is the section count
        assert 0 < lo < hi and 0 < nlo < nhi
        assert hi / nhi >= 400                      # no section budgeted into nothing
        assert lo / nlo <= 10000                    # nor past the per-section ceiling
        assert spec["register"].strip()
        assert pipeline.wrange(spec) == f"{lo:,} y {hi:,}".replace(",", ".")
        assert lo < pipeline.wmid(spec) < hi


def test_apa_name_inverts_and_initialises() -> None:
    assert research._apa_name("Moishe Postone") == "Postone, M."
    assert research._apa_name("Karl Heinrich Marx") == "Marx, K. H."
    assert research._apa_name("Postone, M.") == "Postone, M."   # already APA
    assert research._apa_name("Wikipedia") == "Wikipedia"       # institutional


def test_source_citation_is_apa() -> None:
    s = research.Source(title="Tiempo, trabajo y dominación social",
                        authors=["Moishe Postone"], year="2006",
                        kind="book", venue="Marcial Pons", doi="10.1/x")
    cit = s.citation()
    assert cit.startswith("Postone, M. (2006).")
    assert "*Tiempo, trabajo y dominación social*" in cit
    assert "https://doi.org/10.1/x" in cit


def test_citation_joins_authors_with_y() -> None:
    s = research.Source(title="T", authors=["Ana Uno", "Beto Dos"], year="2020",
                        kind="paper", venue="Revista")
    assert "Uno, A. y Dos, B. (2020)." in s.citation()
    assert "*Revista*" in s.citation()


def test_verify_citations_handles_apa_variants() -> None:
    srcs = research.assign_keys([
        research.Source(title="A", authors=["Moishe Postone"], year="2006"),
        research.Source(title="B", authors=["Robert Kurz", "Anselm Jappe"], year="2016"),
        research.Source(title="C", authors=["Anselm Jappe"], year="2011")])
    text = ("(Postone, 2006, p. 302) y (Kurz y Jappe, 2016) y (Postone et al., 2006) "
            "y la vieja forma (Kurz, 2016: s/n). Pero (Inventado, 2001, p. 5) no.")
    used, unknown = research.verify_citations(text, srcs)
    assert set(used) == {"Postone, 2006", "Kurz, 2016"}, used
    assert unknown == ["(Inventado, 2001, p. 5)"], unknown


def test_verify_citations_catches_narrative_form() -> None:
    """"Postone (1993) sostiene" is a citation too — and just as forgeable."""
    srcs = research.assign_keys([
        research.Source(title="A", authors=["Moishe Postone"], year="1993"),
        research.Source(title="B", authors=["Robert Kurz", "Anselm Jappe"], year="2016")])
    text = ("Postone (1993) ha mostrado que el trabajo media. Kurz y Jappe (2016) "
            "discuten. Pero Fulanez (2001) no existe. La Cultura (1987) no es una cita.")
    used, unknown = research.verify_citations(text, srcs)
    assert set(used) == {"Postone, 1993", "Kurz, 2016"}, used
    assert unknown == ["Fulanez (2001)"], unknown


def test_have_title_needs_real_fulltext() -> None:
    stub = research.Source(title="Tiempo, trabajo y dominación social",
                           authors=["Postone"], abstract="x" * 400)
    assert not research._have_title([stub], "Tiempo, trabajo y dominación social")
    stub.fulltext = "y" * 4000
    assert research._have_title([stub], "Tiempo, trabajo y dominación social")


def test_have_title_matches_author_first_requests() -> None:
    # The wanted list is written "Autor, Título (año)" while a catalogue record keeps
    # the author apart from the title; matching on a concatenated prefix never hit,
    # so every book came back missing even when its full text was already in hand.
    got = research.Source(title="Time, Labor, and Social Domination: A Reinterpretation",
                          authors=["Moishe Postone"], fulltext="y" * 4000)
    assert research._have_title([got], "Postone, Time, Labor, and Social Domination (1993)")
    assert not research._have_title([got], "Kurz, Geld ohne Wert (2012)")


def test_covers_rejects_a_catalogue_false_positive() -> None:
    # archive.org answered "Kurz, Geld ohne Wert" with a declassified CIA cable;
    # a book filed under the wrong title poisons the dossier with fake citations.
    assert not research._covers("Kurz, Geld ohne Wert (2012)",
                                "CIA Declassified Germany Central Intelligence Agency report")
    assert research._covers("Karl Marx, El capital",
                            "El Capital (Libros completos en español) Karl Marx")
    # Two different books by the same author must not collapse into one gap.
    assert not research._covers("Robert Kurz, Dinero sin valor",
                                "Robert Kurz, El colapso de la modernización")


def test_split_wanted_pulls_author_title_year() -> None:
    assert research._split_wanted("Kurz, Geld ohne Wert (2012)") == (
        "Kurz", "Geld ohne Wert", "2012")
    assert research._split_wanted("El capital") == ("", "El capital", "")
    # Quoted article title: the commas before it belong to the author list, not to
    # the title, and the journal after it is not part of it either.
    assert research._split_wanted(
        'Marin, Frédéric y Beluffi, Camille, «Computing the minimal crew», JBIS (2018)'
    ) == ("Marin, Frédéric y Beluffi, Camille", "Computing the minimal crew", "2018")


def test_fallback_links_cover_every_gap() -> None:
    links = research._fallback_links("Postone Tiempo y trabajo")
    assert len(links) >= 5
    assert all(l["url"].startswith("http") and " " not in l["url"] for l in links)


def test_bibliography_only_lists_cited_works() -> None:
    srcs = research.assign_keys([
        research.Source(title="Usada", authors=["Zoe Zeta"], year="2020"),
        research.Source(title="No usada", authors=["Ana Alfa"], year="2019")])
    bib = research.bibliography(srcs, ["Zeta, 2020"])
    assert "Usada" in bib and "No usada" not in bib


def test_bibliography_collapses_the_same_work_cited_twice() -> None:
    srcs = research.assign_keys([
        research.Source(title="Time, Labor, and Social Domination",
                        authors=["Moishe Postone"], year="1993",
                        doi="10.1017/cbo9780511570926"),
        research.Source(title="Time, Labor, and Social Domination",
                        authors=["Moishe Postone"], year="s.f.", kind="local")])
    bib = research.bibliography(srcs, list({s.key for s in srcs}))
    assert bib.count("Postone") == 1 and "doi.org" in bib


def test_citation_drops_aggregator_venues() -> None:
    """OpenAlex names the hosting archive, not the journal; «biblioteca local»
    is not a publisher. Neither may reach the reference list."""
    for venue in ("LA Referencia (Red Federada de Repositorios)",
                  "Project Muse (Johns Hopkins University)",
                  "NASA STI Repository (National Aeronautics and Space Administration)",
                  "biblioteca local"):
        s = research.Source(title="T", authors=["A Alfa"], year="2020",
                            kind="paper", venue=venue, doi="10.1/x")
        assert venue not in s.citation() and "doi.org" in s.citation()
    real = research.Source(title="T", authors=["A Alfa"], year="2020",
                           kind="paper", venue="Utopian Studies", doi="10.1/x")
    assert "*Utopian Studies*" in real.citation()


def test_strip_preamble_removes_model_heading() -> None:
    assert pipeline._strip_preamble("## Titulo\n\nTexto real.") == "Texto real."
    assert pipeline._strip_preamble("Texto real.") == "Texto real."


def test_strip_preamble_removes_a_note_to_self_above_a_rule() -> None:
    # The shape sec08 of the 20260824 run actually shipped with.
    leaked = ("Two-front section, dossier tight (Scholz x2, Varela). Writing now.\n\n"
              "---\n\nRoswitha Scholz firma el texto en 2013.")
    assert pipeline._strip_preamble(leaked) == "Roswitha Scholz firma el texto en 2013."
    # A rule below real prose is the author's own; it is not a preamble.
    kept = "Un párrafo entero de prosa " + "larga " * 70 + ".\n\n---\n\nY sigue."
    assert pipeline._strip_preamble(kept) == kept


def test_notes_are_collected_out_of_the_body() -> None:
    body, notes = pipeline._split_notes(
        "El argumento va acá[1], y sigue.\n\nNOTAS:\n[1] La aclaración lateral.")
    assert body == "El argumento va acá[1], y sigue."
    assert notes == ["[1] La aclaración lateral."]
    # A chunk cached by an older run carries no marker and must survive intact.
    assert pipeline._split_notes("Sin notas.") == ("Sin notas.", [])


def test_a_format_without_apparatus_gets_no_footnotes() -> None:
    corto = pipeline._apparatus_rules(pipeline.FORMATS["corto"], 1400, 1)
    assert corto == ""
    paper = pipeline._apparatus_rules(pipeline.FORMATS["paper"], 1400, 7)
    assert "NOTAS:" in paper and "[7]" in paper   # numbering continues across sections


def test_local_score_separates_the_floors_from_the_real_defects() -> None:
    # `issues_texto` is `issues` minus the messages that only say «this metric is
    # below the corpus floor» — the ones that correlate -0.85 with the judges.
    # The score counts both; the split is what lets the rewriter be fed either.
    flat = ("La forma piensa y el valor se mide en el mercado abierto. " * 40)
    d = humanize.local_score(flat, lang="es")
    assert d["score"] > 0 and len(d["issues_texto"]) < len(d["issues"])
    assert not any("piso humano" in m or "burstiness" in m for m in d["issues_texto"])


def test_chunks_splits_a_long_section_by_subsection() -> None:
    short = pipeline._chunks({"n": 3, "palabras": 1800, "subsecciones": ["a", "b"]})
    assert short == [("03", "", 1800, "")]              # one call, cache name unchanged

    long = pipeline._chunks({"n": 12, "palabras": 6000,
                             "subsecciones": ["a", "b", "c", " "]})
    assert [c[0] for c in long] == ["12_01", "12_02", "12_03"]   # blank subsection dropped
    assert sum(c[2] for c in long) == 6000                       # budget preserved
    assert all(c[2] <= pipeline.MAX_CALL_WORDS for c in long)
    assert [c[3] for c in long] == ["### a", "### b", "### c"]

    # a subsection still over the ceiling is written in parts, headed only once
    huge = pipeline._chunks({"n": 4, "palabras": 12000, "subsecciones": ["x", "y"]})
    assert [c[0] for c in huge] == ["04_01a", "04_01b", "04_01c",
                                    "04_02a", "04_02b", "04_02c"]
    assert huge[1][1] == "x (parte 2 de 3)" and huge[1][3] == ""
    assert all(c[2] <= pipeline.MAX_CALL_WORDS for c in huge)

    # long but unsplittable: no subsections to cut on, so it stays one call
    assert pipeline._chunks({"n": 1, "palabras": 6000}) == [("01", "", 6000, "")]


def test_judges_are_the_fixed_panel_on_their_own_providers() -> None:
    """opus 5.5 and gpt-6-astra judge whatever the chain; each goes only to its owner."""
    import llm
    saved = (llm.CHAIN, llm.PRO, llm.FLASH, llm.JUDGES)
    try:
        llm.configure(["hyper"], pro="qwen3.7-max", flash="deepseek-v4-flash-0731")
        assert llm.JUDGES == ["claude-opus-5-5", "gpt-6-astra"]
        assert [b for b in llm.PROVIDERS if llm._serves(b, "gpt-6-astra")] == ["oauth"]
        assert [b for b in llm.PROVIDERS if llm._serves(b, "claude-opus-5-5")] == ["claude"]
    finally:
        llm.CHAIN, llm.PRO, llm.FLASH, llm.JUDGES = saved


def test_failed_judges_are_skipped_and_all_failing_falls_back_to_the_drafter() -> None:
    import llm
    old, dead = llm.chat_json, set()

    def fake(m, p, *a, **kw):
        if m in dead:
            raise RuntimeError("caído")
        return {"ai_probability": 40, "veredicto": "ia"}
    llm.chat_json = fake
    try:
        dead = {"gpt-6-astra"}
        got = humanize.llm_judges("texto", log=lambda *_: None)
        assert [r["model"] for r in got] == ["claude-opus-5-5"], got
        dead = set(llm.JUDGES)
        got = humanize.llm_judges("texto", log=lambda *_: None)
        drafter = llm.PRO if pipeline.DRAFT_ROLE == "pro" else llm.FLASH
        assert [r["model"] for r in got] == [drafter], got
        # an explicit panel (the benches) never gets a stand-in
        assert humanize.llm_judges("texto", ["gpt-6-astra"], lambda *_: None) == []
    finally:
        llm.chat_json = old


def test_continuous_mode_resumes_a_crashed_run_once() -> None:
    """Stages are cached: a run that died after research is retried, not discarded."""
    import argparse, pathlib, main
    seen, args = [], argparse.Namespace(
        continuo=3, cada=0, fmt="medio", mode="auto", sin_biblioteca=True,
        tema="", tema_exacto=False, no_detector=True, threshold=25.0,
        rounds=1, detector_rounds=0, publicar="no")

    def fake_run_once(a, resume="", run=None):
        seen.append(run)
        run.dir = pathlib.Path("output/roto")     # got past the topic stage
        raise RuntimeError("proveedor caído")

    saved, slept = (main.run_once, main.time.sleep), []
    main.run_once, main.time.sleep = fake_run_once, slept.append
    try:
        main.loop(args)
        assert len(seen) == 3
        assert seen[0] is seen[1], "la corrida caída tiene que retomarse"
        assert seen[2] is not seen[1], "sólo un reintento: dos caídas es rota"
        assert slept and all(s >= 60 for s in slept), "--every 0 no puede girar en vacío"
    finally:
        main.run_once, main.time.sleep = saved


def test_opencode_alias_maps_versioned_ids() -> None:
    import llm
    assert llm.OPENCODE_ALIAS["deepseek-v4-pro-0813"] == "deepseek-v4-pro"
    assert llm._as("go", "qwen3.8-max") == "qwen3.8-max"   # unlisted = as typed


def test_chain_translates_roles_into_each_provider_catalogue() -> None:
    """A backup provider has its own names: sending `opus` to hyper is a 404."""
    saved = (llm.CHAIN, llm.PRO, llm.FLASH, llm.JUDGES)
    try:
        llm.configure(["claude", "hyper", "go"])
        assert (llm.PRO, llm.FLASH) == ("claude-opus-5-5", "sonnet")
        assert llm._as("claude", llm.PRO) == "claude-opus-5-5"
        assert llm._as("hyper", llm.PRO) == "qwen3.8-flash"
        assert llm._as("go", llm.FLASH) == "deepseek-v4.1-flash"
        assert llm._serves("claude", llm.PRO) and llm._serves("claude", llm.FLASH)
        assert not llm._serves("claude", "glm-5.2")
        assert llm._serves("hyper", "glm-5.2")
        llm.configure(["hyper"], pro="qwen3.7-max")
        assert llm.PRO == "qwen3.7-max" and llm.FLASH == "deepseek-v4.1-flash"
    finally:
        llm.CHAIN, llm.PRO, llm.FLASH, llm.JUDGES = saved


def test_a_rotated_chain_still_translates_the_roles() -> None:
    """A section handed to a backup rotates CHAIN; the backup must get its own
    model, not the head's id (claude and hyper were sent a zen-only model)."""
    saved = (llm.CHAIN, llm.HEAD, llm.PRO, llm.FLASH, llm.JUDGES)
    try:
        llm.configure(["zen", "claude", "hyper"], pro="muse-spark-1.3-contributor-free")
        llm.CHAIN = ["claude", "hyper", "zen"]
        assert llm._as("claude", llm.PRO) == "claude-opus-5-5"
        assert llm._as("hyper", llm.PRO) == "qwen3.8-flash"
        assert llm._as("zen", llm.PRO) == "muse-spark-1.3-contributor-free"
    finally:
        llm.CHAIN, llm.HEAD, llm.PRO, llm.FLASH, llm.JUDGES = saved


def test_a_claude_quota_notice_is_not_a_completion() -> None:
    """`claude -p` prints «You've hit your session limit» or an expired-login notice
    on stdout and exits 0. Taken as prose it got drafted into the article and
    parsed as JSON, so the link fails and the chain walks on."""
    for notice in ("You've hit your session limit · resets 9:10am (America/Argentina/Buenos_Aires)",
                   "Failed to authenticate: OAuth session expired and could not be refreshed"):
        with _shared({"claude": [notice]}):
            try:
                llm._send("claude", "opus", "escribí una sección", None,
                          temperature=None, max_tokens=None, retries=1)
                raise AssertionError("una notificación de cuota no es una respuesta")
            except RuntimeError as e:
                assert "sin cuota" in str(e), e
    # ...and a real article that happens to mention limits is not thrown away.
    assert not llm._quota_notice("El límite de la jornada laboral. " * 40)


def test_backup_providers_take_their_own_models() -> None:
    """A chain is only useful if the backups answer with models that exist there."""
    saved = (llm.CHAIN, llm.PRO, llm.FLASH, llm.JUDGES)
    spec = dict(llm.PROVIDERS["hyper"])
    try:
        assert llm.parse_models("hyper:glm-5.2/kimi-k3,basura:x") == {
            "hyper": ("glm-5.2", "kimi-k3")}
        assert llm.parse_models("hyper:glm-5.2") == {"hyper": ("glm-5.2", "")}
        llm.configure(["claude", "hyper"], models=llm.parse_models("hyper:glm-5.2"))
        assert (llm.PRO, llm.FLASH) == ("claude-opus-5-5", "sonnet")     # head untouched
        assert llm._as("hyper", llm.PRO) == "glm-5.2"
        assert llm._as("hyper", llm.FLASH) == "deepseek-v4.1-flash"   # not named
    finally:
        llm.PROVIDERS["hyper"] = spec
        llm.CHAIN, llm.PRO, llm.FLASH, llm.JUDGES = saved


def test_chain_walks_to_the_next_provider_when_one_fails() -> None:
    """The whole point of a backup order is that link one dying is not fatal."""
    saved, tried = (llm.CHAIN, llm.PRO, llm.FLASH, llm.JUDGES), []
    original = llm._send

    def send(backend, model, *a, **kw):
        tried.append((backend, model))
        if backend == "claude":
            raise RuntimeError("claude caido")
        return "TEXTO"

    llm._send = send
    try:
        llm.configure(["claude", "hyper"])
        assert llm.chat(llm.FLASH, "hola") == "TEXTO"
        assert tried == [("claude", "sonnet"), ("hyper", "deepseek-v4.1-flash")], tried
        llm.configure(["claude"])          # no backup left: the error surfaces
        try:
            llm.chat(llm.FLASH, "hola")
            raise AssertionError("tendría que haber fallado")
        except RuntimeError as e:
            assert "claude caido" in str(e), e
    finally:
        llm._send = original
        llm.CHAIN, llm.PRO, llm.FLASH, llm.JUDGES = saved


def test_every_link_runs_on_book_writers_shared_ai_service() -> None:
    """One AI suite: hyper streams through book writer's hyper config with this
    project's key, asks for the model's whole output allowance (no ceiling), and
    only the last link of an unattended chain waits out a usage limit."""
    import os
    from unittest.mock import patch
    saved = (llm.CHAIN, llm.PRO, llm.FLASH, llm.JUDGES, llm.INTERACTIVE)
    try:
        llm.INTERACTIVE = False
        with patch.dict(os.environ, {"AW_API_KEY": "hyper-key", "OPENCODE_API_KEY": "zen-key"}),                 _shared({"hyper": [RuntimeError("503")], "opencode-zen": ["desde zen"]}) as seen:
            llm.configure(["hyper", "zen"], "qwen3.8-max", "deepseek-v4-pro-0813")
            assert llm.chat(llm.FLASH, "p", "sé exacto", max_tokens=900) == "desde zen"
        (hyper, over, calls), (zen, _, zen_calls) = seen
        assert hyper == "hyper" and zen == "opencode-zen"
        assert (over["base_url"], over["api_key"], over["stream"]) == (
            llm.PROVIDERS["hyper"]["base_url"], "hyper-key", True)
        assert "cap_is_ceiling" not in over
        kw = calls[0][1]
        assert (kw["model"], kw["system"], kw["max_completion_tokens"]) == (
            "deepseek-v4-pro-0813", "sé exacto", 900)
        assert kw["wait_for_limits"] is False               # a backup is still there
        assert zen_calls[0][1]["model"] == "deepseek-v4.1-flash"  # role translated
        assert zen_calls[0][1]["wait_for_limits"] is True     # last link: wait it out
        source = pathlib.Path(llm.__file__).read_text(encoding="utf-8")
        for needle in ("from openai import", "OpenAI(", '"-p"', '"run", "-m"'):
            assert needle not in source, needle
    finally:
        llm.CHAIN, llm.PRO, llm.FLASH, llm.JUDGES, llm.INTERACTIVE = saved


def test_markdown_to_html_demotes_headings_and_escapes() -> None:
    md = "## Sección\n\nUn párrafo con *énfasis* y **fuerza**.\n\nDos < tres & cuatro."
    got = publish.markdown_to_html(md)
    assert "<h3>Sección</h3>" in got, got          # article H2 -> page H3
    assert "<em>énfasis</em>" in got and "<strong>fuerza</strong>" in got, got
    assert "&lt; tres &amp; cuatro" in got, got    # raw HTML never reaches the post
    assert got.count("<p>") == 2, got


def test_split_title_pulls_the_h1() -> None:
    title, body = publish.split_title("# El valor y su sombra\n\nPor Sr. Sombra.\n")
    assert title == "El valor y su sombra" and body.startswith("Por Sr. Sombra")
    # A document with no H1 keeps all of its text.
    assert publish.split_title("Sin título aquí") == ("", "Sin título aquí")


def test_publish_is_a_noop_without_credentials() -> None:
    """No key, no request: an unconfigured run must not raise, and must not post."""
    saved = {k: os.environ.pop(k, None)
             for k in ("AW_WP_URL", "AW_WP_USER", "AW_WP_APP_PASSWORD",
                       "DEVTO_API_KEY", "AW_PUBLISH_TARGET")}
    try:
        assert not publish.configured()
        assert publish.publish("# X\n\ncuerpo", log=lambda *_: None) is None
    finally:
        os.environ.update({k: v for k, v in saved.items() if v is not None})


def test_publish_rejects_an_invalid_status() -> None:
    """Status is checked before the network call; WordPress would coerce it silently."""
    try:
        publish.publish("# X\n\ncuerpo", status="publicado")
    except ValueError:
        return
    raise AssertionError("un estado inválido tiene que fallar antes de postear")


def test_make_cover_degrades_without_network() -> None:
    """A dead image service must not take the publish step down with it."""
    saved = publish._COVER_URL
    publish._COVER_URL = "http://127.0.0.1:1/"
    try:
        assert publish.make_cover("Naves generacionales",
                                  timeout=5, log=lambda *_: None) is None
        assert "naves" in publish._cover_prompt("Naves generacionales").lower()
    finally:
        publish._COVER_URL = saved


def test_devto_publish_sends_key_markdown_and_cover(monkeypatch=None) -> None:
    """dev.to gets the API key header and markdown with front matter metadata:
    the API ignores the cover_image JSON field, so it must ride in the body."""
    captured = {}

    class FakeResponse(dict):
        status_code = 201

        def json(self):
            return dict(self)

        def raise_for_status(self):
            pass

    def fake_post(url, json=None, headers=None, timeout=90):
        captured.update(url=url, body=json["article"], key=headers["api-key"])
        return FakeResponse({"id": 7, "url": "https://dev.to/x/art", "status_code": 201})

    saved = {k: os.environ.pop(k, None)
             for k in ("DEVTO_API_KEY", "AW_PUBLISH_TARGET", "AW_COVER", "AW_TAGS")}
    os.environ["DEVTO_API_KEY"] = "k-test"
    real_post = publish.requests.post
    try:
        publish.requests.post = fake_post
        post = publish.publish("# Una tapa\n\ncuerpo", log=lambda *_: None)
        assert post["url"] == "https://dev.to/x/art"
        assert captured["key"] == "k-test"
        assert "/api/articles" in captured["url"]
        md = captured["body"]["body_markdown"]
        assert md.startswith('---\ntitle: "Una tapa"\n')
        assert md.endswith("\n---\n\ncuerpo")      # front matter is closed (422 otherwise)        assert "published: false" in md          # draft default
        assert 'cover_image: "https://image.pollinations.ai/' in md
        assert publish.target() == "devto"                     # key alone selects it
        # a colon in the title is a YAML mapping error unless it is quoted,
        # and dev.to only accepts ASCII alphanumeric tags
        os.environ["AW_TAGS"] = "ciencia ficción,sociología"
        publish.publish("# La nave: un ensayo «largo»\n\ncuerpo", log=lambda *_: None)
        front = captured["body"]["body_markdown"].split("\n---", 1)[0]
        assert 'title: "La nave: un ensayo «largo»"' in front, front
        assert "tags: cienciaficcion,sociologia" in front, front
    finally:
        publish.requests.post = real_post
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_publish_policy_only_goes_live_when_approved_and_clean() -> None:
    import main
    run = pipeline.Run()
    run.state = {"verdict": {"publicable": True}, "detector": {"final_score": 20.0, "passed": True}}
    assert main.wp_status(run, "no", 35.0) is None
    assert main.wp_status(run, "borrador", 35.0) == "draft"
    assert main.wp_status(run, "vivo", 35.0) == "publish"
    assert main.wp_status(run, "auto", 35.0) == "publish"
    run.state["detector"] = {"final_score": 80.0}          # detectors flagged it
    assert main.wp_status(run, "auto", 35.0) == "draft"
    run.state = {"verdict": {"publicable": False}, "detector": {"final_score": 5.0}}
    assert main.wp_status(run, "auto", 35.0) == "draft"     # PRO said no
    # a model that answers the string "false" has not approved anything
    run.state = {"verdict": {"publicable": "false"},
                 "detector": {"final_score": 5.0, "passed": True}}
    assert main.wp_status(run, "auto", 35.0) == "draft"


def test_final_approval_reads_publicable_as_a_real_boolean() -> None:
    real = pipeline.llm.chat_json
    run = pipeline.Run(log=lambda *_: None)
    try:
        for said, expected in (("false", False), ("true", True), (True, True), ("no", False)):
            pipeline.llm.chat_json = lambda *a, said=said, **k: {"publicable": said}
            got = pipeline.final_approval(run, "texto", {"titulo": "t"}, {})
            assert got["publicable"] is expected, (said, got)
    finally:
        pipeline.llm.chat_json = real


def test_a_cover_failure_after_posting_does_not_raise() -> None:
    """The post exists by then: raising would skip the receipt and repost on resume."""
    real_post, real_cover = publish.requests.post, publish.make_cover
    publish.make_cover = lambda *a, **k: b"x" * 6000

    def dead(*a, **k):
        raise publish.requests.ConnectionError("media endpoint down")

    publish.requests.post = dead
    try:
        publish._attach_cover({"id": 1}, "t", site="https://x", auth=("u", "p"),
                              timeout=5, log=lambda *_: None)
    finally:
        publish.requests.post, publish.make_cover = real_post, real_cover


def test_markdown_blockquote_becomes_html_blockquote() -> None:
    html_out = publish.markdown_to_html("> una cita larga\n> que sigue")
    assert html_out.startswith("<blockquote><p>una cita larga"), html_out
    assert "&gt;" not in html_out


def test_already_written_lists_past_topics_newest_first() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        for name, titulo in (("20260101-a", "Tema viejo"), ("20260820-b", "Tema nuevo")):
            d = root / name
            d.mkdir()
            (d / "01_topic.json").write_text(json.dumps({"titulo": titulo}), encoding="utf-8")
        (root / "20260501-roto").mkdir()
        (root / "20260501-roto" / "01_topic.json").write_text("{no json", encoding="utf-8")
        (root / "20260101-sin-tema").mkdir()          # crashed before picking a topic
        assert pipeline.already_written(root) == ["Tema nuevo", "Tema viejo"]
    assert pipeline.already_written(pathlib.Path(tmp) / "no-existe") == []


def test_interests_come_from_a_file_that_setup_rewrites() -> None:
    """A fresh clone has no interests.txt: the shipped example is the fallback, and
    `--setup` copies it and lets the user replace it."""
    saved = pipeline.INTERESTS_FILE, pipeline.INTERESTS_EXAMPLE
    with tempfile.TemporaryDirectory() as tmp:
        example = pathlib.Path(tmp) / "interests.example.txt"
        example.write_text("# comentario\nastronáutica, cohetes\n\ndinosaurios\n", encoding="utf-8")
        pipeline.INTERESTS_FILE = pathlib.Path(tmp) / "interests.txt"
        pipeline.INTERESTS_EXAMPLE = example
        try:
            assert pipeline.interest_areas() == ["astronáutica, cohetes", "dinosaurios"]
            answers = iter(["lingüística", "aviación: historia", ""])
            __import__("main").edit_interests(read=lambda *_: next(answers))
            assert pipeline.interest_areas() == ["lingüística", "aviación: historia"]
            assert pipeline.INTERESTS_FILE.read_text(encoding="utf-8").startswith("# comentario")
            __import__("main").edit_interests(read=lambda *_: "")      # Enter keeps them
            assert pipeline.interest_areas() == ["lingüística", "aviación: historia"]
        finally:
            pipeline.INTERESTS_FILE, pipeline.INTERESTS_EXAMPLE = saved
    assert pipeline._area_query("filosofía de la mente: conciencia, qualia") == "filosofía de la mente"
    assert pipeline._area_query("astronáutica, astrofísica") == "astronáutica"
    assert pipeline.interest_areas(), "the shipped example must not be empty"


def test_windows_split_on_paragraph_boundaries() -> None:
    """Windows never cut or repeat a paragraph; only a short tail may be dropped."""
    paras = [f"p{i} " + "palabra " * 100 for i in range(10)]
    chunks = style.windows("\n\n".join(paras), size=300)
    assert len(chunks) > 1
    got = [p for c in chunks for p in c.split("\n\n")]
    assert got == paras[:len(got)], got[:3]          # in order, nothing duplicated
    assert len(paras) - len(got) <= 3, len(got)      # at most a sub-window tail lost


def test_local_score_flags_flat_prose_with_llm_tells() -> None:
    if not style.FINGERPRINT.exists():
        print("  (saltado: falta style_fingerprint.json — corré --setup)")
        return
    robotic = " ".join(
        ["Es importante destacar que el capital opera de este modo.",
         "En primer lugar, el trabajo abstracto define la forma social.",
         "Por otro lado, la mercancia expresa un valor determinado.",
         "En conclusión, resulta crucial comprender este fenomeno complejo."] * 12)
    got = humanize.local_score(robotic)
    assert got["score"] > 30, got
    assert any("burstiness" in i or "cliché" in i for i in got["issues"]), got["issues"]


def test_local_score_flags_english_tells_and_wires_the_language_override() -> None:
    """--english swaps the tell lists and appends the language directive to prompts."""
    old = (pipeline.LANG, humanize.LANG)
    try:
        pipeline.LANG = humanize.LANG = "en"
        assert pipeline._lang(), "el modo inglés tiene que inyectar la directiva"
        robotic = " ".join(
            ["It is important to note that the landscape is ever-evolving.",
             "Let us delve into this rich tapestry of complexities."] * 10)
        got = humanize.local_score(robotic, lang="en")
        assert got["score"] > 30, got
        assert any("delve" in i or "important to note" in i for i in got["issues"]), \
            got["issues"]
    finally:
        pipeline.LANG, humanize.LANG = old


def test_english_style_layer_degrades_and_detects_language() -> None:
    """--english calibration: graceful without corpus_en/, real with it."""
    from build_corpus_en import _strip_gutenberg_boilerplate, is_english
    got = humanize.local_score(
        "The argument proceeds in three moves; each one fails on its own terms. "
        * 30, lang="en")
    assert set(got) >= {"score", "issues", "metrics"}
    assert isinstance(got["score"], float)
    assert not is_english("El capital opera de este modo en la sociedad moderna "
                          "y la mercancía expresa un valor determinado del trabajo.")
    en_text = ("The accumulation of capital presupposes the wage relation, and the "
               "wage relation presupposes property in the means of production, "
               "which is to say: class society. ")
    assert is_english(en_text * 12)
    raw = ("Project Gutenberg's header, lots of legalese here\n"
           "*** START OF THE PROJECT GUTENBERG EBOOK A BOOK ***\n"
           "Chapter one. The actual text begins here and says something worth keeping.\n"
           "*** END OF THE PROJECT GUTENBERG EBOOK ***\n"
           "Subscription footer follows.")
    assert "legalese" not in _strip_gutenberg_boilerplate(raw)
    assert "actual text begins" in _strip_gutenberg_boilerplate(raw)


def test_local_score_does_not_flag_the_author_himself() -> None:
    """The gate regressed once by scoring 1.500-word drafts against the 194k aggregate."""
    arts = style.articles()
    if not arts or not style.FINGERPRINT.exists():
        print("  (saltado: falta corpus/ o style_fingerprint.json)")
        return
    scores = [humanize.local_score(w)["score"]
              for _, body in arts[:6] for w in style.windows(body, 1200)[:2]]
    assert scores, "el corpus no produjo ventanas medibles"
    assert min(scores) < 40, sorted(scores)[:5]


def test_local_score_penalises_explanatory_colons() -> None:
    """Explanatory colons (setup:elaboration) are the strongest measured AI tell.

    Corpus averages 2-3 per 1k words; generated text hits 12+. The penalty must
    fire on colon-heavy prose and stay silent on corpus-level density.
    """
    if not style.FINGERPRINT.exists():
        print("  (saltado: falta style_fingerprint.json — corré --setup)")
        return
    colon_heavy = "\n\n".join(
        f"El concepto de {w} es central en la teoría crítica: define la estructura "
        f"social que organiza la producción material: determina las relaciones de poder "
        f"que atraviesan cada institución: configura el horizonte de lo posible: limita "
        f"la acción política a lo que el sistema tolera: reproduce la dominación bajo "
        f"la forma de libertad aparente."
        for w in ["valor", "trabajo", "capital", "mercancía", "dinero", "plusvalía",
                  "abstracción", "fetichismo", "alienación", "dominación", "mediación",
                  "forma", "sujeto", "lógica", "totalidad", "crítica"])
    got = humanize.local_score(colon_heavy)
    assert any("dos puntos" in i for i in got["issues"]), \
        f"colon penalty did not fire: {got['issues']}"
    assert got["score"] > 15, f"score too low for colon-heavy text: {got['score']}"

    # Corpus-level density must NOT trigger the penalty.
    mild = ("Marx analiza el capital como forma social. El trabajo abstracto media "
            "las relaciones de producción. La mercancía expresa valor de cambio. "
            "El fetichismo oculta el origen social del valor. La plusvalía surge "
            "del trabajo no pagado. El dinero funciona como equivalente general. "
            "La acumulación reproduce la desigualdad. El ciclo del capital exige "
            "consumo y producción simultáneos. La crisis es estructural, no accidental. "
            "La tasa de ganancia tiende a decrecer. La composición orgánica aumenta. "
            "El ejército industrial de reserva presiona los salarios. "
            "La concentración del capital acelera la centralización. "
            "El monopolio reemplaza la competencia libre. "
            "La financiarización separa la ganancia de la producción.")
    got_mild = humanize.local_score(mild)
    colon_issues = [i for i in got_mild["issues"] if "dos puntos" in i]
    assert not colon_issues, \
        f"false positive on low-colon text: {colon_issues}"


def test_rewrite_keeps_headings_and_covers_every_block() -> None:
    """Headings must survive verbatim; every prose block must reach the model once."""
    text = "## Uno\n\n" + "palabra " * 300 + "\n\n## Dos\n\n" + "otra " * 300
    prompts, original = [], llm.chat

    def fake(model, prompt, *a, **kw):
        prompts.append(prompt)
        return "reescrito " * 300   # a real rewrite keeps the block's length

    llm.chat = fake
    try:
        out = humanize._rewrite(text, ["ritmo plano"], "estilo",
                                log=lambda *_: None, batch_words=200)
    finally:
        llm.chat = original
    assert len(prompts) == 2, len(prompts)
    assert out.count("## Uno") == 1 and out.count("## Dos") == 1, out[:200]
    assert "reescrito" in out and "palabra palabra palabra" not in out, out[:200]


def test_rewrite_keeps_the_original_when_the_model_returns_a_stump() -> None:
    """A rewrite that lost a fifth of its words dropped content, not adjectives."""
    text = "palabra " * 100
    original, log_lines = llm.chat, []

    def short(model, prompt, *a, **kw):
        return "reescrito"

    llm.chat = short
    try:
        out = humanize._rewrite(text, [], "estilo",
                                log=lambda m: log_lines.append(m))
    finally:
        llm.chat = original
    assert out.strip() == text.strip()
    assert any("recortado" in l for l in log_lines)


def test_rewrite_keeps_the_original_block_when_the_model_fails() -> None:
    """Losing a paragraph to a transient API error would be worse than not rewriting."""
    text = "parrafo original que tiene que sobrevivir " * 30
    original = llm.chat

    def boom(*a, **kw):
        raise RuntimeError("upstream caido")

    llm.chat = boom
    try:
        out = humanize._rewrite(text, [], "estilo", log=lambda *_: None)
    finally:
        llm.chat = original
    assert out.strip() == text.strip()


def test_judge_windows_cover_the_tail_not_just_the_opening() -> None:
    """glm read the first 24k chars of a long draft as human while its middle
    sections scored 82%: long texts must be judged on more than the opening."""
    assert humanize._judge_windows("corto") == ["corto"]
    text = "x" * (humanize.JUDGE_WINDOW * 4)
    windows = humanize._judge_windows(text)
    assert len(windows) == 3
    assert windows[0] == text[:humanize.JUDGE_WINDOW]
    assert windows[-1] == text[-humanize.JUDGE_WINDOW:]


def test_judges_never_end_up_empty() -> None:
    """Whatever the provider switch does, the gate always has its two judges."""
    saved = (llm.CHAIN, llm.PRO, llm.FLASH, llm.JUDGES)
    try:
        llm.configure(["zen"])
        assert llm.JUDGES == llm.JUDGE_MODELS and len(llm.JUDGES) == 2
    finally:
        llm.CHAIN, llm.PRO, llm.FLASH, llm.JUDGES = saved


def test_dead_chain_asks_interactively_to_hold_or_change() -> None:
    """An attended run gets to wait or repoint instead of crashing mid-article."""
    import builtins
    saved = (llm.CHAIN, llm.PRO, llm.FLASH, llm.JUDGES, llm.INTERACTIVE)
    original = llm._send
    try:
        seen: list[str] = []

        def send(backend, model, *a, **kw):
            if backend == "claude":
                raise RuntimeError("claude caido")
            seen.append(model)
            return "TEXTO"
        llm._send = send
        llm.configure(["claude"])
        llm.INTERACTIVE = True
        answers = iter(["c", "2", "", ""])          # cambiar → hyper, defaults
        __import__("main").SHARED_MENU = False     # plain menu reads builtins.input
        real_input = builtins.input
        builtins.input = lambda *_: next(answers)
        try:
            assert llm.chat(llm.PRO, "hola") == "TEXTO"
            assert llm.CHAIN[0] == "hyper"
            # The retry chased the new role: it must not resend the old model
            # name (opus) to a provider that has never heard of it.
            assert seen == [llm.PRO], seen
        finally:
            builtins.input = real_input
        # aborting surfaces as KeyboardInterrupt, not a silent swallow
        llm.configure(["claude"])
        llm.INTERACTIVE = True
        builtins.input = lambda *_: "a"
        try:
            llm.chat(llm.PRO, "hola")
            raise AssertionError("abortar tenía que cortar el run")
        except KeyboardInterrupt:
            pass
        finally:
            builtins.input = real_input
    finally:
        llm.INTERACTIVE = False
        llm._send = original
        llm.CHAIN, llm.PRO, llm.FLASH = saved[0], saved[1], saved[2]
        llm.JUDGES, llm.INTERACTIVE = saved[3], saved[4]


def test_revise_discards_only_the_patch_that_drops_citations() -> None:
    """«This citation is decorative» must never be answered by deleting it —
    and one bad patch must not take the whole revision down with it."""
    srcs = research.assign_keys([
        research.Source(title="A", authors=["Moishe Postone"], year="2006"),
        research.Source(title="B", authors=["Robert Kurz"], year="2016")])
    text = ("El valor es histórico (Postone, 2006) y también (Kurz, 2016).\n\n"
            "Un estudio reciente lo confirma.")
    report = {"problemas": [{"gravedad": "alta", "problema": "x", "correccion": "y"}],
              "citas_validas": ["Kurz, 2016", "Postone, 2006"], "citas_sin_respaldo": []}
    run = pipeline.Run(log=lambda *_: None)
    original = llm.chat_json
    llm.chat_json = lambda *a, **kw: {"parches": [
        # se lleva a Kurz puesto: se descarta
        {"buscar": "El valor es histórico (Postone, 2006) y también (Kurz, 2016).",
         "reemplazar": "El valor es histórico (Postone, 2006)."},
        # inventa una fuente: se descarta
        {"buscar": "Un estudio reciente lo confirma.",
         "reemplazar": "Lo confirma (Bourdieu, 1979)."},
        # legítimo, y con el salto de línea retipeado: entra igual
        {"buscar": "Un estudio    reciente lo confirma.",
         "reemplazar": "Lo confirma Kurz (2016)."}]}
    try:
        out = pipeline.revise(run, text, report, srcs)
        assert "(Kurz, 2016)" in out and "(Postone, 2006)" in out, out
        assert "Bourdieu" not in out, out
        assert out.endswith("Lo confirma Kurz (2016)."), out
    finally:
        llm.chat_json = original


def test_draft_fetches_fulltext_for_cited_sources_that_arrived_abstract_only() -> None:
    """The blanket enrichment pass stops at budget*2 candidates in gather order,
    so sources the outline cites usually reached the draft as abstracts only.
    With the outline's per-section `fuentes` in hand, draft() gives each one a
    last targeted try — and a cached resume pays nothing."""
    sec = {"n": 1, "titulo": "Uno", "tesis": "t", "contenido": "c",
           "fuentes": ["Postone, 2006"], "palabras": 400}
    outline = {"titulo_final": "T", "secciones": [sec]}
    srcs = research.assign_keys([research.Source(title="A", authors=["Moishe Postone"],
                                                 year="2006", doi="10.1/abc")])
    assert srcs[0].key == "Postone, 2006"
    seen: list[set | None] = []
    saved = (llm.chat, style.style_block, research.enrich_fulltext)
    llm.chat = lambda m, p, *a, **kw: "Cuerpo escrito y breve. " * 100
    style.style_block = lambda *a, **kw: "ESTILO"

    def fake_enrich(sources, *a, keys=None, **kw):
        seen.append(keys)
        return 0
    research.enrich_fulltext = fake_enrich
    try:
        with tempfile.TemporaryDirectory() as d:
            run = pipeline.Run(dir=pathlib.Path(d), log=lambda *_: None)
            pipeline.draft(run, {"titulo": "T"}, outline, srcs)
            assert seen == [{"Postone, 2006"}], seen
            calls: list = []
            research.enrich_fulltext = lambda *a, **kw: calls.append(1) or 0
            pipeline.draft(run, {"titulo": "T"}, outline, srcs)
            assert calls == [], calls
    finally:
        llm.chat, style.style_block, research.enrich_fulltext = saved


def test_a_fully_cached_draft_costs_nothing_to_resume() -> None:
    """`--resume` re-read every section from disk and still paid one FLASH call
    per section for a rolling synopsis nobody was going to read."""
    sec = {"n": 1, "titulo": "Uno", "tesis": "t", "contenido": "c",
           "fuentes": [], "palabras": 400}
    outline = {"titulo_final": "T", "secciones": [sec, {**sec, "n": 2, "titulo": "Dos"}]}
    calls: list[str] = []
    saved = (llm.chat, style.style_block)
    llm.chat = lambda m, p, *a, **kw: calls.append(p) or "Cuerpo escrito y breve. " * 100
    style.style_block = lambda *a, **kw: "ESTILO"
    try:
        with tempfile.TemporaryDirectory() as d:
            run = pipeline.Run(dir=pathlib.Path(d), log=lambda *_: None)
            pipeline.draft(run, {"titulo": "T"}, outline, [])
            # two sections written + one synopsis for the second (never for the last)
            assert len(calls) == 3, calls
            calls.clear()
            pipeline.draft(run, {"titulo": "T"}, outline, [])
            assert calls == [], calls
            # one section gone: it is rewritten, and section 1 pays for its summary
            (pathlib.Path(d) / "04_sec02.md").unlink()
            calls.clear()
            pipeline.draft(run, {"titulo": "T"}, outline, [])
            assert len(calls) == 2, calls
    finally:
        llm.chat, style.style_block = saved


def test_a_resumed_run_does_not_re_review_or_re_humanize() -> None:
    """The review is a PRO call over the whole draft and the detector stage is
    several more: a `--resume` used to pay for both again for the same verdict."""
    calls: list[str] = []
    saved = (pipeline.pick_topic, pipeline.do_research, pipeline.make_outline,
             pipeline.draft, pipeline.review, pipeline.revise,
             pipeline.final_approval, humanize.humanize)
    srcs = research.assign_keys([research.Source(title="A", authors=["Moishe Postone"],
                                                year="2006")])
    pipeline.pick_topic = lambda run: {"titulo": "T", "hipotesis": "h"}
    pipeline.do_research = lambda run, topic, **kw: (srcs, [])
    pipeline.make_outline = lambda run, t, s: {"titulo_final": "T", "secciones": []}
    pipeline.draft = lambda run, t, o, s: "Cuerpo (Postone, 2006)."
    pipeline.review = lambda run, *a: (calls.append("review"),
                                       {"veredicto": "rechazado", "puntaje": 20,
                                        "problemas": [{"gravedad": "alta"}],
                                        "citas_validas": ["Postone, 2006"],
                                        "citas_sin_respaldo": []})[1]
    pipeline.revise = lambda run, text, *a: calls.append("revise") or text
    pipeline.final_approval = lambda run, *a: {"publicable": True, "puntaje": 90}
    humanize.humanize = lambda text, **kw: (calls.append("humanize")
                                            or (text, {"final_score": 20, "passed": True,
                                                       "text_hash": humanize.text_hash(text),
                                                       "lang": kw.get("lang", "es")}))
    try:
        with tempfile.TemporaryDirectory() as d:
            run = pipeline.Run(dir=pathlib.Path(d), mode="auto", log=lambda *_: None)
            pipeline.run_pipeline(run, rounds=2)
            assert calls == ["review", "revise", "review", "humanize"], calls
            calls.clear()
            pipeline.run_pipeline(run, rounds=2)
            assert calls == [], calls
            # drop the second round's report: only that round is paid again
            (pathlib.Path(d) / "06_review_r2.json").unlink()
            calls.clear()
            pipeline.run_pipeline(run, rounds=2)
            assert calls == ["review"], calls
    finally:
        (pipeline.pick_topic, pipeline.do_research, pipeline.make_outline,
         pipeline.draft, pipeline.review, pipeline.revise,
         pipeline.final_approval, humanize.humanize) = saved


def test_a_rejected_article_can_be_fixed_by_hand_and_re_approved() -> None:
    """PRO rejects over things that take minutes to fix (a "[dato a verificar]" left
    in the body); the only options used to be publishing anyway or losing the run."""
    srcs = research.assign_keys([research.Source(title="A", authors=["Moishe Postone"],
                                                year="2006")])
    verdicts = [{"publicable": False, "puntaje": 68, "dictamen": "queda un marcador",
                 "ajustes_menores": ["sacar [dato a verificar]"]},
                {"publicable": True, "puntaje": 88, "dictamen": "ok"}]
    seen: list[str] = []
    saved = (pipeline.pick_topic, pipeline.do_research, pipeline.make_outline,
             pipeline.draft, pipeline.review, pipeline.final_approval,
             humanize.humanize, pipeline.Run.ask)
    pipeline.pick_topic = lambda run: {"titulo": "T", "hipotesis": "h"}
    pipeline.do_research = lambda run, topic, **kw: (srcs, [])
    pipeline.make_outline = lambda run, t, s: {"titulo_final": "T", "secciones": []}
    pipeline.draft = lambda run, t, o, s: "Cuerpo [dato a verificar] (Postone, 2006)."
    pipeline.review = lambda run, *a: {"veredicto": "aprobado", "puntaje": 80,
                                       "problemas": [], "citas_validas": [],
                                       "citas_sin_respaldo": []}
    pipeline.final_approval = lambda run, text, *a: (seen.append(text),
                                                     verdicts.pop(0))[1]
    humanize.humanize = lambda text, **kw: (text, {"final_score": 8.2})
    try:
        with tempfile.TemporaryDirectory() as d:
            def fake_ask(self, prompt: str, default: str = "") -> str:
                if prompt.startswith("Editá"):        # the user fixes the file in place
                    self.save("04_draft_corregido.md", "Cuerpo limpio (Postone, 2006).")
                    return ""
                return "e" if "No aprobado" in prompt else default
            pipeline.Run.ask = fake_ask
            run = pipeline.Run(dir=pathlib.Path(d), mode="asistido", log=lambda *_: None)
            pipeline.run_pipeline(run, rounds=1)
            assert "[dato a verificar]" in seen[0] and len(seen) == 2, seen
            assert "limpio" in seen[1], seen
            assert run.state["verdict"]["publicable"] is True, run.state["verdict"]
            assert "limpio" in (pathlib.Path(d) / "05_final.md").read_text("utf-8")
            # the edit is on disk: a resume starts from it, it is not written again
            verdicts.append({"publicable": True, "puntaje": 88, "dictamen": "ok"})
            seen.clear()
            pipeline.run_pipeline(run, rounds=1)
            assert seen == ["Cuerpo limpio (Postone, 2006)."], seen
    finally:
        (pipeline.pick_topic, pipeline.do_research, pipeline.make_outline,
         pipeline.draft, pipeline.review, pipeline.final_approval,
         humanize.humanize, pipeline.Run.ask) = saved


def test_an_outline_audit_that_is_not_a_dict_keeps_the_outline() -> None:
    """PRO answered the outline audit with a bare JSON list; the run died on
    `critique.get` and an article with hours of research behind it was lost."""
    outline = {"titulo_final": "T", "resumen": "", "palabras_clave": [],
               "secciones": [{"n": 1, "titulo": "Uno", "tesis": "t", "contenido": "c",
                              "fuentes": [], "palabras": 800, "subsecciones": []}]}
    answers = [outline, ["la sección dos repite la uno"]]
    saved = llm.chat_json
    llm.chat_json = lambda *a, **kw: answers.pop(0)
    try:
        with tempfile.TemporaryDirectory() as d:
            run = pipeline.Run(dir=pathlib.Path(d), log=lambda *_: None)
            got = pipeline.make_outline(run, {"titulo": "T"}, [])
            assert answers == [], answers
            assert got["titulo_final"] == "T" and len(got["secciones"]) == 1, got
    finally:
        llm.chat_json = saved


def test_every_prompt_renders_with_what_the_pipeline_passes() -> None:
    """A placeholder added to a prompt but not to its .format() call is a KeyError
    three hours into a run. Render the real calls instead of trusting the diff."""
    srcs = research.assign_keys([research.Source(title="A", authors=["Moishe Postone"],
                                                year="2006")])
    sec = {"n": 1, "titulo": "Uno", "tesis": "t", "contenido": "c",
           "fuentes": ["Postone, 2006"], "palabras": 400}
    outline = {"titulo_final": "T", "secciones": [sec, {**sec, "n": 2, "titulo": "Dos"}]}
    seen: list[str] = []
    saved = (llm.chat, llm.chat_json, style.style_block)
    llm.chat = lambda m, p, *a, **kw: seen.append(p) or "Cuerpo (Postone, 2006). Corta. " * 100
    llm.chat_json = lambda m, p, *a, **kw: seen.append(p) or {"veredicto": "aprobado"}
    style.style_block = lambda *a, **kw: "ESTILO"
    try:
        with tempfile.TemporaryDirectory() as d:
            run = pipeline.Run(dir=pathlib.Path(d), log=lambda *_: None)
            pipeline.draft(run, {"titulo": "T"}, outline, srcs)
            pipeline.review(run, "Cuerpo (Postone, 2006).", {"hipotesis": "h"}, srcs)
    finally:
        llm.chat, llm.chat_json, style.style_block = saved
    joined = "\n".join(seen)
    for marker in ("REGISTRO DEL FORMATO", "PROHIBIDO volver a escribirlas",
                   "REPETICIONES LITERALES", "REGISTRO EXIGIDO POR EL FORMATO"):
        assert marker in joined, marker


def test_repeated_phrases_catches_a_formula_served_twice() -> None:
    """The same sentence three sections later is what reads as machine-written."""
    filler = "una oracion cualquiera con palabras distintas cada vez que aparece aqui. "
    formula = "No hay huelga contra una probabilidad calculada por nadie. "
    text = filler + formula + filler * 3 + formula + filler
    hits = humanize.repeated_phrases(text)
    assert any("huelga contra una probabilidad" in h for h in hits), hits
    assert len(hits) <= 2, hits          # one entry per repetition, not one per window
    assert not humanize.repeated_phrases(filler + "otra cosa completamente distinta aca.")
    # and the score has to notice, or the rewriter never hears about it
    assert any("repetidas" in i for i in humanize.local_score(text)["issues"])


def test_zen_is_a_first_class_openai_compatible_provider() -> None:
    import llm
    spec = llm.PROVIDERS["zen"]
    assert spec["base_url"].startswith("https://opencode.ai/zen/")
    assert spec["key_env"] == "OPENCODE_API_KEY"
    assert llm.ZEN_ALIAS["deepseek-v4-pro-0813"] == "deepseek-v4-pro"
    saved = (llm.CHAIN, llm.PRO, llm.FLASH, llm.JUDGES)
    try:
        llm.configure(["hyper", "zen"])
        assert llm._as("zen", llm.FLASH) == "deepseek-v4.1-flash"
        assert llm._as("zen", llm.PRO) == llm.PROVIDERS["zen"]["pro"]
    finally:
        llm.CHAIN, llm.PRO, llm.FLASH, llm.JUDGES = saved
    assert "opus" in llm.CLAUDE_ALIAS.values() and "sonnet" in llm.CLAUDE_ALIAS.values()


def test_grok_is_a_first_class_openai_compatible_provider() -> None:
    import llm
    spec = llm.PROVIDERS["grok"]
    assert spec["base_url"].startswith("https://api.x.ai/")
    assert spec["key_env"] == "XAI_API_KEY"
    saved = (llm.CHAIN, llm.PRO, llm.FLASH, llm.JUDGES)
    try:
        llm.configure(["hyper", "grok"])
        assert llm._as("grok", llm.FLASH) == "grok-4-fast"
        assert llm._as("grok", llm.PRO) == llm.PROVIDERS["grok"]["pro"]
    finally:
        llm.CHAIN, llm.PRO, llm.FLASH, llm.JUDGES = saved


def test_g4f_is_a_keyless_local_gpt4free_link() -> None:
    """gpt4free's `g4f api` server: ai-suite's gpt4free provider, no key by default.
    g4f hands any other bearer to its backends as their key, so hyper's AW_API_KEY
    must never ride along; G4F_API_KEY (the server's own) is sent when set."""
    spec = llm.PROVIDERS["g4f"]
    assert spec["base_url"] == "http://127.0.0.1:1337/v1"
    assert llm.SHARED["g4f"] == "gpt4free"
    assert spec["pro"] in spec["models"] and spec["flash"] in spec["models"]

    class Resp:
        def json(self):
            return {"data": [{"id": "deepseek-v4-pro"}, {"id": "flux", "image": True},
                             {"id": "PollinationsAI", "provider": True}]}
    import requests
    saved_get = requests.get
    requests.get = lambda *a, **kw: Resp()
    try:
        assert llm._live_models("g4f") == ["deepseek-v4-pro"]
    finally:
        requests.get = saved_get

    seen = []

    class Service:
        def generate_content(self, *a, **kw):
            return "texto"
    saved_shared = llm.shared_service
    saved_env = {k: os.environ.pop(k, None) for k in ("G4F_API_KEY", "AW_API_KEY")}
    llm.shared_service = lambda provider, overrides: (seen.append((provider, overrides)), Service())[1]
    try:
        os.environ["AW_API_KEY"] = "hyper-key"
        assert llm._send("g4f", "glm-5.3", "hola", None, temperature=None,
                         max_tokens=None, retries=0) == "texto"
        os.environ["G4F_API_KEY"] = "server-key"
        llm._send("g4f", "glm-5.3", "hola", None, temperature=None, max_tokens=None, retries=0)
    finally:
        llm.shared_service = saved_shared
        for k, v in saved_env.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
    assert seen[0][0] == "gpt4free"
    assert seen[0][1]["base_url"] == "http://127.0.0.1:1337/v1"
    assert "api_key" not in seen[0][1], seen[0][1]
    assert seen[1][1]["api_key"] == "server-key"


def test_go_provider_is_pinned_and_legacy_name_still_works() -> None:
    """The CLI route is plain `go` in every menu and chain: no variant to pick,
    and old .env files that still say `opencode` keep their last link."""
    assert llm.OPENCODE_PROVIDER == "opencode-go"
    spec = llm.PROVIDERS["go"]
    assert "variants" not in spec
    assert (spec["pro"], spec["flash"]) == ("qwen3.8-flash", "deepseek-v4.1-flash")
    saved = (llm.CHAIN, llm.PRO, llm.FLASH, llm.JUDGES)
    try:
        llm.configure(["opencode"])            # el nombre viejo, desde un .env
        assert llm.CHAIN == ["go"]
    finally:
        llm.CHAIN, llm.PRO, llm.FLASH, llm.JUDGES = saved


def test_go_live_models_unsuffix_the_hyper_spelling() -> None:
    """The CLI prints opencode-go ids; the catalogue knows the hyper spelling."""
    saved_run, saved_which = llm.subprocess.run, llm.shutil.which
    llm.shutil.which = lambda name: "C:/fake/opencode.exe"
    llm.subprocess.run = lambda *a, **kw: type("R", (), {
        "stdout": "opencode-go/deepseek-v4-pro\nopencode-go/glm-5.2"})()
    try:
        got = llm._live_models("go")
        assert got == ["deepseek-v4-pro-0813", "glm-5.2"], got
    finally:
        llm.subprocess.run, llm.shutil.which = saved_run, saved_which


def test_judges_report_their_worst_window() -> None:
    """A clean opening must not launder a machine-sounding tail."""
    prompts, old = [], llm.chat_json
    llm.chat_json = lambda m, p, *a, **kw: (
        prompts.append(p),
        {"ai_probability": 91 if "COLA DELATORA" in p else 9,
         "veredicto": "ia" if "COLA DELATORA" in p else "humano"})[-1]
    body = "apertura limpia. " * 20000 + "COLA DELATORA. "
    try:
        got = humanize.llm_judges(body, models=["juez-uno"], log=lambda *_: None)
    finally:
        llm.chat_json = old
    assert len(prompts) == 3, len(prompts)      # inicio, medio y final
    assert got and got[0]["ai_probability"] == 91, got
    assert got[0]["model"] == "juez-uno"


def test_byline_is_a_pseudonym() -> None:
    assert pipeline.BYLINE and "Rehfeldt" not in pipeline.BYLINE, pipeline.BYLINE


def test_resolve_choice_takes_names_and_indexes() -> None:
    import main as cli
    fmts = list(pipeline.FORMATS)
    assert cli.resolve_choice("1", fmts) == fmts[0]
    assert cli.resolve_choice(str(len(fmts)), fmts) == fmts[-1]
    assert cli.resolve_choice("largo", fmts) == "largo"
    assert cli.resolve_choice("0", fmts) == ""
    assert cli.resolve_choice(str(len(fmts) + 1), fmts) == ""
    assert cli.resolve_choice("gigante", fmts) == ""


_OFFLINE = ("archive_org_book", "libgen_book", "fetch_paper", "web_fulltext")


def _no_network() -> dict:
    """Stub every route in ``fetch_book`` that would leave the machine.

    ``_work_info`` included: it is a model call, and left live it would both reach
    the network and decide the length floor differently on every run.
    """
    saved = {n: getattr(research, n) for n in (*_OFFLINE, "_work_info")}
    for name in _OFFLINE:
        setattr(research, name, lambda *a, **k: None)
    research._work_info = lambda a, t, y: {"titles": (), "book": False}
    return saved


def _restore(saved: dict) -> None:
    for name, fn in saved.items():
        setattr(research, name, fn)


def test_fetch_book_reads_the_library_before_the_network() -> None:
    """An interrupted run must not re-download what already landed in library/."""
    old, saved = research.LIBRARY, _no_network()
    with tempfile.TemporaryDirectory() as tmp:
        research.LIBRARY = pathlib.Path(tmp)
        (research.LIBRARY / "Paul Cockshott - Towards a New Socialism (1993).txt"
         ).write_text("planificación " * 1000, encoding="utf-8")
        try:
            got = research.fetch_book("Paul Cockshott, Towards a New Socialism (1993)",
                                      log=lambda *_: None)
        finally:
            research.LIBRARY = old
            _restore(saved)
    assert got is not None and len(got.fulltext) > 3000, got
    assert got.url.endswith(".txt"), got.url          # dedupe folds it into scan_library's


def _fake_calibre(root: pathlib.Path) -> None:
    """A minimal Calibre library: one book, one txt file, real schema shape."""
    import sqlite3
    (root / "Paul Cockshott" / "Towards a New Socialism (1)").mkdir(parents=True)
    (root / "Paul Cockshott" / "Towards a New Socialism (1)"
     / "Towards a New Socialism - Paul Cockshott.txt").write_text(
        "planificación " * 1000, encoding="utf-8")
    con = sqlite3.connect(root / "metadata.db")
    con.executescript("""
        create table books (id integer primary key, title text, path text, pubdate text);
        create table data (id integer primary key, book int, name text, format text);
        create table authors (id integer primary key, name text);
        create table books_authors_link (id integer primary key, book int, author int);
        create table publishers (id integer primary key, name text);
        create table books_publishers_link (id integer primary key, book int, publisher int);
        insert into books values (1, 'Towards a New Socialism',
                                  'Paul Cockshott/Towards a New Socialism (1)',
                                  '1993-01-01 00:00:00+00:00');
        insert into data values (1, 1, 'Towards a New Socialism - Paul Cockshott', 'TXT');
        insert into authors values (1, 'Paul Cockshott');
        insert into books_authors_link values (1, 1, 1);
        insert into publishers values (1, 'Spokesman Books');
        insert into books_publishers_link values (1, 1, 1);
    """)
    con.commit()
    con.close()


def test_calibre_answers_before_the_network() -> None:
    """A work the user already owns must come off disk, not off archive.org."""
    old_cal, old_lib = research.CALIBRE, research.LIBRARY
    with tempfile.TemporaryDirectory() as tmp:
        research.CALIBRE = pathlib.Path(tmp) / "Calibre Library"
        research.CALIBRE.mkdir()
        research.LIBRARY = pathlib.Path(tmp) / "library"
        _fake_calibre(research.CALIBRE)
        try:
            got = research.calibre_book("Paul Cockshott, Towards a New Socialism (1993)",
                                        log=lambda *_: None)
            cached = list(research.LIBRARY.glob("*.txt"))
            miss = research.calibre_book("Robert Kurz, Geld ohne Wert (2012)",
                                         log=lambda *_: None)
        finally:
            research.CALIBRE, research.LIBRARY = old_cal, old_lib
    assert got is not None and len(got.fulltext) > 3000, got
    assert got.year == "1993" and got.venue == "Spokesman Books", got
    assert got.authors == ["Paul Cockshott"], got.authors
    # The citation is APA, and the local path never reaches the reference list.
    assert got.citation().startswith("Cockshott, P. (1993)."), got.citation()
    assert "Calibre Library" not in got.citation(), got.citation()
    # Extracted once: the next run reads library/ instead of converting again.
    assert len(cached) == 1, cached
    assert miss is None, miss           # a book the user does not own is not invented


def test_calibre_missing_library_is_not_an_error() -> None:
    old = research.CALIBRE
    research.CALIBRE = pathlib.Path("Z:/no/such/calibre")
    try:
        assert research.calibre_rows() == []
        assert research.calibre_book("Alguien, Un libro (2001)", log=lambda *_: None) is None
    finally:
        research.CALIBRE = old


def test_web_fulltext_gates_on_the_downloaded_text() -> None:
    """The open-web fallback: PDFs first, junk hosts out, wrong document rejected.

    No browser and no network — ``browser_search`` and ``fetch_text`` are replaced.
    """
    pages = {
        "https://www.scribd.com/doc/1": "Boulding spaceship earth " * 900,
        "https://ejemplo.org/otra-cosa.pdf": "una cosa completamente distinta " * 900,
        "https://ejemplo.org/boulding.pdf":
            "The Economics of the Coming Spaceship Earth, by Kenneth Boulding. " * 900,
        "https://arxiv.org/abs/1234.5678": "resumen corto",
        "https://arxiv.org/pdf/1234.5678": "the economics of the coming spaceship earth " * 900,
    }
    asked: list[str] = []

    def fake_search(query, limit=8):
        asked.append(query)
        return [{"title": "Scribd", "url": "https://www.scribd.com/doc/1"},
                {"title": "otra", "url": "https://ejemplo.org/otra-cosa.pdf"},
                {"title": "Boulding PDF", "url": "https://ejemplo.org/boulding.pdf"}]

    old_search, old_fetch, old_lib = (research.browser_search, research.fetch_text,
                                      research.LIBRARY)
    old_title = research._titlepage
    research.browser_search = fake_search
    research.fetch_text = lambda url, *a, **k: pages.get(url, "")
    research._titlepage = lambda body, t, a, y: {"title": t, "authors": a,
                                                 "year": y, "venue": ""}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            research.LIBRARY = pathlib.Path(tmp)
            got = research.web_fulltext(
                "Boulding, Kenneth, «The Economics of the Coming Spaceship Earth» (1966)",
                log=lambda *_: None)
            cached = list(research.LIBRARY.glob("*.txt"))
        # An abs page is the abstract; the fallback has to ask for the PDF instead.
        research.browser_search = lambda q, limit=8: [
            {"title": "arXiv", "url": "https://arxiv.org/abs/1234.5678"}]
        with tempfile.TemporaryDirectory() as tmp:
            research.LIBRARY = pathlib.Path(tmp)
            arx = research.web_fulltext(
                "Boulding, Kenneth, «The Economics of the Coming Spaceship Earth» (1966)",
                log=lambda *_: None)
    finally:
        research.browser_search, research.fetch_text = old_search, old_fetch
        research._titlepage, research.LIBRARY = old_title, old_lib
    assert got is not None, "el PDF correcto tenía que pasar el filtro"
    assert got.url == "https://ejemplo.org/boulding.pdf", got.url   # ni Scribd ni la otra
    assert asked and asked[0].endswith(" pdf"), asked               # PDF primero
    assert len(cached) == 1, cached                                 # queda en library/
    assert arx is not None and arx.url == "https://arxiv.org/pdf/1234.5678", arx


def test_titlepage_only_accepts_what_the_document_says() -> None:
    """The title page corrects the reading list's guess; it may not invent past it."""
    body = ("Kenneth E. Boulding\nECONOMICS OF THE COMING\nSPACESHIP EARTH\n1966\n"
            "We are now in the middle of a long process of transition. " * 60)
    guess = ("Chrysalis: informe de diseño", ["Project Hyperion"], "2025")
    answers: list[dict] = []
    old = llm.chat_json
    llm.chat_json = lambda *a, **k: answers.pop(0)
    try:
        answers.append({"titulo": "Economics of the Coming Spaceship Earth",
                        "autores": ["Kenneth E. Boulding"], "anio": "1966",
                        "editorial": "Resources for the Future"})
        good = research._titlepage(body, *guess)
        # Everything hallucinated: a title, an author and a year the text never prints.
        answers.append({"titulo": "Tratado de los sistemas cerrados",
                        "autores": ["Ilya Prigogine"], "anio": "1979",
                        "editorial": "Alianza"})
        bad = research._titlepage(body, *guess)
        answers.append({"titulo": None, "autores": "no es una lista"})
        junk = research._titlepage(body, *guess)
        llm.chat_json = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("sin proveedor"))
        dead = research._titlepage(body, *guess)
    finally:
        llm.chat_json = old
    assert good["title"] == "Economics of the Coming Spaceship Earth", good
    assert good["authors"] == ["Kenneth E. Boulding"] and good["year"] == "1966", good
    assert good["venue"] == "", good      # la editorial no está en el texto: no se inventa
    # Nothing verifiable came back, so the guess stands untouched in all three cases.
    for out in (bad, junk, dead):
        assert (out["title"], out["authors"], out["year"]) == guess, out
        assert out["venue"] == "", out


def test_a_work_enters_the_dossier_whole() -> None:
    """No route may store a slice: the cut belongs to ``Source.brief`` per call.

    «Paradises Lost» closes *The Found and the Lost*, so any windowing of the
    stored text is one bad guess away from keeping the other novellas and not the
    work that was asked for. The only correct window is the whole file.
    """
    other = "Otra novela corta que abre el tomo. " * 12000      # ~430k
    body = other + "PARADISES LOST\n" + "El viaje generacional empieza. " * 400
    old, saved = research.LIBRARY, _no_network()
    with tempfile.TemporaryDirectory() as tmp:
        research.LIBRARY = pathlib.Path(tmp)
        (research.LIBRARY / "Ursula K Le Guin - The Found and the Lost (2016).txt"
         ).write_text(body, encoding="utf-8")
        try:
            got = research.fetch_book(
                "Ursula K Le Guin, The Found and the Lost (2016)", log=lambda *_: None)
        finally:
            research.LIBRARY = old
            _restore(saved)
    assert got is not None, "no leyó el archivo de library/"
    assert got.fulltext == body, f"guardó {len(got.fulltext)} de {len(body)} caracteres"
    assert "PARADISES LOST" in got.fulltext, "perdió la obra pedida"


def test_partial_flags_a_cut_download_and_a_sample() -> None:
    whole = "Una obra entera. " * 6000                          # 102k, cierra bien
    assert research._partial(whole) == ""
    assert "cortado" in research._partial(whole[:90000] + " y entonces la frase se")
    assert "extracto" in research._partial("Un capítulo suelto. " * 500)


def test_web_fulltext_prefers_the_whole_book_over_the_sample() -> None:
    """A publisher's sample and the whole book both clear the floor; length decides."""
    sample = "El colapso de la modernización, de Robert Kurz. " * 700      # ~33k
    whole = "El colapso de la modernización, de Robert Kurz. " * 6000      # ~282k
    pages = {"https://editorial.example/muestra.pdf": sample,
             "https://archivo.example/completo.pdf": whole}
    old_search, old_fetch, old_lib = (research.browser_search, research.fetch_text,
                                      research.LIBRARY)
    old_title = research._titlepage
    research.browser_search = lambda q, limit=8: [
        {"title": "muestra", "url": "https://editorial.example/muestra.pdf"},
        {"title": "completo", "url": "https://archivo.example/completo.pdf"}]
    research.fetch_text = lambda url, *a, **k: pages.get(url, "")
    research._titlepage = lambda body, t, a, y: {"title": t, "authors": a,
                                                 "year": y, "venue": ""}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            research.LIBRARY = pathlib.Path(tmp)
            got = research.web_fulltext(
                "Kurz, Robert, «El colapso de la modernización» (2016)",
                log=lambda *_: None)
            cached = [p.stat().st_size for p in research.LIBRARY.glob("*.txt")]
    finally:
        research.browser_search, research.fetch_text = old_search, old_fetch
        research._titlepage, research.LIBRARY = old_title, old_lib
    assert got is not None, got
    assert got.url.endswith("completo.pdf"), got.url        # no el primero que pasó
    assert len(got.fulltext) > 200000, len(got.fulltext)
    # And the whole thing is what goes to disk, not the sample.
    assert len(cached) == 1 and cached[0] > 200000, cached


def test_fetch_book_finds_the_original_language_edition() -> None:
    """The list asks in Spanish; the copy on disk is the German original."""
    old_lib, old_other = research.LIBRARY, research._work_info
    saved = _no_network()
    research._work_info = lambda a, t, y: {
        "titles": ("Der Kollaps der Modernisierung",), "book": True}
    with tempfile.TemporaryDirectory() as tmp:
        research.LIBRARY = pathlib.Path(tmp)
        (research.LIBRARY / "Robert Kurz - Der Kollaps der Modernisierung.txt"
         ).write_text("Zusammenbruch " * 6000, encoding="utf-8")
        # A different book by the same author must not stand in for it.
        (research.LIBRARY / "Robert Kurz - Geld ohne Wert (2012).txt"
         ).write_text("Wert " * 1000, encoding="utf-8")
        try:
            got = research.fetch_book(
                "Kurz, Robert, «El colapso de la modernización» (2016)",
                log=lambda *_: None)
            miss = research.fetch_book("Kurz, Robert, «Un libro inexistente» (2030)",
                                       log=lambda *_: None)
        finally:
            research.LIBRARY, research._work_info = old_lib, old_other
            _restore(saved)
    assert got is not None, "no encontró la edición en su idioma original"
    assert "Kollaps" in got.url, got.url
    # Cited as the edition actually read, not as the title the list asked for.
    assert got.title == "Der Kollaps der Modernisierung", got.title
    assert miss is None or "Geld" not in miss.url, miss   # ni el otro libro del autor


def test_fetch_book_keeps_looking_after_a_download_that_stops_mid_sentence() -> None:
    """A text cut in the middle of a phrase is held, never returned as the work."""
    old_lib, saved = research.LIBRARY, _no_network()
    research._work_info = lambda a, t, y: {"titles": (), "book": True}
    # 84k characters, over the 60.000 floor, ending mid-phrase: this is the shape
    # *Der Kollaps* arrived in off the open web, and it used to end the search.
    research.web_fulltext = lambda w, log=print: research.Source(
        title="Der Kollaps der Modernisierung", authors=["Robert Kurz"], year="1991",
        kind="book", venue="", url="https://ejemplo/completo.pdf",
        abstract="", fulltext="Zusammenbruch der Modernisierung. " * 4000,
        origin="web")
    with tempfile.TemporaryDirectory() as tmp:
        research.LIBRARY = pathlib.Path(tmp)
        (research.LIBRARY / "Robert Kurz - Der Kollaps der Modernisierung.txt"
         ).write_text("Zusammenbruch " * 6000, encoding="utf-8")   # 78k, cortado
        try:
            got = research.fetch_book(
                "Kurz, Robert, «Der Kollaps der Modernisierung» (1991)",
                log=lambda *_: None)
        finally:
            research.LIBRARY = old_lib
            _restore(saved)
    assert got is not None, "un extracto cortado no puede volverse una laguna"
    # The disk copy cleared the floor but was cut, so the chain ran on and the whole
    # text won. Before this it returned the extract and never asked the open web.
    assert got.origin == "web", got.origin
    assert not research._cut(got.fulltext), "devolvió un texto cortado a la mitad"


def test_missing_books_tries_every_work_on_the_list() -> None:
    """A gap has to mean the chain failed, not that the budget ran out first."""
    # Sixteen unrelated titles: shared words would fold them into one gap through
    # ``_covers`` before any of this is reached, and that dedupe is not what is tested.
    titulos = ["Non-Stop", "Orphans of the Sky", "Aurora boreal", "Realismo capitalista",
               "Der Kollaps", "Spaceship Earth", "Towards a New Socialism", "Chrysalis",
               "Time and Domination", "Las aventuras", "Paradises Lost", "Long Sun",
               "Minimal Crew", "Genetically Viable Population", "Anarres", "Ubik"]
    wanted = [f"Autor{i}, {t} (19{i:02d})" for i, t in enumerate(titulos)]
    asked: list[str] = []
    old_fetch = research.fetch_book
    research.fetch_book = lambda w, log=print: asked.append(w)
    try:
        gaps = research.missing_books([], wanted, log=lambda *_: None, workers=2)
    finally:
        research.fetch_book = old_fetch
    assert sorted(asked) == sorted(wanted), f"sin pedir: {set(wanted) - set(asked)}"
    assert len(gaps) == 16, len(gaps)
    # A gap is hand-download links and nothing else: scraping Anna's Archive for
    # "direct candidates" only ever cost three 45s timeouts and returned the links.
    assert all(set(g) == {"wanted", "fallbacks"} and g["fallbacks"] for g in gaps), gaps[0]


def test_work_info_refuses_a_variant_it_cannot_trust() -> None:
    """Only a real alternative title survives; a paraphrase or a blank does not."""
    old = llm.chat_json
    answers = [{"titulos": ["Der Kollaps der Modernisierung",
                            "El colapso de la modernizacion",   # el mismo, traducido
                            "The Collapse of Modernization"],    # tercero: se descarta
                "tipo": "libro"},
               {"titulos": [], "tipo": "breve"},
               "no es un dict"]
    llm.chat_json = lambda *a, **k: answers.pop(0)
    try:
        research._work_info.cache_clear()
        got = research._work_info("Robert Kurz", "El colapso de la modernización", "2016")
        empty = research._work_info("Alguien", "Un informe cualquiera", "2001")
        junk = research._work_info("Otro", "Otro título distinto", "2002")
    finally:
        llm.chat_json = old
        research._work_info.cache_clear()
    assert got["titles"] == ("Der Kollaps der Modernisierung",), got   # el eco, fuera
    assert got["book"] is True, got
    assert empty["titles"] == () and empty["book"] is False, empty
    # Sin respuesta utilizable no hay piso: un texto breve entero no se descarta.
    assert junk == {"titles": (), "book": False}, junk
    # A one-word title has nothing to translate against: no call, no variants.
    assert research._work_info("X", "Ur", "")["titles"] == ()


def test_fetch_book_refuses_a_chapter_of_a_book() -> None:
    """A book that only turns up as one chapter is a gap, not a source."""
    old_lib, old_info = research.LIBRARY, research._work_info
    saved = _no_network()
    chapter = research.Source(title="El colapso de la modernización",
                              authors=["Robert Kurz"], year="2016", kind="book",
                              fulltext="capítulo uno. " * 2000)          # ~26k
    research.web_fulltext = lambda *a, **k: chapter
    said: list[str] = []
    try:
        with tempfile.TemporaryDirectory() as tmp:
            research.LIBRARY = pathlib.Path(tmp)
            research._work_info = lambda a, t, y: {"titles": (), "book": True}
            book = research.fetch_book("Kurz, Robert, «El colapso de la modernización» (2016)",
                                       log=said.append)
            # The same text, when the work really is a short one, is kept.
            research._work_info = lambda a, t, y: {"titles": (), "book": False}
            brief = research.fetch_book("Boulding, Kenneth, «The Economics of the "
                                        "Coming Spaceship Earth» (1966)", log=lambda *_: None)
    finally:
        research.LIBRARY, research._work_info = old_lib, old_info
        _restore(saved)
    assert book is None, book
    assert any("descartado" in s for s in said), said
    assert brief is chapter, brief


def _mini_epub(path: pathlib.Path, xhtml: bytes) -> None:
    """The smallest thing ebooklib accepts as an epub: container, OPF, spine."""
    import zipfile
    opf = (b'<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf"'
           b' version="2.0" unique-identifier="id"><metadata'
           b' xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>T</dc:title>'
           b'<dc:language>es</dc:language></metadata><manifest>'
           b'<item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/>'
           b'<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>'
           b'</manifest><spine toc="ncx"><itemref idref="c1"/></spine></package>')
    ncx = (b'<?xml version="1.0"?><ncx xmlns="http://www.daisy.org/z3986/2005/ncx/">'
           b'<head><meta name="dtb:uid" content="x"/></head><docTitle><text>T'
           b'</text></docTitle><navMap/></ncx>')
    cont = (b'<?xml version="1.0"?><container version="1.0"'
            b' xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
            b'<rootfile full-path="OEBPS/content.opf"'
            b' media-type="application/oebps-package+xml"/></rootfiles></container>')
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr("META-INF/container.xml", cont)
        z.writestr("OEBPS/content.opf", opf)
        z.writestr("OEBPS/toc.ncx", ncx)
        z.writestr("OEBPS/c1.xhtml", xhtml)


def test_read_local_epub_parses_xhtml_as_xml() -> None:
    """XHTML chapters carry a real XML declaration; an HTML parser over them is
    the XMLParsedAsHTMLWarning itself, and entities must survive the extraction."""
    import warnings
    from bs4 import XMLParsedAsHTMLWarning
    xhtml = (b'<?xml version="1.0" encoding="utf-8"?>'
             b'<html xmlns="http://www.w3.org/1999/xhtml"><head><title>T</title>'
             b'</head><body><h1>Cap&#237;tulo uno</h1>'
             b'<p>planificaci&#243;n &amp; mercado<br/></p></body></html>')
    with tempfile.TemporaryDirectory() as tmp:
        p = pathlib.Path(tmp) / "min.epub"
        _mini_epub(p, xhtml)
        with warnings.catch_warnings():
            warnings.simplefilter("error", XMLParsedAsHTMLWarning)
            text = research.read_local(p)
    assert "Capítulo uno" in text and "planificación & mercado" in text, text


def test_read_local_epub_recovers_an_html_only_chapter() -> None:
    """A hand-edited chapter without the declaration falls back to the tolerant
    HTML parser instead of failing the whole book."""
    with tempfile.TemporaryDirectory() as tmp:
        p = pathlib.Path(tmp) / "min.epub"
        _mini_epub(p, b"<html><body><p>texto <b>roto</p></body></html>")
        text = research.read_local(p)
    assert "texto" in text and "roto" in text, text


def test_pdf_pages_are_marked_and_never_published() -> None:
    """A PDF's text carries [p. N]; the article it feeds must not.

    The markers exist so a verbatim quote can be cited as (Apellido, año, p. N)
    instead of pageless or invented. They are read by the drafting model and
    stripped from what it writes back.
    """
    import pypdf

    class _Page:
        def __init__(self, text: str) -> None:
            self.text = text

        def extract_text(self) -> str:
            return self.text

    class _Reader:
        def __init__(self, _stream) -> None:
            self.pages = [_Page("Primera página."), _Page("Segunda página."),
                          _Page("   ")]           # página en blanco al final

    old = pypdf.PdfReader
    pypdf.PdfReader = _Reader
    try:
        text = research.pdf_text(b"%PDF-fake")
    finally:
        pypdf.PdfReader = old
    assert "[p. 1]" in text and "[p. 2]" in text, text
    # La página vacía no deja marca: si no, el texto termina en «[p. 3]» y _cut()
    # lo lee como una obra cortada a la mitad.
    assert "[p. 3]" not in text and not research._cut(text), text

    drafted = "[p. 12]\nComo sostiene el autor (Kurz, 1998, p. 12), el valor [p. 13] se disuelve."
    clean = pipeline._strip_preamble(drafted)
    assert "[p." not in clean, clean
    assert "(Kurz, 1998, p. 12)" in clean, clean
    src = research.Source(key="Kurz, 1998", authors=["Robert Kurz"], year="1998")
    used, unknown = research.verify_citations(clean, [src])
    assert used == ["Kurz, 1998"] and not unknown, (used, unknown)


def test_every_format_prints_its_reference_list() -> None:
    """«corto» has no abstract and no keywords, but it does cite — so it needs
    the bibliography those citations point at."""
    src = research.Source(key="Kurz, 1998", title="Der Kollaps der Modernisierung",
                          authors=["Robert Kurz"], year="1998")
    with tempfile.TemporaryDirectory() as tmp:
        run = pipeline.Run(fmt="corto", dir=pathlib.Path(tmp), log=lambda *_: None)
        doc = pipeline.assemble(run, {"titulo": "T"}, {}, "Cuerpo (Kurz, 1998).",
                                [src], {"citas_validas": ["Kurz, 1998"]}, {}, {})
    assert "## Referencias" in doc, doc
    assert "Kurz, R. (1998)" in doc, doc
    assert "Palabras clave" not in doc, doc      # eso sí es aparato académico


def test_citations_require_the_work_year_and_real_coauthors() -> None:
    sources = research.assign_keys([
        research.Source(title="A", authors=["Ana Smith", "Ben Jones"], year="2020"),
        research.Source(title="B", authors=["Ana Smith"], year="2021"),
        research.Source(title="C", authors=["Ana Smith"], year="2021"),
        research.Source(title="D", authors=["Ben Jones"], year="")])
    used, unknown = research.verify_citations(
        "Smith and Jones (2020) argue; (Smith & Jones, 2020) agrees. "
        "(Smith, 2021b) differs. Jones (n.d.) notes a limit. "
        "(Smith, 2099), (Smith, 2021), (Smith & Invented, 2020) are unsupported.", sources)
    assert set(used) == {"Smith, 2020", "Smith, 2021b", "Jones, s/f"}, used
    assert len(unknown) == 3 and any("2099" in s for s in unknown), unknown
    assert research.bibliography(sources, []) == ""
    assert "(2021b)" in sources[2].citation()
    assert ", & Jones, B. (2020)" in sources[0].citation(lang="en")
    used, unknown = research.verify_citations(
        "Grouped (Smith, 2020; Jones, n.d.; Invented, 2001).", sources)
    assert set(used) == {"Smith, 2020", "Jones, s/f"} and unknown == ["(Invented, 2001)"], (used, unknown)


def test_rewrite_preserves_citations_quotes_numbers_and_notes_in_both_languages() -> None:
    from unittest.mock import patch
    for lang, opening in (("en", "The record shows"), ("es", "El registro muestra")):
        text = (opening + " 42 casos (Smith, 2020, p. 7). «La cifra es provisional»[1]. "
                + "contexto " * 100 + "\n\n[1] Smith (2020) explica el límite.")
        good = text.replace(opening, "El documento confirma" if lang == "es" else "The document confirms")
        candidates = [good,
                      good.replace("(Smith, 2020, p. 7)", "(Smith, 2020, p. 8)"),
                      good.replace("(Smith, 2020, p. 7)", ""),
                      good.replace("42", "43"),
                      good.replace("La cifra es provisional", "La cifra es definitiva"),
                      good.replace("[1]", "[2]"), good + " (Newauthor, 2025).",
                      good + " extra" * 30]
        for i, candidate in enumerate(candidates):
            with patch.object(llm, "chat", return_value=candidate):
                got = humanize._rewrite(text, [], "style", lambda *_: None, lang=lang)
            assert got == (good if i == 0 else text), (lang, i, got)


def test_rewrite_receives_neighbours_and_native_editorial_rules() -> None:
    from unittest.mock import patch
    text = "## Heading\n\n" + ("word " * 99 + "FIRST") + "\n\n" + ("SECOND " + "word " * 100).strip()
    prompts = []
    def echo(model, prompt, **kwargs):
        prompts.append(prompt)
        return prompt.split("=== BLOCK TO REWRITE ===\n")[1].split("\n=== END OF BLOCK ===")[0]
    with patch.object(llm, "chat", side_effect=echo):
        assert humanize._rewrite(text, [], "style", lambda *_: None,
                                 batch_words=90, lang="en") == text
    assert len(prompts) == 2
    second = next(p for p in prompts if "\nSECOND " in p.split("=== BLOCK TO REWRITE ===")[1])
    assert "FIRST" in second.split("=== BLOCK TO REWRITE ===")[0]
    assert all("Write idiomatic English" in p and "MANDATORY QUOTAS" not in p for p in prompts)
    assert "español rioplatense natural" in style.writing_rules("es")


def test_detector_unavailability_is_not_a_pass_and_local_is_diagnostic() -> None:
    from unittest.mock import patch
    text = "The argument is supported by the evidence."
    local = {"score": 99, "issues": ["metric quota"], "issues_texto": []}
    with patch.object(humanize, "local_score", return_value=local), \
         patch.object(humanize, "llm_judges", return_value=[]) as judges, \
         patch.object(humanize, "external_detectors", return_value=[]), \
         patch.object(humanize, "_rewrite") as rewrite:
        _, report = humanize.humanize(text, style_block="style", lang="en", log=lambda *_: None)
        assert report["passed"] is None and report["final_score"] is None, report
        rewrite.assert_not_called()
        assert judges.call_args.kwargs["lang"] == "en"
        judges.return_value = [{"model": "judge", "ai_probability": 12}]
        _, report = humanize.humanize(text, style_block="style", lang="en", log=lambda *_: None)
        assert report["passed"] is True and report["final_score"] == 12, report
        assert report["text_hash"] == humanize.text_hash(text)


def test_humanize_returns_the_best_measured_text_when_rewrite_regresses() -> None:
    from unittest.mock import patch
    text, worse = "Original article.", "Worse article."
    local = {"score": 0, "issues": ["numeric floor"], "issues_texto": ["repeated opening"]}
    scores = [[{"model": "judge", "ai_probability": n}] for n in (55, 85)]
    with patch.object(humanize, "local_score", return_value=local), \
         patch.object(humanize, "llm_judges", side_effect=scores), \
         patch.object(humanize, "external_detectors", return_value=[]), \
         patch.object(humanize, "_rewrite", return_value=worse) as rewrite:
        got, report = humanize.humanize(text, rounds=2, style_block="style", log=lambda *_: None)
    assert got == text and report["final_score"] == 55 and report["passed"] is False
    assert report["selected_round"] == 1 and report["text_hash"] == humanize.text_hash(got)
    assert rewrite.call_args.args[1] == ["repeated opening"]


def test_judges_reject_invalid_scores_and_use_the_requested_language() -> None:
    from unittest.mock import patch
    for score in (None, True, "NaN", "inf", -1, 101, "not a score"):
        with patch.object(llm, "chat_json", return_value={"ai_probability": score}):
            assert humanize.llm_judges("Text", ["judge"], lambda *_: None, lang="en") == []
    with patch.object(llm, "chat_json", return_value={"ai_probability": "20", "verdict": "human"}) as chat:
        assert humanize.llm_judges("Text", ["judge"], lambda *_: None, lang="en")[0]["ai_probability"] == 20
        assert "English text" in chat.call_args.args[1]


def test_english_assembly_and_notes_have_no_spanish_labels() -> None:
    from unittest.mock import patch
    source = research.Source(key="Smith, 2020", title="Original Title", authors=["Ana Smith"], year="2020")
    with tempfile.TemporaryDirectory() as tmp, patch.object(pipeline, "LANG", "en"):
        run = pipeline.Run(dir=pathlib.Path(tmp), log=lambda *_: None)
        doc = pipeline.assemble(run, {"titulo": "English title"},
                                {"resumen": "An abstract.", "palabras_clave": ["work"]},
                                "Evidence (Smith, 2020).", [source], {}, {}, {})
    assert "By " in doc and "Keywords: work" in doc and "## References" in doc
    assert "Palabras clave" not in doc and "## Referencias" not in doc and "Por " not in doc
    assert pipeline._split_notes("Body[1].\n\nNOTES:\n[1] A note.") == ("Body[1].", ["[1] A note."])


def test_auto_publication_requires_a_confirmed_detector_pass() -> None:
    import main as cli
    run = pipeline.Run(state={"verdict": {"publicable": True}})
    for detector in ({}, {"skipped": True}, {"final_score": None},
                     {"final_score": 0, "passed": None}, {"final_score": 10, "passed": False},
                     {"final_score": float("nan"), "passed": True}):
        run.state["detector"] = detector
        assert cli.wp_status(run, "auto", 25) == "draft", detector


def test_english_without_a_corpus_or_guide_still_has_writing_context() -> None:
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmp, \
         patch.object(style, "GUIDE_EN", pathlib.Path(tmp) / "missing.md"), \
         patch.object(style, "articles_en", return_value=[]), \
         patch.object(style, "corpus_fingerprint", return_value={}):
        assert "STYLE GUIDE" in style.style_block(lang="en")


def test_rejected_humanization_cannot_lend_its_score_to_the_original() -> None:
    from unittest.mock import patch
    source = research.Source(key="Smith, 2020", title="A", authors=["Ana Smith"], year="2020")
    original = "Original argument (Smith, 2020)."
    substituted = "Replacement argument (Invented, 2024)."
    def fake_humanize(text, **kwargs):
        body = substituted if kwargs.get("rounds") != 1 else text
        score = 5 if body == substituted else 75
        return body, {"lang": "es", "passed": score < 25, "final_score": score,
                      "text_hash": humanize.text_hash(body)}
    patches = {
        "pick_topic": lambda run: {"titulo": "Title"},
        "do_research": lambda *a: ([source], []),
        "make_outline": lambda *a: {},
        "draft": lambda *a: original,
        "review": lambda *a: {"veredicto": "aprobado"},
        "final_approval": lambda *a: {"publicable": True},
    }
    with tempfile.TemporaryDirectory() as tmp, patch.multiple(pipeline, **patches), \
         patch.object(pipeline, "LANG", "es"), patch.object(humanize, "humanize", side_effect=fake_humanize):
        run = pipeline.Run(dir=pathlib.Path(tmp), log=lambda *_: None)
        pipeline.run_pipeline(run)
        assert run.state["detector"]["final_score"] == 75, run.state
        assert run.state["detector"]["passed"] is False
        doc = (run.dir / "05_final.md").read_text(encoding="utf-8")
        assert original in doc and "Invented" not in doc


def test_review_and_revision_can_read_the_source_evidence() -> None:
    from unittest.mock import patch
    source = research.Source(key="Smith, 2020", title="A", authors=["Ana Smith"], year="2020",
                             fulltext="VERIFIABLE PASSAGE: the observation does not establish causation.")
    report = {"veredicto": "revisar", "problemas": [{"gravedad": "alta", "problema": "overclaim"}]}
    with patch.object(llm, "chat_json", side_effect=[report, {"parches": []}]) as chat:
        run = pipeline.Run(log=lambda *_: None)
        report = pipeline.review(run, "Evidence (Smith, 2020).", {}, [source])
        pipeline.revise(run, "Evidence (Smith, 2020).", report, [source])
    assert len(chat.call_args_list) == 2
    assert all("VERIFIABLE PASSAGE" in call.args[1] for call in chat.call_args_list)


def test_detect_cli_passes_the_detected_language_to_judges() -> None:
    from unittest.mock import patch
    import main as cli
    text = "The record of the trial is in the report and the evidence is there. " * 25
    with tempfile.TemporaryDirectory() as tmp:
        file = pathlib.Path(tmp) / "english.md"
        file.write_text(text, encoding="utf-8")
        with patch.object(sys, "argv", ["main.py", "--detect", str(file)]), \
             patch.object(pipeline, "LANG", "es"), patch.object(humanize, "LANG", "es"), \
             patch.object(humanize, "local_score", return_value={"score": 5, "issues": []}), \
             patch.object(humanize, "llm_judges", return_value=[]) as judges, \
             patch.object(humanize, "external_detectors", return_value=[]) as external:
            assert cli.main() == 0
            assert judges.call_args.kwargs["lang"] == "en"
            assert external.call_args.kwargs["lang"] == "en"


def test_draft_retries_invalid_length_and_never_caches_a_failed_attempt() -> None:
    from unittest.mock import patch
    valid = "The report and the record describe the evidence with care. " * 40
    words = len(valid.split())
    outline = {"secciones": [{"n": 1, "titulo": "One", "palabras": words}]}
    with tempfile.TemporaryDirectory() as tmp, patch.object(pipeline, "LANG", "en"), \
         patch.object(style, "style_block", return_value="style"), \
         patch.object(llm, "chat", side_effect=["Too short.", valid]) as chat:
        run = pipeline.Run(dir=pathlib.Path(tmp), log=lambda *_: None)
        got = pipeline.draft(run, {"titulo": "Title"}, outline, [])
        assert valid.strip() in got and chat.call_count == 2
        assert "Too short" not in (run.dir / "04_sec01.md").read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp, patch.object(pipeline, "LANG", "en"), \
         patch.object(style, "style_block", return_value="style"), \
         patch.object(llm, "chat", return_value="Still short."):
        run = pipeline.Run(dir=pathlib.Path(tmp), log=lambda *_: None)
        try:
            pipeline.draft(run, {"titulo": "Title"}, outline, [])
        except RuntimeError:
            assert not (run.dir / "04_sec01.md").exists()
        else:
            raise AssertionError("Invalid draft was accepted")


def test_rewrite_rejects_a_switch_to_the_other_language() -> None:
    from unittest.mock import patch
    english = "The report and the record describe the evidence with care. " * 12
    spanish = "El informe y los datos muestran que la prueba tiene límites. " * 11
    assert style.detect_language(english) == "en" and style.detect_language(spanish) == "es"
    assert style.detect_language("Too short.") is None
    for original, replacement, lang in ((english, spanish, "en"), (spanish, english, "es")):
        with patch.object(llm, "chat", return_value=replacement):
            got = humanize._rewrite(original, [], "style", lambda *_: None, lang=lang)
        assert got.strip() == original.strip()


def test_approved_review_still_applies_material_corrections() -> None:
    from unittest.mock import patch
    source = research.Source(key="Smith, 2020", title="A", authors=["Ana Smith"], year="2020")
    patches = {"pick_topic": lambda run: {"titulo": "Title"},
               "do_research": lambda *a: ([source], []), "make_outline": lambda *a: {},
               "draft": lambda *a: "Evidence (Smith, 2020).",
               "review": lambda *a: {"veredicto": "aprobado", "problemas": [{"gravedad": "media"}]},
               "final_approval": lambda *a: {"publicable": True}}
    with tempfile.TemporaryDirectory() as tmp, patch.multiple(pipeline, **patches), \
         patch.object(pipeline, "revise", return_value="Corrected (Smith, 2020).") as revise:
        run = pipeline.Run(dir=pathlib.Path(tmp), log=lambda *_: None)
        pipeline.run_pipeline(run, detector_rounds=0)
        revise.assert_called_once()
        assert "Corrected" in (run.dir / "05_final.md").read_text(encoding="utf-8")


def test_a_rewrite_is_not_blamed_for_citations_the_draft_already_lacked() -> None:
    """One unbacked citation in the draft used to discard every humanized rewrite."""
    from unittest.mock import patch
    source = research.Source(key="Smith, 2020", title="A", authors=["Ana Smith"], year="2020")
    draft = "Evidence (Smith, 2020) and (Nadie, 1999)."
    rewritten = "Better evidence (Smith, 2020) and (Nadie, 1999)."
    det = {"rounds": [{"round": 1, "worst": 80.0, "text_hash": humanize.text_hash(draft)}],
           "final_score": 70.0, "passed": False, "lang": "es",
           "text_hash": humanize.text_hash(rewritten)}
    patches = {"pick_topic": lambda run: {"titulo": "Title"},
               "do_research": lambda *a: ([source], []), "make_outline": lambda *a: {},
               "draft": lambda *a: draft,
               "review": lambda *a: {"veredicto": "aprobado", "problemas": []},
               "revise": lambda run, text, *a: text,
               "final_approval": lambda *a: {"publicable": False}}
    with tempfile.TemporaryDirectory() as tmp, patch.multiple(pipeline, **patches), \
         patch.object(humanize, "humanize", return_value=(rewritten, det)) as hz:
        run = pipeline.Run(dir=pathlib.Path(tmp), log=lambda *_: None)
        pipeline.run_pipeline(run, detector_rounds=2)
        kept = (run.dir / "04_draft_humanizado.md").read_text(encoding="utf-8")
        assert kept == rewritten, kept
    # a rewrite that drops a verified citation is still refused, and the round-1
    # score of the kept draft is reused instead of judging it again
    lossy = "Better evidence and (Nadie, 1999)."
    det = {**det, "text_hash": humanize.text_hash(lossy)}
    with tempfile.TemporaryDirectory() as tmp, patch.multiple(pipeline, **patches), \
         patch.object(humanize, "humanize", return_value=(lossy, det)) as hz:
        run = pipeline.Run(dir=pathlib.Path(tmp), log=lambda *_: None)
        pipeline.run_pipeline(run, detector_rounds=2)
        assert (run.dir / "04_draft_humanizado.md").read_text(encoding="utf-8") == draft
        saved = json.loads((run.dir / "07_detector_humanizado.json").read_text(encoding="utf-8"))
        assert saved["text_hash"] == humanize.text_hash(draft) and saved["final_score"] == 80.0
        assert hz.call_count == 1, hz.call_count


def test_fold_transliterates_letters_nfkd_cannot_split() -> None:
    srcs = research.assign_keys([research.Source(title="T", authors=["Czesław Miłosz"], year="1953")])
    used, unknown = research.verify_citations("según (Milosz, 1953)", srcs)
    assert used == ["Miłosz, 1953"] and not unknown, (used, unknown)


def test_an_unparseable_201_still_yields_a_receipt() -> None:
    class Created:
        status_code = 201

        def json(self):
            raise ValueError("html, not json")

    post = publish._created(Created(), lambda *_: None)
    assert isinstance(post, dict) and post, post
    Created.status_code = 200          # a home page, not a created post
    try:
        publish._created(Created(), lambda *_: None)
    except RuntimeError:
        return
    raise AssertionError("a 200 HTML page was taken for a created post")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok   {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERR  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} pasaron")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
