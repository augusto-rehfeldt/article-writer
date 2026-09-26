"""Live, source-controlled bilingual writing check (uses the configured API key).

    python bench_bilingual.py --backend hyper

The dataset is explicitly fictional. Generated prose and the model's editorial
assessment are saved separately from deterministic checks; neither proves authorship.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib

from dotenv import load_dotenv

load_dotenv()

import humanize
import llm
import pipeline
import research
from build_corpus_en import is_english


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="hyper", choices=["hyper", "grok", "zen"])
    args = parser.parse_args()
    llm.configure([args.backend])
    # One bounded attempt per call: a live check must not wait through provider retries.
    def chat(model, prompt, system=None, **kwargs):
        return llm._send(args.backend, llm._as(args.backend, model), prompt, system, retries=1,
                         temperature=kwargs.get("temperature", 0.6), max_tokens=24000)
    llm.chat = chat
    # This benchmark evaluates the selected provider's judges. External services
    # remain available in the application; they are not a reproducible test fixture.
    humanize.external_detectors = lambda *a, **kw: []
    root = pathlib.Path(__file__).parent / "output" / dt.datetime.now().strftime("bilingual-check-%Y%m%d-%H%M%S")
    root.mkdir(parents=True)
    sources = research.assign_keys([
        research.Source(title="Illustrative library pilot: attendance log", authors=["Pilot Team"],
                        year="2024", kind="report", fulltext="""FICTIONAL TEST DATA, not a real study.
A library tested longer opening hours for four weeks in 2024. Before the pilot,
weekly visits were 1200; in the final week they were 1500. Weekly paid staff hours
rose from 70 to 90. The log counts visits, not distinct visitors. No comparison
library, visitor survey or seasonal adjustment was available. No costs were recorded.
These observations alone cannot establish that longer opening hours caused the increase."""),
        research.Source(title="Illustrative library pilot: staff note", authors=["Staff Committee"],
                        year="2024", kind="report", fulltext="""FICTIONAL TEST DATA, not a real study.
Staff reported that the later closing time made returning books easier for some visitors.
The note did not count how many visitors benefited. It did not record wages or fatigue.
The committee recommended a longer observation period before changing the timetable
permanently. The recommendation was not an approval of permanent extended hours.""")])
    research.save_dossier(sources, root / "sources.json")
    results = {}
    for lang in ("en", "es"):
        pipeline.LANG = lang
        title = "What longer library hours can tell us" if lang == "en" else "Qué nos dicen los horarios de una biblioteca"
        run = pipeline.Run(fmt="corto", dir=root / lang)
        run.dir.mkdir()
        topic = {"titulo": title, "hipotesis": "A fictional pilot suggests a benefit but cannot establish causation or cost efficiency."}
        outline = {"titulo_final": title, "secciones": [{
            "n": 1, "titulo": title, "tesis": topic["hipotesis"], "palabras": 500,
            "contenido": "Explain the explicitly fictional case to a general reader. Compare attendance and staff hours; distinguish visits from visitors, evidence from inference, and a recommendation from an adopted policy. Use only the supplied facts. Do not describe this as a real study.",
            "fuentes": [s.key for s in sources]}]}
        draft = pipeline.draft(run, topic, outline, sources)
        run.save("draft.md", draft)
        peer_review = pipeline.review(run, draft, topic, sources)
        run.save("peer_review.json", peer_review)
        revised = pipeline.revise(run, draft, peer_review, sources)
        run.save("revised.md", revised)
        edited, detector = humanize.humanize(revised, rounds=2, lang=lang,
                                             register=pipeline.FORMATS["corto"]["register"])
        run.save("edited.md", edited)
        run.save("detector.json", detector)
        used, unknown = research.verify_citations(edited, sources)
        prose = "\n".join(line for line in edited.splitlines() if not line.startswith("#"))
        draft_prose = "\n".join(line for line in draft.splitlines() if not line.startswith("#"))
        checks = {"language": is_english(prose) == (lang == "en"),
                  "draft_length": 450 <= len(draft_prose.split()) <= 550,
                  "rewrite_length": 0.9 * len(revised.split()) <= len(edited.split()) <= 1.1 * len(revised.split()),
                  "citations": len(used) == 2 and not unknown,
                  "rewrite_fidelity": humanize._protected(revised) == humanize._protected(edited)}
        review = llm.chat_json(llm.PRO, """Assess this article as a bilingual editor.
Score each criterion from 1 (poor) to 5 (excellent): idiomatic language, clarity,
coherence, specificity, and fidelity to the source packet. Quote concrete problems.
Do not infer authorship or reward decorative variation in sentence length. Penalize
invented details, unsupported certainty, and presenting fictional data as a real study.
Return JSON with keys idiomatic_language, clarity, coherence, specificity, fidelity
(integer scores), and problems (a list).\n\nSOURCES:\n"""
            + "\n".join(s.brief(4000) for s in sources) + "\n\nARTICLE:\n" + edited,
            temperature=0.2)
        run.save("editorial_review.json", review)
        editorial_passed = all(isinstance(review.get(k), (int, float)) and review[k] >= 4
                              for k in ("idiomatic_language", "clarity", "coherence", "specificity", "fidelity"))
        results[lang] = {"checks": checks, "editorial_review": review,
                         "editorial_passed": editorial_passed,
                         "detector_passed": detector["passed"], "detector_score": detector["final_score"]}
        print(lang, json.dumps(results[lang], ensure_ascii=False), flush=True)
    (root / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {root}")
    return 0 if all(all(r["checks"].values()) and r["editorial_passed"] for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
