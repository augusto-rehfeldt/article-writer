"""CLI entry point.

    python main.py --setup                          # download corpus + build the style guide
    python main.py                                  # interactive wizard
    python main.py --fmt medium --mode auto         # topic to article, no questions
    python main.py --topic "..." --fmt long         # user-fixed topic
    python main.py --resume                         # resume the latest run
    python main.py --resume output/2026...-slug     # resume a given run
    python main.py --detect text.md                 # only run the detectors
    python main.py --continuous 0 --publish auto    # never stops, picks its own topics
    python main.py --provider claude --backups hyper,go --pro opus --flash sonnet

The wizard opens with provider, models, backup order and, if asked, each backup's
models; the flags above plus `--models "zen:glm-5.2/deepseek-v4-pro"` are the
non-interactive version, and `.env` (AW_BACKEND, AW_MODEL_PRO, AW_MODEL_FLASH,
AW_MODELS) holds the defaults.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time
import traceback

from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).parent
load_dotenv(ROOT / ".env")

# Windows consoles default to cp1252 and choke on « » — this pipeline is all Spanish.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import humanize  # noqa: E402
import llm  # noqa: E402
import pipeline  # noqa: E402
import publish  # noqa: E402
import style  # noqa: E402
import ui  # noqa: E402


# Same reader the recovery prompt in `llm` uses: a menu answer means the same
# thing wherever it is typed.
resolve_choice = llm.resolve_choice


def ask_choice(prompt: str, options: list[str], default: str) -> str:
    while True:
        pick = resolve_choice(
            input(f"{ui.c(prompt, ui.BOLD)} {ui.c('[' + default + ']', ui.DIM)}: ").strip()
            or default, options)
        if pick:
            return pick
        print(ui.c(f"Inválido. Elegí 1-{len(options)} o uno de {options}.", ui.RED))


def ask_free(prompt: str, options: list[str], default: str) -> str:
    """Like ask_choice, but a name outside the catalogue is accepted as typed.

    Provider catalogues go stale the week a new model ships; refusing an unlisted
    name would mean editing the source to use one.
    """
    raw = input(f"{ui.c(prompt, ui.BOLD)} {ui.c('[' + default + ']', ui.DIM)}: ").strip()
    if not raw:
        return default
    return resolve_choice(raw, options) or raw


def ask_order(prompt: str, options: list[str], default: str) -> list[str]:
    """A comma-separated ordered subset. «no» = empty list."""
    while True:
        raw = input(f"{ui.c(prompt, ui.BOLD)} {ui.c('[' + default + ']', ui.DIM)}: ").strip()
        raw = raw or default
        if raw.lower() in ("no", "ninguno", "-"):
            return []
        picks = [resolve_choice(x.strip(), options) for x in raw.split(",") if x.strip()]
        if all(picks):
            return list(dict.fromkeys(picks))
        print(ui.c(f"Inválido. Nombres o números de {options}, separados por coma.", ui.RED))


def save_env(pairs: dict[str, str], path: pathlib.Path = ROOT / ".env") -> None:
    """Upsert KEY=value lines, leaving every other line (the API key!) untouched."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    for key, value in pairs.items():
        for i, line in enumerate(lines):
            if line.split("=", 1)[0].strip() == key:
                lines[i] = f"{key}={value}"
                break
        else:
            lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# The shared menu reads keys straight from the console (Tab re-sorts), so piped or
# scripted input keeps the plain numbered menu below.
SHARED_MENU = sys.stdin.isatty()


def edit_interests(read=input) -> None:
    """First `--setup` copies interests.example.txt to interests.txt (personal,
    gitignored); every `--setup` shows the list and offers to replace it."""
    f = pipeline.INTERESTS_FILE
    if not f.exists():
        f.write_text(pipeline.INTERESTS_EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    print(ui.c(f"Intereses del autor ({f}):", ui.BOLD))
    for area in pipeline.interest_areas():
        print(f"  - {area}")
    if not sys.stdin.isatty() and read is input:
        return
    first = read("\n¿Reemplazarlos? Escribí un área por línea y una línea vacía para "
                 "terminar (Enter directo = dejarlos así; también podés editar el archivo): ").strip()
    if not first:
        return
    areas = [first]
    while (line := read("  + ").strip()):
        areas.append(line)
    header = [ln for ln in pipeline.INTERESTS_EXAMPLE.read_text(encoding="utf-8").splitlines()
              if ln.startswith("#")]
    f.write_text("\n".join(header + areas) + "\n", encoding="utf-8")
    print(ui.c(f"Guardados {len(areas)} intereses en {f.name}.", ui.GREEN))


def pick_pair(prov: str) -> tuple[str, str]:
    """Ask for a provider's PRO/FLASH pair.

    Providers book writer also serves go through its shared menu (`choose_ai`: live
    catalogue, context, price, AA index), so every AI script picks models the same way.
    """
    if SHARED_MENU and prov in llm.SHARED:
        if str(llm.BOOK_WRITER) not in sys.path:
            sys.path.insert(0, str(llm.BOOK_WRITER))
        from ai_book_creator.cli import choose_ai
        _, _, (pro, flash) = choose_ai(
            llm.SHARED[prov], roles=("PRO", "FLASH"),
            defaults=(llm.PROVIDERS[prov]["pro"], llm.PROVIDERS[prov]["flash"]),
            state_file=ROOT / "output" / "provider_state.json")
        return pro, flash
    catalogue = llm.catalogue(prov)
    live = [m for m in catalogue if m not in llm.PROVIDERS[prov]["models"]]
    print("\n" + ui.c(f"Modelos de {prov}:", ui.BOLD) + " "
          + ui.c("(o escribí cualquier otro nombre)", ui.DIM))
    if prov == "oauth":
        print(ui.c("  " + llm.oauth_value_summary(), ui.DIM))
    if live:
        print(ui.c(f"  ({len(live)} además de los de siempre, leídos del proveedor ahora)",
                   ui.DIM))
    for i, m in enumerate(catalogue, 1):
        mark = ui.c(" nuevo", ui.GREEN) if m in live else ""
        if m.endswith("-free"):
            mark += ui.c(" gratis", ui.GREEN)
        cost = llm.oauth_cost_label(m) if prov == "oauth" else ""
        if cost:
            mark += ui.c("  " + cost, ui.DIM)
        print(f"  {ui.c(str(i) + ')', ui.CYAN)} {m}{mark}")
    pro = ask_free("PRO   — tema, auditoría del esquema, revisión, aprobación",
                   catalogue, llm.PROVIDERS[prov]["pro"])
    flash = ask_free("FLASH — consultas, esquema, redacción, reescritura",
                     catalogue, llm.PROVIDERS[prov]["flash"])
    return pro, flash


def pick_models() -> None:
    """Provider, models and backup order. Writes straight into llm's router."""
    before = (tuple(llm.CHAIN), llm.PRO, llm.FLASH)
    provs = list(llm.PROVIDERS)
    print("\n" + ui.c("Proveedores:", ui.BOLD))
    for i, k in enumerate(provs, 1):
        p = llm.PROVIDERS[k]
        print(f"  {ui.c(str(i) + ')', ui.CYAN)} {k:<9} {ui.c(p['label'], ui.DIM)}")
        print(f"     {ui.c('por defecto: PRO=' + p['pro'] + '  FLASH=' + p['flash'], ui.DIM)}")
        if k == "oauth":
            print(f"     {ui.c(llm.oauth_value_summary(), ui.DIM)}")
    main_prov = ask_choice("Proveedor principal", provs, llm.CHAIN[0])

    pro, flash = pick_pair(main_prov)

    rest = [p for p in provs if p != main_prov]
    print("\n" + ui.c("Respaldos", ui.BOLD) + ui.c(
        " — en orden, separados por coma; «no» para ninguno.", ui.DIM))
    for i, k in enumerate(rest, 1):
        print(f"  {ui.c(str(i) + ')', ui.CYAN)} {k}")
    backups = ask_order("Respaldos", rest, ",".join(rest))

    # Each backup answers with its own pair; asking for all of them every time is
    # six extra prompts for a chain that usually never gets used, so it is opt-in.
    models: dict[str, tuple[str, str]] = {}
    if backups and input(
            f"{ui.c('¿Elegir los modelos de los respaldos?', ui.BOLD)} "
            f"{ui.c('(Enter = los de cada proveedor)', ui.DIM)} [s/N]: "
    ).strip().lower().startswith("s"):
        for b in backups:
            models[b] = pick_pair(b)

    llm.configure([main_prov, *backups], pro, flash, models)
    print("\n" + ui.c("→ " + llm.describe(), ui.GREEN))
    # Nothing changed: whatever .env (or the built-in default) already says is
    # exactly this, so there is nothing to save and nothing to ask.
    if (tuple(llm.CHAIN), llm.PRO, llm.FLASH) == before and not models:
        return
    pairs = {"AW_BACKEND": ",".join(llm.CHAIN),
             "AW_MODEL_PRO": llm.PRO, "AW_MODEL_FLASH": llm.FLASH}
    if models:
        pairs["AW_MODELS"] = ",".join(
            f"{b}:{llm.PROVIDERS[b]['pro']}/{llm.PROVIDERS[b]['flash']}"
            for b in models)
    if input(f"{ui.c('¿Guardar como predeterminado en .env?', ui.BOLD)} [s/N]: "
             ).strip().lower().startswith("s"):
        save_env(pairs)
        print(ui.c("  guardado.", ui.GREEN))


def wizard(args: argparse.Namespace) -> argparse.Namespace:
    print("\n" + ui.rule("Escritor de artículos de ciencias sociales"))
    pick_models()
    print("\n" + ui.c("Formatos:", ui.BOLD))
    fmts = list(pipeline.FORMATS)
    for i, k in enumerate(fmts, 1):
        v = pipeline.FORMATS[k]
        print(f"  {ui.c(str(i) + ')', ui.CYAN)} {k:<10} {pipeline.wrange(v):>15} palabras "
              f"— {ui.c(v['kind'], ui.DIM)}")
    args.fmt = ask_choice("Formato", fmts, args.fmt)
    print("\n" + ui.c("Modos:", ui.BOLD))
    print(f"  {ui.c('1)', ui.CYAN)} auto      de punta a punta sin preguntar")
    print(f"  {ui.c('2)', ui.CYAN)} asistido  te consulta en tema, esquema, correcciones y aprobación")
    args.mode = ask_choice("Modo", ["auto", "asistido"], args.mode)
    args.tema = input("\nTema o pista temática (Enter = que lo elija el modelo): ").strip()
    if args.tema:
        args.tema_exacto = input(
            "¿Escribo sobre ESE tema tal cual, sin proponerte alternativas? [S/n]: "
        ).strip().lower() not in ("n", "no")
    print()
    return args


def wp_status(run: pipeline.Run, policy: str, threshold: float) -> str | None:
    """Translate --publish into a WordPress post status. None = do not upload.

    ``auto`` only goes live when PRO approved the piece *and* the detectors came
    back under threshold; anything short of that lands as a draft for review.
    """
    if policy == "no":
        return None
    if policy in ("borrador", "vivo"):
        return "draft" if policy == "borrador" else "publish"
    verdict = run.state.get("verdict") or {}
    detector = run.state.get("detector") or {}
    score = detector.get("final_score")
    clean = (detector.get("passed") is True and score is not None
             and 0 <= float(score) < threshold)
    # `is True`: the string "false" is truthy, and going live has no undo
    return "publish" if verdict.get("publicable") is True and clean else "draft"


CURRENT: pipeline.Run | None = None  # what Ctrl-C tells the user to resume


# English --fmt names -> the pipeline's format keys.
FORMAT_NAMES = {"short": "corto", "medium": "medio", "long": "largo", "paper": "paper",
                "thesis": "tesis", "book": "libro", "debate": "discusion"}


def latest_run() -> str:
    """The run folder touched most recently (any finished stage counts)."""
    runs = [d for d in pipeline.OUTPUT.iterdir() if (d / "01_topic.json").exists()] \
        if pipeline.OUTPUT.exists() else []
    newest = lambda d: max(f.stat().st_mtime for f in d.iterdir())  # noqa: E731
    return str(max(runs, key=newest)) if runs else ""


def resume_hint() -> None:
    run = CURRENT
    if run is None or run.dir == pipeline.OUTPUT:
        print("\nDetenido antes de elegir tema; no hay nada que retomar.", flush=True)
        return
    print(f"\nDetenido. Cada etapa terminada quedó en {run.dir}"
          f"\nPara retomar: python main.py --resume", flush=True)


def adopt(run: pipeline.Run) -> None:
    """A resumed run takes the format and language it was started with."""
    saved = run.load("00_corrida.json") or {}
    if saved.get("fmt") in pipeline.FORMATS:
        run.fmt = saved["fmt"]
    if saved.get("lang") in ("es", "en"):
        pipeline.LANG = humanize.LANG = saved["lang"]


def run_once(args: argparse.Namespace, resume: str = "",
             run: pipeline.Run | None = None) -> pipeline.Run:
    run = run or pipeline.Run(fmt=args.fmt, mode=args.mode, brief=args.tema,
                              exact_topic=args.tema_exacto and bool(args.tema),
                              ask_library=not args.sin_biblioteca)
    global CURRENT
    CURRENT = run
    if resume:
        run.dir = pathlib.Path(resume)
        adopt(run)
        ui.log(f"[retomar] {run.dir}")
    threshold = 101.0 if args.no_detector else args.threshold
    pipeline.run_pipeline(run, rounds=args.rounds,
                          detector_rounds=0 if args.no_detector else args.detector_rounds,
                          threshold=threshold)
    status = wp_status(run, args.publicar, threshold)
    if status:
        publish.publish_run(run.dir, status=status)
    return run


def loop(args: argparse.Namespace) -> int:
    """Write article after article, picking each topic itself.

    ``--continuous 0`` never stops. A failed article must not end the run: the next
    topic is unrelated to the one that broke, so the loop logs and carries on.
    """
    n, done, failures = args.continuo, 0, 0
    resumable: pipeline.Run | None = None   # a crashed run, retried once from its cache
    # --resume with --continuous: finish the cached run first, then keep looping.
    initial_resume = getattr(args, "resume", "") or ""
    brief = args.tema   # --topic steers the first new article only; the rest pick their own
    while n == 0 or done + failures < n:
        print("\n" + ui.rule(f"artículo {done + failures + 1}"
                             f"{'' if n == 0 else f' de {n}'}"))
        run, resumable = resumable, None
        retry = run is not None
        if run is None:
            run = pipeline.Run(fmt=args.fmt, mode=args.mode,
                               brief="" if initial_resume else brief,
                               exact_topic=args.tema_exacto and bool(brief) and not initial_resume,
                               ask_library=not args.sin_biblioteca)
            if not initial_resume:
                brief = ""
            if initial_resume:
                run.dir = pathlib.Path(initial_resume)
                adopt(run)
                ui.log(f"[continuo] retomo: {run.dir}")
                retry = True
                initial_resume = ""
        else:
            ui.log(f"[continuo] retomo la corrida caída: {run.dir}")
        ok = False
        try:
            run_once(args, run=run)
            done, ok = done + 1, True
        except KeyboardInterrupt:
            ui.log("\n[continuo] cortado a mano.")
            break
        except Exception:  # noqa: BLE001 - one bad topic must not end the loop
            failures += 1
            traceback.print_exc()
            ui.log(f"[continuo] ese artículo falló ({failures} en total); sigo.")
            # Stages are cached per file, so an article that died after research
            # is hours of work sitting on disk — the next cycle picks it up where
            # it stopped. Once only: a run that fails twice is broken, not
            # unlucky, and the next topic is unrelated to the one that broke.
            if not retry and run.dir != pipeline.OUTPUT:
                resumable = run
        # A dead provider with --every 0 would spin the loop hot against it: a
        # failed cycle always waits, a successful one honours --every as given.
        wait = args.cada * 60 if ok else max(args.cada * 60, 60)
        if (n == 0 or done + failures < n) and wait:
            ui.log(f"[continuo] espero {wait // 60} min hasta el próximo.")
            try:
                time.sleep(wait)
            except KeyboardInterrupt:
                ui.log("\n[continuo] cortado a mano.")
                break
    ui.log(f"\n[continuo] {done} publicados, {failures} fallidos.")
    return 1 if failures and not done else 0


def main() -> int:
    p = argparse.ArgumentParser(description="Social-science article writer.")
    p.add_argument("--fmt", default="medium", choices=list(FORMAT_NAMES),
                   help="length and genre of the text")
    p.add_argument("--mode", default="assisted", choices=["auto", "assisted"],
                   help="auto runs end to end; assisted asks at topic, outline, fixes and approval")
    p.add_argument("--topic", dest="tema", default="", metavar="TOPIC", help="topic or thematic hint")
    p.add_argument("--exact-topic", dest="tema_exacto", action="store_true",
                   help="use --topic as given, without proposing alternatives")
    p.add_argument("--no-library", dest="sin_biblioteca", action="store_true",
                   help="do not stop to ask for missing books")
    p.add_argument("--resume", nargs="?", const="latest", default="",
                   help="resume a previous run folder; bare --resume takes the latest one")
    p.add_argument("--rounds", type=int, default=2, help="PRO review rounds")
    p.add_argument("--detector-rounds", type=int, default=4,
                   help="rewrite rounds against the AI detectors")
    p.add_argument("--threshold", type=float, default=25.0,
                   help="highest tolerated AI score (0=human, 100=machine)")
    p.add_argument("--no-detector", action="store_true", help="skip the AI detectors")
    p.add_argument("--drafter", dest="borrador", default="", choices=["", "pro", "flash"],
                   help="who writes the sections. FLASH by default; 'pro' is the largest "
                        "measured gain in how human the text reads (worst judge 68 with "
                        "opus vs 88-93 with the hyper trio, same prompt and topic) and "
                        "also the most expensive")
    p.add_argument("--english", action="store_true",
                   help="write the article in English (detectors and tell lists use the "
                        "English corpus and guide)")
    p.add_argument("--setup", action="store_true",
                   help="set your interests, download the corpus and build the style guide")
    p.add_argument("--refresh-style", action="store_true", help="rebuild the style guide")
    p.add_argument("--detect", default="", help="only run the detectors over a file")
    p.add_argument("--publish", dest="publicar", default="no",
                   choices=["no", "draft", "auto", "live"],
                   help="upload: always as draft, auto (live only if approved and under "
                        "the detector threshold) or always live")
    p.add_argument("--provider", dest="proveedor", default="", choices=["", *llm.PROVIDERS],
                   help="main provider (default: AW_BACKEND or hyper)")
    p.add_argument("--backups", dest="respaldo", default="", metavar="A,B",
                   help="backup providers, in order; 'no' for none")
    p.add_argument("--pro", default="", metavar="MODEL",
                   help="PRO role model (topic, review, approval)")
    p.add_argument("--flash", default="", metavar="MODEL",
                   help="FLASH role model (outline, drafting, rewriting)")
    p.add_argument("--models", dest="modelos", default="", metavar="PROV:PRO/FLASH,...",
                   help="backup models, e.g. 'zen:glm-5.3-flash/deepseek-v4.1-flash,go:qwen3.8-flash'")
    p.add_argument("--wizard", action="store_true",
                   help="ask for provider, models and format even when other flags are given")
    p.add_argument("--continuous", dest="continuo", type=int, default=None, metavar="N",
                   help="write N articles in a row, picking topics itself; 0 = never stop")
    p.add_argument("--every", dest="cada", type=int, default=0, metavar="MIN",
                   help="minutes to pause between articles in continuous mode")
    args = p.parse_args()
    # Flags are English; the pipeline keeps its Spanish internal names.
    args.fmt = FORMAT_NAMES[args.fmt]
    args.mode = {"assisted": "asistido"}.get(args.mode, args.mode)
    args.publicar = {"draft": "borrador", "live": "vivo"}.get(args.publicar, args.publicar)
    if args.resume == "latest":
        args.resume = latest_run()
        if not args.resume:
            ui.log("[error] no previous run to resume in output/")
            return 1
        ui.log(f"[retomar] latest run: {args.resume}")
    pipeline.LANG = humanize.LANG = "en" if args.english else "es"
    if args.borrador:
        pipeline.DRAFT_ROLE = args.borrador
    # An attended run answers for a dead provider instead of crashing mid-article,
    # whatever flags launched it; --continuous has nobody watching, so it never prompts.
    llm.INTERACTIVE = sys.stdin.isatty() and args.continuo is None
    # Ctrl-C stops at once, even inside a minutes-long model call: every stage
    # is already on disk (atomic writes), so nothing needs flushing first.
    if str(llm.BOOK_WRITER) not in sys.path:
        sys.path.insert(0, str(llm.BOOK_WRITER))
    from ai_book_creator.env import exit_on_ctrl_c
    exit_on_ctrl_c(resume_hint, "")

    if args.proveedor or args.respaldo or args.pro or args.flash or args.modelos:
        head = args.proveedor or llm.CHAIN[0]
        if args.respaldo.strip().lower() in ("no", "ninguno"):
            backups = []
        elif args.respaldo:
            backups = [b.strip() for b in args.respaldo.split(",") if b.strip()]
        else:
            backups = [b for b in llm.CHAIN if b != head]
        llm.configure([head, *backups], args.pro, args.flash,
                      llm.parse_models(args.modelos))
    ui.log(f"[modelos] {llm.describe()}")

    if args.setup:
        edit_interests()
        import scrape_corpus
        scrape_corpus.main()
        print(style.build_guide(refresh=True)[:1200])
        print("\n" + ui.c(f"Guía completa en {style.GUIDE}", ui.GREEN))
        return 0

    if args.detect:
        text = pathlib.Path(args.detect).read_text(encoding="utf-8")
        print("\n" + ui.rule(f"detección de IA — {args.detect}"))
        # Score against the calibration the file is written in; a Spanish corpus
        # says nothing about an English text and vice versa.
        from build_corpus_en import is_english
        lang = "en" if (is_english(text) or args.english) else "es"
        local = humanize.local_score(text, lang=lang)
        score = local["score"]
        print("\nEstilometría local ({}): ".format(
            "inglés" if lang == "en" else "español")
              + ui.c(f"{score}/100", ui.BOLD, ui.GREEN if score < 35 else ui.RED))
        for i in local["issues"]:
            ui.log(f"  · {i}")
        ui.log("[detector] jueces LLM:")
        humanize.llm_judges(text, lang=lang)
        ui.log("[detector] detectores externos:")
        humanize.external_detectors(text, lang=lang)
        return 0

    # Someone at the terminal picks the models every run, --resume and --continuous
    # included (asked once, before the loop starts); model flags skip the question.
    model_flags = args.proveedor or args.respaldo or args.pro or args.flash or args.modelos
    if sys.stdin.isatty() and not model_flags and not (args.wizard or len(sys.argv) == 1):
        pick_models()

    if args.refresh_style:
        style.build_guide(refresh=True, lang=pipeline.LANG)

    if pipeline.LANG == "es" and not style.GUIDE.exists():
        ui.log("[error] falta la guía de estilo. Corré primero: python main.py --setup")
        return 1

    if args.wizard or len(sys.argv) == 1:
        args = wizard(args)

    if args.publicar != "no" and not publish.configured():
        ui.log("[error] falta la configuración del destino de publicación: "
               "DEVTO_API_KEY (AW_PUBLISH_TARGET=devto) o AW_WP_URL, AW_WP_USER, "
               "AW_WP_APP_PASSWORD en .env. Corré sin --publish o completala.")
        return 1

    ui.COMPACT = args.mode == "auto" or args.continuo is not None

    if args.continuo is not None:
        # Nothing is watching: every interactive gate has to be off, or the loop
        # blocks forever on an input() nobody will answer.
        args.mode, args.sin_biblioteca, args.tema = "auto", True, ""
        return loop(args)

    run_once(args, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
