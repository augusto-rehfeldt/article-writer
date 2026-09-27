# article-writer

## What This Is
Automated writer of social-science articles in the voice of Augusto Germán Rehfeldt
(Revista La Cueva). Goes from topic selection to reviewed, cited, AI-detector-tested
final text. Everything is written in Spanish (rioplatense); the code and comments are
in English.

## Non-Negotiables
- **Never invent a citation.** The drafting model may only cite keys present in the
  dossier (`output/<run>/02_dossier.json`). `research.verify_citations()` re-checks
  every `(Autor, año)` in the text after drafting *and* after humanizing; unbacked
  citations are sent back for correction.
- **Citations are APA 7 in Spanish.** In-text via `pipeline.APA_RULES` (injected into
  every drafting and revision prompt), reference list via `Source.citation()`. The
  author's older `(Kurz, 1998: s/n)` form is still accepted by `CITE_RE` because a
  humanizing rewrite sometimes reintroduces it. Verification covers **both** the
  parenthetical form (`CITE_RE`) and the narrative one (`NARRATIVE_RE`, "Postone (1993)
  sostiene"); dropping either leaves half the citations unchecked.
- `.env` holds the API key and is gitignored. Never commit it.
- `corpus/`, `library/` and `output/` are gitignored — corpus is scraped, library is
  the user's books, output is generated.
- Model routing: **PRO** = `qwen3.8-max` (topic, outline audit, review, **applying the
  review's corrections** — `pipeline.revise()` —, final approval),
  **FLASH** = `deepseek-v4-pro-0813` (queries, outlining, drafting, humanizing rewrites).
  `revise()` moved to PRO on 2026-08-23: FLASH answered a report about invented
  citations by inventing more, and shipped «Now the final section text:» as prose. Since
  2026-09-26 the detector judges are a fixed panel, `llm.JUDGE_MODELS` =
  `claude-opus-5-5` (on `claude`) + `gpt-6-astra` (on `oauth`); `llm._serves` sends a
  model owned by an exclusive provider only there. A failing judge is skipped; if every
  judge fails, `humanize.llm_judges` judges with the drafting model (FLASH, or PRO under
  `DRAFT_ROLE=pro`). An explicit `models=` list (the benches) never gets that stand-in.
  kimi-k3 is out of both this panel and `bench2.JUDGES`.
- Providers: `claude`, `hyper`, `zen` (OpenCode Zen, OpenAI-compatible at
  `https://opencode.ai/zen/v1`, key in `OPENCODE_API_KEY`), `grok` (xAI directo,
  OpenAI-compatible at `https://api.x.ai/v1`, key in `XAI_API_KEY`, URL
  override `AW_GROK_URL`), `oauth` (openai-oauth
  local proxy at `127.0.0.1:10531`, uses your ChatGPT subscription), `g4f`
  (gpt4free's local `g4f api` server at `127.0.0.1:1337`, URL override `AW_G4F_URL`;
  keyless, or `G4F_API_KEY` only, because g4f hands any other bearer to its backends)
  and the local `opencode` CLI. In any attended run (stdin is a tty, wizard or flags alike),
  exhausting the whole chain prompts the user to wait 30s, change provider/model,
  or abort (`llm.INTERACTIVE`, set in `main.main()`, main thread only); `--continuous`
  forces it off because nothing is watching. The change sticks for the rest of the
  run, and the retry chases the new PRO/FLASH instead of resending the old model
  name (which an exclusive head would not serve, silently skipping it);
- **Every completion runs on book writer's AIService** (the workspace's one AI suite,
  sibling `book writer` folder or `AW_BOOK_WRITER`). `llm.py` keeps what is this
  project's own — the chain, PRO/FLASH roles and their per-provider translation,
  catalogues, key discovery, heartbeat, interactive recovery, JSON repair — and sends
  each link through `llm.shared_service()`/`llm._send()`, the same port lamplight's
  engine runs. Links map to book writer providers (`llm.SHARED`): `claude` → the
  Claude Code CLI, `hyper`/`grok` → their configs, `zen` → `opencode-zen`, `go` →
  `opencode-go`, `oauth` → `openai-oauth` (book writer starts the proxy). OpenAI-
  compatible links stream (`stream: true`, so the timeout is per chunk; zen answered 503
  at 207s on a completion the stream finished in 341s) and send `max_tokens`. There is
  **no `cap_is_ceiling`**: AIService asks for the model's whole output allowance from
  book writer's config, and a truncated reply is retried with a doubled budget, never
  returned. A usage limit makes a link fail so the chain walks on; only the last link of
  an unattended chain (`not llm.INTERACTIVE`) waits the limit out. Tests replace
  `shared_service` with fakes (`_shared` in the self-check); no SDK client, CLI call or
  raw request lives here. Gone with the port (2026-09-25): the `redactor` opencode agent,
  the `OPENCODE_SIBLING` empty-run retry, `AW_OPENCODE_TIMEOUT`, and the per-status
  backoff (`_status`/`_backoff`); retries and transports are book writer's now.
- **`AW_BACKEND` is an ordered chain**, not one name: `claude,hyper,go` = main
  provider first, backups after, in order. `llm.PROVIDERS` holds each one's label,
  catalogue and role defaults; `llm.chat()` walks the chain and `llm._as()` translates
  the role into each provider's own names (PRO is `opus` on claude, `qwen3.8-max` on
  hyper, and go unsuffixes `deepseek-v4-pro-0813`). Sending one provider's model
  name to another is a 404, which is why the translation is not optional.
  Pick it in the wizard (first question), with `--provider/--backups/--pro/--flash`,
  or in `.env`. `llm.configure()` is the single entry point — it also recomputes
  `JUDGES`.
- **Every attended run asks for the models** (`pick_models`), `--resume` and
  `--continuous` included (once, before the loop). Only a non-tty stdin or a model
  flag (`--provider`, `--backups`, `--pro`, `--flash`, `--models`) skips it.
- **The wizard's model list is live.** `llm.catalogue(backend)` merges the built-in
  list with whatever the provider answers right now: OpenAI-compatible `/models`
  for hyper and zen, `opencode models` for the CLI. Built-ins first, live additions
  after, deduped, marked « nuevo» in the wizard; a dead catalogue degrades to the
  built-in list. When the live list answers it also prunes: built-ins the provider dropped disappear from the menu instead of 401-ing as defaults (zen removed qwen3.7-max server-side while the spec still defaulted to it). Zen's `/models` also lists gpt/gemini/claude ids that only speak
  `/responses` and `/messages`, not chat completions — `skip_live` filters them out.
- **The local `opencode` CLI route is called `go`** and is pinned to opencode-go
  (hyper-spelled ids, defaults PRO=`qwen3.8-max` FLASH=`deepseek-v4-pro-0813`):
  no variant question anywhere — zen-via-CLI is gone because `zen` is already a
  first-class provider. `OPENCODE_PROVIDER` is the constant `opencode-go`. Old
  `.env` files that still say `opencode` are canonicalized to `go` by `llm._canon()`
  (`_default_chain`, `configure`, `parse_models`), so a stored chain never loses its
  last link; `AW_OPENCODE_VARIANT` and `AW_OPENCODE_PROVIDER` no longer do anything.
- **A rewrite that comes back under 80% of its block's words is discarded**
  (`humanize._rewrite`): losing a fifth of the words means dropped arguments or
  citations, not tighter prose. Citations are re-verified later anyway, but the
  original block is kept rather than shipping the amputation.
- **The rewriter's own quotas manufacture AI tells if left unbounded** (measured
  2026-08-23). The short-sentence quota produced «La pregunta no es tecnológica.»
  verdict paragraphs at 10x the corpus max (author: one hit in 54 articles; our
  humanized draft: five in 4.2k words), and the rewrite doubled the «no X sino Y»
  antithesis density (2.8/1k vs his ~1.2/1k). Now checked and priced:
  `STRUCTURAL_TELLS` flags short negation-verdict paragraphs (ceiling 0.15/1k,
  measured above every corpus article), `local_score` charges antithesis density
  past 2.5/1k, eighteen corpus-absent stock phrases joined `LLM_TELLS`
  («cobra sentido», «pone sobre la mesa», «surge la pregunta», …), and both
  `REWRITE_PROMPT` (max one antithesis per block, short sentences must carry a
  concrete datum) and `JUDGE_PROMPT` name the pattern. The default detector
  threshold dropped from 35 to 25 (`humanize.humanize`, `pipeline.run_pipeline`,
  `main --threshold`; README updated). Corpus scores are unchanged (14–49), so
  none of it prices his real tics.
- **No third-party humanizer beats ours for Spanish** (surveyed 2026-08-23,
  re-measured against real prose 2026-08-24 — full numbers in `bench2.md`,
  harness `bench2.py`, raw data `bench/scores.json`). Every pip humanizer is
  English-only, cosmetic, or dead: `texthumanize` 0.35 returned the English
  input **byte for byte** and changed 2% of the Spanish one (its
  `syntax_rewriting` stage only speaks en/ru/uk/de, so «25 languages» is the
  cosmetic stages); `humano` 2.0.1 injects English filler into Spanish
  («…no tenía casi nadie. **Here's the deal -** dos operarios…») and scores
  worse than the raw draft on every judge in both languages;
  `ai-text-humanizer` does not build. lynote's translation chain rewrites
  through Finnish and drops Spanish articles («visité fábrica textil»); their
  «standard pipeline» (chain + rewrite at temp 1.3) is the only method that
  beats ours anywhere — 72 vs 93 worst-judge in **English** — and it gets there
  by discarding the text (0.02 character similarity), which would orphan every
  dossier-grounded citation. **Ours wins Spanish** (78 worst-judge vs 82/93/99).
  Re-test only if a Spanish detector or humanizer actually ships.
- **Re-measured 2026-09-27 (`bench_en.py`, table in `bench2.md`): opus 5.5 drafts
  English ~20 points less machine-like than gpt-6-luna/sol/astra** (worst
  other-family judge 60-67 vs 90-94; the three gpt-6 models are indistinguishable),
  and one `_rewrite` pass by the drafter no longer hurts (opus 80→72, gpt-6-luna
  94→90, one sample each). The note below is the 2026-08-24 state. Human windows
  still read 1-3; nothing generated reads under 60.
- **`humanize._rewrite` makes English text *worse* to the judges** (measured
  2026-08-24 with two different rewriters, so it is not one model's quirk).
  Under `LANG="en"` it drives the local score to 3 with sonnet and 2.2 with
  deepseek — the two best numbers in the bench — while pushing the worst judge
  from 78 to **95** and **82**. It is optimising the stylometric floor and
  manufacturing tells doing it: the same failure the Spanish quotas were capped
  for on 2026-08-23, except `REWRITE_PROMPT_EN` never got that pass. The English
  quotas need the same measure-and-cap treatment before `--english` runs the
  rewrite loop for real. **Which model rewrites is itself a variable and it
  flips by language:** in Spanish sonnet beats deepseek (78 vs 82), in English
  deepseek damages far less than sonnet (82 vs 95).
- **Not one downloadable classifier is usable, in either language** (seven
  tested 2026-08-24 over 12 real human windows and 28 generated texts).
  `desklib/ai-text-detector-v1.01` leads the RAID benchmark and still reads one
  of the author's own published articles at 87% machine while reading
  `raw_sonnet_es` at 0.4%; `desklib/…-academic-v1.01` is worse (90% on a real
  arXiv paper). `Oxidane/tmr` is anti-correlated. `yaya36095/xlm-roberta-text-
  detector` emits a constant 100, and so do both AuTexTification (IberLEF 2023)
  fine-tunes — 5.5 and 99.8 on everything, because their generators were
  BLOOM-era. **There is no Spanish AI detector.**
  `Hello-SimpleAI/chatgpt-detector-roberta`, the one `humanize._hf_detector()`
  loads in English mode, answers 0.0 on 26 of the 28 generated texts: it was
  trained on HC3 question-answer pairs, fires only on blatant assistant
  register (`raw_gpt-5.6-terra_en`, 91.7) and has zero false positives across
  the twelve human windows. Keep it — it costs nothing and never lies — but it
  is not a signal.
- **A quota notice from `claude -p` is not a completion.** It arrives on *stdout*
  with exit code 0 — «You've hit your session limit · resets 9:10am» — so
  the old `claude_chat` returned it as prose: it got drafted into sections, scored by the
  judges and handed to `chat_json`, where the run died on `Could not parse JSON`.
  `llm._quota_notice()` makes `_send` raise instead, and the chain walks on to the next
  provider. The <400-character guard is what keeps an article that discusses
  limits from being thrown away; a quota notice is one line, never a section.
- **Each backup answers with its own models**, not just its built-in defaults.
  `llm.set_models(backend, pro, flash)` writes the pair into `PROVIDERS[backend]`,
  which is exactly where `_as()` reads it — so overriding it is the whole
  mechanism. Set it with `--models "zen:glm-5.2/deepseek-v4-pro,go:qwen3.8-max"`
  (FLASH optional), with `AW_MODELS` in `.env` (same syntax, parsed at import), or
  in the wizard, which asks per backup only if you say yes — six extra prompts for
  a chain that usually never runs is why it is opt-in. `describe()` prints them.
- `claude` is the Claude Code CLI in print mode (`claude -p`), running on the user's
  subscription instead of a metered key. It is marked `exclusive`, so it only ever answers for models in its own catalogue.
  Since the judges are PRO/FLASH themselves (2026-08-22), they ride the head provider
  like any other call; the exclusive flag only keeps foreign model names out.
- **Tone: a comment, not a fight.** The text takes a position and argues it with data.
  It does not descend to «no estoy de acuerdo con X», «X se queda en la superficie»,
  «desconfío de X», irony at third parties, or frases de efecto. At most one sober
  disagreement per section, aimed at an argument and never at the person who signs it.
  Encoded in `pipeline.SECTION_PROMPT` (TAKE A POSITION, NOT A FIGHT), checked by
  `REVIEW_PROMPT`, and no longer injected per block by `humanize.REWRITE_PROMPT` — the
  old per-block quota is what produced one objection every 700 words.

- **English is the default language since 2026-09-27** (`--spanish` for the author's
  voice; `--english` still accepted). `pipeline.LANG`/`humanize.LANG` keep `"es"` as
  module default for the tests; `main.main()` sets `args.lang or "en"`.
- **Prompts and console output are English since 2026-09-27**, in both languages.
  `pipeline._lang()` appends the output-language directive to every article prompt
  (Rioplatense Spanish + `y`/`s/f` APA conventions for `es`, `&`/`n.d.` for `en`);
  JSON field names and enum values (`titulo`, `veredicto: aprobado`, `gravedad: alta`,
  `señales`) and `output/` file names stay Spanish so old runs resume. Stage tags are
  `[topic] [research] [bibliography] [outline] [draft] [review] [detector] [approval]
  [publish] [continuous] [resume]` (`ui.TAGS`); `ui._chatter` keys on «reusing» and
  «cached». Still Spanish, reached only under `--spanish`: `humanize.JUDGE_PROMPT`,
  `humanize.REWRITE_PROMPT`, `style.BUILD_PROMPT`, and the `LLM_TELLS` data.
- **`--english` is calibrated against `corpus_en/`, not against the author.**
  `pipeline.LANG`/`humanize.LANG` switch to `"en"`: every article-producing prompt
  gets an override directive appended (`pipeline._lang()`), `humanize` swaps to
  `ENGLISH_TELLS`, `JUDGE_PROMPT_EN`, and an English rewrite prompt without the
  Spanish-only quotas (enclisis, guillemets), plus an extra verdict-paragraph tell.
  The stylometric floors come from `style_fingerprint_en.json`, built by
  **`build_corpus_en.py --guide`** over human-written English prose: recent arXiv
  papers in society-facing categories, Gutenberg social-theory classics (boilerplate
  stripped), and English works already in `library/`, chunked into ~3.500-word files
  so `corpus_ranges(lang="en")` measures like-for-like. `_corpus_allowed_en()`
  whitelists tells that corpus actually uses — real academics write some on purpose.
  If `transformers` is installed, `humanize._hf_detector()` adds a locally-run
  trained classifier (`Hello-SimpleAI/chatgpt-detector-roberta`, ~500 MB once,
  English-only) as a fourth signal family; it reports its worst 512-token window
  like the LLM judges do and degrades silently when the package or model is absent.
  Without corpus_en/ everything still runs on built-in floors (`style.corpus_
  fingerprint` returns empty ranges instead of dying).
  **State (2026-08-23):** `corpus_en/` built — 113 chunks (~731k words): 20 arXiv
  papers in society-facing categories, 5 Gutenberg social-theory classics
  (`_covers`-verified; gutendex matches loosely, two titles returned same-topic
  stand-ins), 24 English works from `library/`. `style_fingerprint_en.json`
  measured (563 windows of 1.200 words; burstiness p25 0.54). At that window size
  the corpus itself scores local 3–58 (median ~24 — repeated-phrase noise on
  academic terminology, same expected noise as his didactic articles) while AI
  English scores >90: the floor separates. **The EN style guide
  (`style_guide_en.md`) was built 2026-08-24** by opus over the whole corpus:
  1.834 words, every trait quoted literally, and it earns its place — with the
  compact block alone (samples plus the counted quotas) every model drafts
  *worse* English than with no style instruction at all (opus 78→88 worst-judge,
  sonnet 78→91), and the full block with the guide puts opus back at 78 with a
  local score of 0. Rebuild with `python build_corpus_en.py --guide`. What no calibration fixes:
  there is no single author's voice behind the English corpus, so the text reads
  like competent generic academic English, not like anyone in particular.

## Layout
| File | Role |
|---|---|
| `main.py` | CLI: wizard, flags, `--setup`, `--detect`, `--resume`, `--continuous`, `--publish` |
| `pipeline.py` | Stage machine: topic → research → outline → draft → humanize → review → approval |
| `ui.py` | ANSI colouring of the console output (`ui.log` colours `[etapa]` tags; off when stdout is not a tty or `NO_COLOR` is set) |
| `publish.py` | Self-hosted WordPress upload (`/wp-json/wp/v2/posts`, application password) |
| `research.py` | All source connectors + dossier + citation verification |
| `style.py` | Corpus stylometry (fingerprint) + PRO-written style guide |
| `humanize.py` | Local stylometric scorer, cross-family LLM judges, external detector APIs, rewrite loop |
| `scrape_corpus.py` | Downloads the 16 reference articles via the WordPress.com API |
| `build_corpus_en.py` | Builds `corpus_en/` (arXiv + Gutenberg + `library/`) and, with `--guide`, the EN fingerprint and style guide |
| `ingest_facultad.py` | Filters `D:/facultad` down to the author's own writing → `corpus_facultad/` |
| `bench2.py` | Humanizer / detector / drafting-model bench, ES+EN, against real human windows (`gen`→`hum`→`score`→`report`; results in `bench2.md`, raw data `bench/`) |

## Commands
```bash
python main.py --setup                      # corpus + style guide (run once)
python main.py                              # interactive wizard
python main.py --fmt medium --mode auto      # fully automated
python main.py --topic "..." --fmt long     # user-fixed topic, assisted
python main.py --resume                     # resume the latest run (stages are cached)
python main.py --resume output/2026...-slug # resume a given run
python main.py --detect archivo.md          # only run the AI detectors
python main.py --continuous 0 --every 90 --publish auto   # never stops, picks its own topics
python main.py --provider claude --backups hyper,go --pro opus --flash sonnet
python main.py --wizard --english             # wizard aunque haya banderas (elegir modelos)
python test_article_writer.py               # self-check, no network, no API
```

## Gotchas
- **`FORMATS[fmt]["words"]` is a `(mínimo, máximo)` tuple, not a number.** Prompts get
  the range through `pipeline.wrange()`; internal budgeting (context size, fulltext
  budget) uses `pipeline.wmid()`. A single target made the model pad to hit it.
- **`FORMATS[fmt]["sections"]` is a `(mínimo, máximo)` tuple too**, and the model
  decides how many units and how long each one is (500 a 10.000 palabras, explicitly
  uneven). A fixed *n* produced ten interchangeable blocks of identical length; a
  chapter is sometimes a whole part and sometimes a two-page interlude.
- **A drafted chunk is asked for at ±10% but kept at 70–150%.** The first reply
  outside ±10% gets one retry; after that the right-language reply closest to the
  budget is cached if it lands in 70–150%. A plainly amputated or wrong-language
  chunk is retried `DRAFT_TRIES` times on the head link, then handed to each backup
  in the chain (`pipeline._draft_chunk`); it raises only when every link failed.
  Two replies at 1.2x used to end an hours-long run, and a free-tier model
  answering 12 words twice ended another (2026-09-25).
  `test_a_section_near_its_budget_is_kept_not_fatal` is the guard.
- **Citation matching is accent- and case-blind** (`research._fold`, which also
  transliterates ł/ø/ß/đ/æ/œ): «(Zizek, 2009)» resolves to the key «Žižek, 2009»
  instead of being flagged as invented.
- **The humanized rewrite is judged only on what it broke.** It is refused when it
  drops a verified citation or adds an unbacked one the draft did not already carry;
  pre-existing unbacked citations go to the correction step. When it is refused, the
  kept draft's round-1 score is reused, so `07_detector_humanizado.json` always
  hashes to `04_draft_humanizado.md` and a resume does not humanize again.
- **`pipeline.MAX_CALL_WORDS` (2.500) is the ceiling of one drafting call**, not of a
  section. `_chunks()` splits any longer section subsection by subsection, and a
  subsection still over it into «parte j de n», one call and one cache file each
  (`04_sec02_01a.md`); the heading is emitted once, on the first part. Asking FLASH for
  8.000 words in one call returns two or three thousand and a section that stops
  mid-argument, which is why `tesis`/`libro` used to land at a third of their budget.
  Inside a section the rolling synopsis is one entry behind, so each chunk also gets
  the previous chunk's last 120 words to splice onto.
- **`chapters: True` formats outline in two passes** (`_plan_outline`): `PLAN_PROMPT`
  for the architecture (title, function, word budget, optional `parte`), then
  `DETAIL_PROMPT` in batches of 6, each batch seeing the whole architecture. One call
  cannot plan 40 units: the JSON runs past the output limit and comes back truncated.
  `03_plan.json` caches the architecture. For the same reason PRO's outline audit only
  replaces the outline if `esquema_corregido` still carries every section.
- **Each format carries its own `register`**, injected into the outline prompt, the
  section prompt, the review prompt and the rewrite prompt, and declared there as
  outranking the generic style rules. A `paper` and an `artículo breve de intervención`
  are not the same voice, and before this they got identical instructions.
- **Every format ends in `## Referencias`.** `assemble()` gates the abstract and the
  keywords on `FORMATS[fmt]["apparatus"]`, but not the reference list: `corto` is the
  only format without apparatus and it still cites (`min_citas` applies to every
  format), so it used to print «(Kurz, 1998)» and no way to reach the book. The list is
  omitted only when `research.bibliography()` comes back empty.
  `test_every_format_prints_its_reference_list` is the guard.
- **Page numbers come from `[p. N]` markers, not from the model.** `research.pdf_text()`
  writes one before each non-empty PDF page, so the dossier text a section prompt sees
  carries them and `APA_RULES` tells the model to cite the nearest preceding marker for
  a verbatim quote — `(Kurz, 1998, p. 302)`. Only PDFs have pages: ePub, HTML and plain
  text stay pageless, and «sin página no inventes una» still governs them. `N` is the
  PDF's own index, so front matter can shift it by a few. `pipeline._strip_preamble()`
  deletes any marker the model copies into the draft; the parenthetical APA form is not
  bracketed and survives. An empty page gets no marker on purpose — a trailing blank
  page in a scan would leave the text ending on «[p. 312]», which `research._cut()`
  reads as a work that stops mid-sentence, and the whole book would be held as a
  fragment. `test_pdf_pages_are_marked_and_never_published` is the guard.
- **Every study, index, report or figure must be cited where it is mentioned**
  (`APA_RULES`). Naming a consultancy or an institution that is not in the dossier is
  forbidden, and so are «un estudio reciente», «según informes», «se estima que».
- `humanize.repeated_phrases()` finds verbatim 6-grams the text serves twice, merging
  overlapping windows so a repeated sentence reports once instead of a dozen times. It
  feeds three places: `local_score` (so the rewriter hears about it), `REVIEW_PROMPT`,
  and `pipeline._banned()`, which lists the short sentences already used so the *next*
  section cannot serve them again. Sections are drafted independently — without this the
  same remate comes back three sections later word for word. Topic terminology repeats
  legitimately; both prompts say so.
- Search engines block keyless HTTP scraping. Site archives are queried through
  native endpoints instead (`/wp-json/wp/v2/search`, libcom's Drupal search, the
  WordPress.com API). Do not reintroduce a **requests-based** `web_search`: measured
  on 2026-08-22, `html.duckduckgo.com` answers a 202 challenge page after a handful of
  queries (and stays blocked for well over a minute), the public SearXNG instances sit
  behind Anubis JS proof-of-work, and Bing and searx-via-Bing answer a detected scraper
  with results for a *different* query — a search for Boulding came back with Finnish
  office rentals, which is worse than an empty list because it looks like data.
- **The one search route that works is `research.browser_search()`**: duckduckgo.com's
  own Javascript site driven by headless Chrome through Selenium. It passes where the
  HTTP endpoints fail, including while the same IP is 202-blocked. One driver for the
  whole run (`_browser()`, ~3s to start, ~8s per search), serialised on `_driver_lock`
  because `missing_books` is threaded, closed at exit. No Selenium or no Chrome
  installed returns `[]` instead of raising — the route degrades, the run does not.
- arXiv, GDELT and Semantic Scholar rate-limit hard. `research._get()` throttles per
  host via `_HOST_DELAY`; lowering those delays empties the dossier.
- GDELT needs multi-word queries wrapped in double quotes or it returns an error page
  instead of JSON.
- **The heartbeat has no tick limit.** It used to stop at ten ticks (600s) while
  a CLI link runs up to 900–1800s (book writer's config timeouts), so the longest
  calls went silent exactly when the wait started looking like a hang. An opencode call is slow for structural reasons: one
  fresh `opencode run` process per completion (CLI boot, plugin load, provider
  registration), no server reuse, and nothing streams.
- **`--continuous` retries a crashed run once, from its cache.** `loop()` owns the
  `Run` and hands it to `run_once`; if the article dies after the topic stage
  (`run.dir != pipeline.OUTPUT`) the next cycle resumes that directory instead of
  starting a new topic — research alone is hours of work and every stage is on
  disk. Only once: a run that fails twice is broken, not unlucky. A failed cycle
  also always waits at least 60s, or `--every 0` against a dead provider spins the
  loop hot.
- **`revise()` applies patches, not a rewrite.** PRO answers `{"parches": [{"buscar",
  "reemplazar"}]}` and `_apply_patch()` swaps each fragment in place: `buscar` must
  occur exactly once, and if the retyped copy does not match literally it is matched
  again with `\s+` between its words (the model turns a line break into a space). The
  citation guard is per patch — a patch that loses a verified citation or introduces
  an unbacked one is dropped and the rest still land. Whole-document revision cost a
  4.000-word PRO completion (210s) and was thrown away entirely when one citation
  slipped («descartada: bajó de 14 a 13 citas válidas»).
- Stage outputs are cached per file in `output/<run>/`. Delete a stage file to force
  that stage to re-run.
- **The rolling synopsis is only built for a section that is still going to be
  written.** `draft()` precomputes which `04_secNN.md` are missing (`pending`) and
  skips `_update_synopsis` when nothing ahead needs it: a `--resume` on a finished
  draft used to pay one FLASH call per section for a summary no one would read,
  and every run paid one more after the last section. `test_a_fully_cached_draft_
  costs_nothing_to_resume` is the guard.
- Research is checkpointed *inside* the stage: `02_estado.json` lists which substeps
  (`fuentes`, `textos`, `libros`) finished, and `02_dossier.json` is written after each
  one, so `--resume` after a Ctrl-C does not re-gather or re-download. Since the
  dossier is now saved half-built, its mere existence no longer means the stage
  finished — `02_estado.json` is what says so. Books already in `library/` are read off
  disk by `fetch_book()` before any network call.
- `research.gather()` and `enrich_fulltext()` are threaded. Throttling lives in
  `_get()` behind a per-host lock — do not bypass `_get()` with a bare `requests.get`,
  or the pool will race past the rate limits and empty the dossier.
- `style.articles()` reads both `corpus/` and `corpus_facultad/`. After running
  `ingest_facultad.py`, the guide and fingerprint are stale until `--refresh-style`.
- `research.missing_books()` **downloads** the works the catalogues only knew by title:
  archive.org first (free OCR full text, fails on lending-restricted items), Library
  Genesis second (`libgen.li`/`libgen.vg`; `libgen.is/.rs/.st` are dead). Everything that
  lands is written into `library/`, so the next run reads it off disk. **Anna's Archive
  is a hand-download link in `_fallback_links()` and nothing else** — the scraper was
  deleted 2026-08-23: Cloudflare blocks every keyless client, so its three hosts spent
  45s each timing out per gap and then returned the same link list the fallbacks
  already build. A gap is `{"wanted", "fallbacks"}`; there is no `candidates` key.
- **Not every "obra necesaria" is a book.** When both catalogues fail, `fetch_book()`
  falls through to `research.fetch_paper()`: Crossref/arXiv by title → DOI → unpaywall
  → Sci-Hub, reusing `_retrieve()`. archive.org and Library Genesis index books, so a
  journal article («Smith, "Estimation of a genetically viable population…", *Acta
  Astronautica*, 2014») is unreachable there no matter how many mirrors are tried.
  Two details make it work and are not optional: `_split_wanted()` reads a «quoted»
  segment as the title (otherwise the first comma splits an author *list*, and the
  query becomes two names plus a journal), and the search string is
  `" ".join(sorted(_words(title)))` — arXiv's API ANDs the whole string, so "for a …
  towards … b" pushes the paper off its own result list. Reports (Project Hyperion's
  *Chrysalis*) and pre-digital essays (Boulding, 1966) have no DOI and no catalogue
  record, so they used to stay gaps — `research.web_fulltext()` is what closes them.
- **`fetch_book()` ends on the open web, on purpose.** Every catalogue above it matches
  on metadata and so needs the work's *real* title, and the model that writes the
  reading list gives a translated or approximate one («Chrysalis: informe de diseño»,
  «HERITAGE simulation generation ships»). A search engine tolerates that; a catalogue
  does not. Two details are load-bearing: `_covers()` runs against the **downloaded
  text** (`title + body[:4000]`), not the search snippet — a paper's first page carries
  its real title and author list, which is what an approximate wanted string has to be
  checked against — and the floor is **6.000 characters, not 3.000**: at that point the
  candidate is supposed to be a whole work, and anything shorter is a landing page or
  an abstract. PDFs are asked for first and sorted first for the same reason, and
  `arxiv.org/abs/X` is rewritten to `/pdf/X` before downloading. Junk hosts
  (`_JUNK_HOST`: Scribd, ResearchGate, Academia.edu, Studocu…) are dropped unfetched.
  Measured 2026-08-22: the four works that had stayed gaps all landed (Marin y Beluffi
  via arXiv PDF, Kurz via archive.org, *Chrysalis* via i4is.org, Boulding via gwern).
- **`research._titlepage()` is what keeps a web hit from becoming an invented
  citation.** Library Genesis and the open web both identify the work by the string
  the *reading list* wrote, and that string is a model's guess — so the download used
  to be filed, and cited, under a title nobody published. The title page is read by
  FLASH (`_TITLEPAGE_PROMPT`, first 3.000 characters) because parsing it does not
  work: a PDF's first line is the author or a journal header, the title itself wraps
  over three lines, and arXiv PDFs carry no metadata title. Everything the model
  returns is checked back against `body[:4000]` — a title whose words are not on the
  opening page, an author whose surname is not there, a year or a publisher the
  document never prints are all dropped and the guess stands. It can only *correct*,
  never invent past the text. Authors come back name-first ("Kenneth E. Boulding")
  because `_apa_name()` treats a comma as "already APA" and would print the given
  name in full; the prompt also asks it to undo OCR breakage («Fr ́ ed ́ eric») and
  cover-page ALL CAPS, or the reference list carries both. Measured on the four:
  «Chrysalis: informe de diseño» → Davies, J. I. (2025), *Project Hyperion
  Competition Results*, Principium; «HERITAGE simulation generation ships» → Marin,
  F. y Beluffi, C. (2018), JBIS. The file in `library/` is named from the corrected
  metadata too, so `scan_library()` reads back the real title on the next run.
- **The user's Calibre library is consulted by title, never scanned.**
  `research.calibre_book()` reads `metadata.db` read-only (`AW_CALIBRE`, default
  `~/Calibre Library`), matches with the same `_covers()` gate as any catalogue, and
  runs *before* archive.org in `fetch_book()`. AZW3/MOBI go through Calibre's own
  `ebook-convert` (`research.ebook_convert()`), and the extracted text is cached into
  `library/` so the next run skips the conversion. It is **not** wired into
  `scan_library()` on purpose: that function puts everything it finds in the dossier,
  and a general-purpose ebook library is mostly fiction.
- **The word count in the bibliography log is extracted text, not book length.**
  Downloading full texts is uncapped everywhere: `fetch_text`, `pdf_text`,
  `read_local`, `archive_org_book`, `web_fulltext` and the dossier sources all keep
  the whole work. The only cut is the per-call one in `Source.brief()` when the
  drafting context is built. There is no «recortado» flag anymore — `_partial()`
  appends «cortado a la mitad» when the text stops mid-sentence or «¿extracto?»
  when a work called a book arrives under 60k characters, and only ever warns:
  Boulding's 1966 essay is genuinely 35k characters, so rejecting on length would
  throw away a whole work.
- **What goes to `library/` is never capped, and neither is the dossier.**
  `read_local(path, 0)` reads the whole file by default, and every route that caches
  a work (`calibre_book`, the libgen branch, `fetch_book`'s disk branch) uses it — a
  capped write stores a mutilated copy and every later run inherits the cut with no
  way of knowing. **There is no slicing left at all.** `_slice_for()` existed to cut
  an anthology around the wanted title — «Paradises Lost» closes *The Found and the
  Lost*, so a plain `body[:cap]` kept 55.000 words of the other novellas and not one
  line of the one that was asked for — but every caller had already moved to `cap=0`,
  so it returned the body untouched on every run and was deleted (2026-08-23). Store
  the whole text and let `Source.brief()` slice per drafting call; if a cap ever comes
  back it must cut around the title, never from the front.
  `test_a_work_enters_the_dossier_whole` is the guard.
- **`web_fulltext()` takes the longest candidate, not the first past the floor.**
  A publisher's sample and the whole book both clear the 6.000-character floor and
  both carry the real title page, so first-past-the-post filed one chapter of Kurz's
  212 pages as the book. Bounded at six downloads, with an early exit at 150k
  characters — a hit that size is not a sample.
- Every catalogue hit is checked with `research._covers()` (≥60% word overlap, `ceil`)
  before being accepted. Without it archive.org answers "Kurz, Geld ohne Wert" with a
  declassified CIA cable, and with `round` instead of `ceil` two different books by the
  same author collapse into one gap.
- **`research._work_info()` answers two questions in one FLASH call**, both of which
  change what `fetch_book()` accepts. It is `lru_cache`d and called once per gap.
  - *Other titles.* The reading list is written in Spanish and the copy that exists is
    the German original — «El colapso de la modernización» shares no word with *Der
    Kollaps der Modernisierung*. `fetch_book()` runs its **whole chain once per title**
    (disk, Calibre, archive.org, Library Genesis, `fetch_paper`, `web_fulltext`), and
    each variant is still matched by `_covers` at full strength: a bad translation
    matches nothing instead of matching anything. Loosening the threshold until the two
    Kurz titles matched is precisely what makes *Geld ohne Wert* stand in for *Der
    Kollaps*. The Source is filed under the title of the edition actually read.
    `_have_title()` stays monolingual — it runs per gap check and would turn one call
    into dozens.
  - *Book or short text.* This sets the length floor, and it cannot be guessed from
    the text: Boulding's 1966 essay is 35.000 characters and complete, «El colapso de
    la modernización» at 31.000 is chapter one of 212 pages. Nothing in the download
    tells them apart.
- **`missing_books(budget=…)` is a ceiling on how many works are even attempted**, and
  a work past it is reported as «falta el texto completo» exactly like one the whole
  chain hunted and failed to find. It was 8 and reading lists routinely ask for
  sixteen: measured 2026-08-22, Boulding, *Chrysalis* and Marin y Beluffi were all
  logged as missing on a run in which nothing ever requested them, and all three had
  landed on an earlier run. Now 32 — above any plausible reading list — with `workers`
  as the thing that bounds the cost. A gap must mean the chain failed.
- **A text that stops mid-sentence is held, not returned** (`research._cut()`, used by
  `_partial` for the log and by `fetch_book._take()` for the decision). Clearing the
  60.000-character book floor is not enough: *Der Kollaps der Modernisierung* came off
  the open web at 12.307 words ending in the middle of a phrase — a slice of a
  212-page book that passed every gate and stopped the search before the alt-title lap
  ever reached the catalogues. Held, never rejected: if nothing whole lands, the
  fragment is still what `fetch_book` returns. `web_fulltext()` uses the same test —
  four query passes and twelve downloads instead of two and six, and it only breaks
  early on something that is plainly whole (≥150k characters, or ≥60k and not cut).
- **A chapter is never filed as the book.** For a work `_work_info` calls a book,
  anything under 60.000 characters is held by `fetch_book._take()` and the search goes
  on; if nothing whole ever lands, `fetch_book()` logs «descartado: solo apareció un
  extracto» and returns **None**. Measured 2026-08-22: archive.org answers «El colapso
  de la modernización» with *El-colapso-de-la-modernizacion-capitulo-1.pdf*, chapter
  one, under the exact title asked for, clearing every gate — and the open web answers
  with the same 31k excerpt. A gap in the dossier is recoverable; a chapter cited as
  the whole book is not. Works `_work_info` calls short carry no floor, and Calibre is
  exempt — a book the user owns is the edition to read.
- **`libgen_book(max_bytes=…)` is 200 MB, not 40.** 40 MB is a text PDF's size; a
  scan of a 368-page book is 75 MB. Measured 2026-08-23: Library Genesis holds the
  exact edition of Finney y Jones, *Interstellar Migration and the Human Experience*
  (75 MB, 119.442 words once extracted, 32s to download) and it was the only
  candidate in either mirror — dropped unread, and the run reported the work as
  unfindable. Candidates are still sorted small-first, so the cap only decides for a
  work that has nothing smaller.
- **Nothing overwrites a file already in `library/`.** The libgen branch builds
  `library/<Autor> - <Título>.<ext>` and then `unlink`s it if the download will not
  extract; a user-owned scan sitting under exactly that name would be replaced and
  then deleted. A colliding download is written to `… [libgen].<ext>` instead.
- **ePub chapters are XHTML and get parsed as XML.** `read_local` hands any document
  carrying an `<?xml` declaration to BeautifulSoup with `features="xml"` — an HTML
  parser over them is exactly what `XMLParsedAsHTMLWarning` complains about, and the
  lxml XML parser is the reliable route for entities and empty elements. A chapter
  with no declaration, or one the XML parser rejects, falls back to the tolerant
  HTML parser. `epub.read_epub` runs with `ignore_ncx=True`: the spine alone orders
  the chapters, and a broken or missing NCX must not turn a readable book into `""`
  (it only costs the table of contents).
- Prompts that touch current events carry `pipeline.hoy()` (topic, develop, outline,
  section, review). The models' cutoff is otherwise ~a year behind and they write the
  present in past tense. News sources carry `Source.date` and are sorted freshest first;
  `google_news(..., days=N)` uses Google's own `when:Nd` operator.
- **`corpus_facultad/` is style-only.** It must never reach the dossier or the
  reference list. `research.scan_library()` reads `library/` and nothing else; keep
  those two directories separate.
- `_library_gate()` blocks on `input()` in assisted mode. Anything that runs it
  unattended must pass `mode="auto"` or `ask_library=False`. `--continuous` forces
  both, for exactly that reason.
- **Articles are signed with a pen name**, `pipeline.BYLINE` (`AW_FIRMA`, default
  "Solaris"), never the author's real name. The *style* is his; the signature is not.
- **A rejected article can be fixed by hand and re-approved.** In assisted mode
  `run_pipeline()` loops on `final_approval`: it prints the dictamen plus
  `ajustes_menores`, writes the current text to `04_draft_corregido.md` and offers
  `[e]ditar / [p]ublicar igual / [n]o`. On `e` it waits on Enter, re-reads the file,
  re-verifies citations (reporting any the edit left unbacked) and asks PRO again, so a
  run rejected over a leftover «[dato a verificar]» is not lost. The detector score is
  the one measured before the edit — re-run `--detect` if the edit was substantial. In
  `auto` (and therefore `--continuous`) the loop never runs.
- **`04_draft_corregido.md` is the cache of the post-humanizing correction**, and the
  file the approval gate hands the user to edit. That correction (the one triggered by
  «la reescritura dejó N citas sin respaldo») is a whole-document PRO call and it used
  to be the only stage without a cache file: every `--resume` paid for it again, and a
  hand edit made at the approval prompt was thrown away by the next resume, which
  restarted from `04_draft_humanizado.md`. Citations are re-verified after loading it,
  so `report["citas_validas"]` — and therefore the reference list — reflects the text
  actually published. Delete the file to force the correction to run again. The verdict
  itself is deliberately *not* cached: a resume must re-ask PRO, or an edited article
  would keep its old rejection.
- Publishing defaults to `draft`. `--publish auto` only goes live when PRO
  returned `publicable` **and** the detector score is under `--threshold`.
  `final_approval()` coerces `publicable` to a real bool (a model's `"false"` string
  is truthy and used to skip the edit gate) and `wp_status` checks `is True`.
  dev.to front matter quotes title, description and cover as JSON strings — an
  unquoted «X: Y» title is a YAML error and the post is refused — and tags are
  folded to ASCII alphanumerics. A cover failure or an unparseable 2xx body after
  the post exists only logs (`publish._created`), so the receipt below is always written;
  `publish.publish_run()` writes `09_publicado.json` as the receipt that stops a
  resumed or looping run from posting the same article twice.
- **Covers are real Wikimedia Commons images, not generated ones** (2026-09-27).
  Keyless Pollinations serves the small `sana` model whatever model is asked for
  (its EXIF says `"actualModel":"sana"`), and g4f's image models (`flux-pro`,
  `gpt-image`) route to that same endpoint; `flux` via HuggingSpace returns a
  `/tmp` URL dev.to cannot fetch later. `publish.find_cover()` has FLASH write
  search queries, filters Commons hits (≥1200px, landscape, jpeg/png, nothing
  tagged AI-generated or `PD-algorithm`), and FLASH picks one or none. The credit
  line (`publish.credit()`) is appended to the body because CC BY/BY-SA require
  it. `python publish.py --replace-covers [--dry-run]` swaps the cover of every
  post in `output/*/09_publicado.json` on dev.to, from the live body_markdown;
  posts already on Commons are skipped, and a generated cover with no fitting
  replacement is dropped.
- `pipeline.already_written()` feeds past run titles back into `TOPIC_PROMPT`. Without
  it a continuous run rediscovers the same three topics forever. The twelve newest go
  in a separate `{recent}` list so the `written[:3000]` cut can never drop them.
- **Interests are a file, not a constant.** `interests.txt` (personal, gitignored) or
  the shipped `interests.example.txt`, one area per line; `--setup` copies the example
  and offers to replace it (`main.edit_interests`). `pick_topic` samples `TOPIC_AREAS`
  (4) areas per round and queries the news on each area's first term
  (`_area_query`): handed the whole list and fixed economy headlines, PRO proposed
  political economy every single time. The prompt caps Marxism/value-critique at one
  lens in one topic, and `auto` takes a random topic without listing the five.

## Detector calibration (measured, do not re-derive)

### 2026-08-25/26, real sections, prompt A/B, judge reliability (`bench2.md`, part 2)
Harnesses `bench_secciones.py`, `bench_reglas.py`, `bench_ruido.py`; raw data in
`bench/{secciones,reglas,ruido}.json`. All on the 20260824 `--english` run.

- **Dossier grounding buys nothing on the judges.** Seven real, `[p. N]`-anchored,
  citation-dense sections read 62–92 to the worst judge, **median 90** — worse than the
  78 of the previous bench's dossier-less `guided_sonnet_en`. **corr(citas, peor juez) =
  −0.42**: the two zero-citation sections read 90 and 92, the twelve-citation one 92.
  The judges *say* «datos concretos, citas» and do not score on it. The 68 of the
  2026-08-24 table was never a bench artifact — do not re-litigate it.
- **corr(local score, peor juez) = −0.85.** The section the stylometric floors like
  least (sec03, local 22.2) is the one the judges like most (62). `local_score()` now
  returns `issues_texto` — everything except the five pure-floor messages (burstiness,
  sentence length, long-sentence ratio, ttr, paragraph cv) — and `humanize()` feeds
  **only that** to `_rewrite`. The floors still count towards the score; they are no
  longer instructions, because asking a model to hit a burstiness target is what
  manufactures the short-verdict and antithesis paragraphs the judges price as tells.
- **Six interventions on one section, six nulls** (worst judge, per sample): prompt as it
  ships 78/90/90/94 (mediana 90); sin las siete reglas 90/93; sin las siete y sin las
  cuotas de ritmo 92/93; con aparato al pie, densidad documental y textura tipográfica
  88/90/95 (mediana 90); **magro** —sin style block, sin guía, sin muestras, sin reglas,
  58.017 → 26.961 caracteres— 88/94; el mismo prompt escrito por **opus** en vez de
  sonnet 91/92. Every arm lands between 88 and 95. **The seven rules and the quotas
  stay** (removing them is slightly worse), and **the 2026-08-24 model ranking does not
  transfer**: it was measured on standalone ~50-word prompts, and opus drafting a real
  dossier-grounded section is no better than sonnet. On this task the score is a function
  of neither the prompt nor the model.
- **`--threshold 25` is unreachable and costs four rewrite rounds per article to fail.**
  The floor for anything this pipeline writes is 78. Either gate relatively (no worse
  than the drafted text) or set `--detector-rounds 0` and read the judges as a report.
  Left alone deliberately — it is a product decision, not a bug.
- **The gate is not noise.** Same text, three passes, per judge: human windows move ≤3
  points and never read above 22 (`human_es_1` 8–22, `human_es_6` 3–12, Postone 1–4);
  generated text sits at 62–95 just as stably. The 78-vs-94 swing between two samples of
  one arm is *generation* variance. **`corpus_en/`'s OCR debris is not what earns the low
  score**: stripping every soft hyphen and hard wrap from the Postone window moves it by
  at most 5 points on one judge. **`opus` is nearly constant at 72 on generated text** —
  it separates, it does not rank; read `gpt-5.6-terra`'s ordering.
- **Where the gap is.** Clean contemporary published human prose: **8–22**. Everything
  this pipeline produces: **78–95**. Reproducible to ±3 and untouched by every
  prompt-level change tried. The only measured lever remains the drafting model, now
  reachable: `pipeline.DRAFT_ROLE` (`AW_BORRADOR=pro`, `--drafter pro`) makes PRO write
  the sections instead of FLASH; `draft()` had `llm.FLASH` hardcoded.

### 2026-08-24, seven drafting models, four judges, ES and EN (`bench2.md`)
Judged over 1.200-word windows: six real windows of `corpus/`, six of `corpus_en/`, and
54 generated texts. Judges are opus, sonnet, gpt-5.6-terra and kimi-k3 — one per family.

Human floor, per judge:

| judge | on the author (ES) | on real English academic prose |
|---|---|---|
| opus | 3–12% | 2–4% |
| sonnet | 3–12% | 2–15% |
| kimi-k3 | 4–10% | 2–12% |
| gpt-5.6-terra | 12–28% | 1–8% |

- Worst judge over a *human* window never passes **28**; worst judge over the best
  generated text is **68**. The four judges agree on humans and disagree on machines.
- **`gpt-5.6-terra` runs ~10 points hotter on Spanish humans and is still the sharpest
  separator.** **`kimi-k3` is the best-calibrated of the four** — tightest human range
  in both languages, and the only non-Anthropic, non-OpenAI read in the pool.
- **A model is soft on its own prose.** `guided_opus_es` reads 18% to opus and 78% to
  terra; `guided_sonnet_es` reads 22% to sonnet and 91% to terra. That is the argument
  for a judge pool wider than the drafting model, and the cost of `_judges()` being
  exactly PRO and FLASH.

**Drafting models, worst of opus/sonnet/gpt-5.6-terra, best of three prompt
conditions** (kimi is excluded from the ranking so every model is compared on the same
three judges — hyper and zen were both out of credit by the time x-preview ran):

| model | ES | EN |
|---|---|---|
| **opus** | **68** | **78** |
| sonnet | 82 | 78 |
| gpt-5.6-terra | 82 | 88 |
| x-preview-f-free | 86 | 90 |
| kimi-k3 | 88 | 88 |
| deepseek-v4-pro | 88 | 92 |
| qwen3.8-max | 93 | 92 |

- **Opus wins both languages by ~15 points, and the hyper trio is the bottom of the
  table — which is where the pipeline drafts.** `qwen3.8-max` is PRO and
  `deepseek-v4-pro` is FLASH in a default `.env`, and they are the two worst drafters
  measured. If a run can afford `AW_BACKEND=claude`, it should draft there.
- **The rhythm quotas without a style guide are actively harmful, worst for the weak
  models.** The compact block plus quotas (`styled`) is worse than a bare instruction
  for every model in EN (opus 78→88, sonnet 78→91) and for the whole hyper trio in ES
  (deepseek 93→97, qwen 93→96, kimi 93→96). The full `style_block` (`guided`) repairs
  it: kimi 96→88, deepseek 97→88, opus back to 78 at local 0. A model that cannot
  absorb the quotas obeys them mechanically and manufactures its own tells.
- **The local score is not a gate, and this bench is the clearest proof.**
  `styled_qwen3.8-max_en` scores local **0** — a perfect stylometric match — and **96**
  to the judges. Four texts score local 0; none is under 78.
- **Nothing generated clears 25.** These are standalone sections with no dossier, so no
  citations and no data density, and the judges name «datos concretos, citas» as their
  main human signal — the 68 is not the pipeline's number, the *ordering* is what the
  bench measures.
- **`kimi-k3`'s read is missing on six `styled_*` texts and on all six x-preview ones**
  (hyper out of credits, and zen answers `Insufficient balance` for everything but its
  `-free` tier); the ranking above is computed without it for every model.

### 2026-08-22, the older judge pool
Re-measured on 2026-08-22 with the then-current judge pool (`minimax-m3` sits where qwen
did in the older table) and against a draft written under the capped quotas, so the
numbers below already fold in both changes. Corpus side: the 16 reference articles.
Draft side: the 10.8k-word draft of the 20260822 naves-generacionales run.

| | local | glm-5.2 | kimi-k3 | minimax-m3 |
|---|---|---|---|---|
| author's real articles | 14–49 | 4–18% | 3–10% | 5–18% |
| generated draft, mid-sections | — | 82% | 52–88% | 62–72% |
| generated draft, first 24k | 57 | **18%** | 87–90% | 58–72% |

- **The cross-family judges are still the gate.** Every corpus article reads
  «humano»; every drafted mid-section clears 50% on all three judges.
- **But the judges used to only ever see the first 24k characters** — fixed
  2026-08-22: `llm_judges` now splits a long text into three windows (opening,
  middle, tail, ~24k characters each) and every judge reports its *worst* window
  (`humanize.JUDGE_SPAN`). The reason it mattered: glm read *that opening* of the
  naves-generacionales draft as human (18%, stable across two passes) while scoring
  the same draft's mid-sections at 82%. From `largo` up the old single-window gate
  scored the opening, so a machine-sounding tail could slip past under one judge.
  The windowing triples the judge calls on long formats; that is the price of the
  gate measuring the whole text.
- **The local stylometric score is not a gate.** At 1.500-word scale the author's own
  burstiness (0.46–0.58) overlaps a draft's (0.49); whole articles measure 0.53–1.09
  and a fresh draft sits just under the p25 floor. It exists to tell the rewriter
  *what* to change. Two didactic corpus articles («conceptos-claves…», «de la pasión
  de lo real») score 43–49 locally because the repetition check fires on their
  legitimately repeated definitions — expected noise, and one more reason the pass
  decision belongs to the judges. It was previously scored against the 194k-word
  aggregate, which flagged the author's own prose at 49–58; it is now scored against
  `ranges` (p25 of same-sized windows, degenerate windows filtered).
- `humanize._rewrite()` works block by block with counted quotas. A whole-document
  rewrite regresses to the model's default register and barely moves the judges.
- Convergence below 35 is gradual (98 → 88 → …) and not guaranteed in four rounds.

### What actually moves the judges (measured, 2026-08-21)
Two drafts of the same article on the same dossier, written by hand:

| | local | glm-5.2 | kimi-k3 | qwen3.8-max |
|---|---|---|---|---|
| v1 | 14 | 8% | 22% | 64% |
| v2 | 4.2 | 8% | 20% | **22%** |

v2 is under threshold — the target *is* reachable. The seven edits that did it are
now encoded in `pipeline.SECTION_PROMPT` and `humanize.REWRITE_PROMPT`:
open on a concrete particular rather than a thesis; one-line paragraphs between long
ones; explicit disagreement with a cited author; a first-person self-correction; one
verbatim quote left in its original language; no metadiscursive connectives
("empecemos por", "detengámonos en"); a close with no programme.
The judges' own reasons name these exactly — they read "riesgo retórico real" and
"idiosincrasia" as human, and "estructura ensayística muy regular" as machine.

**Two of those seven were capped on 2026-08-21** — disagreement and self-correction
now run at most once per section (they used to be a per-block quota in the rewriter,
which is why `05_final.md` of the CABA run objects to an author roughly every 700
words and repeats "yo también lo sentí…" verbatim four times). The trade is
deliberate: readability and academic register over one detector signal. Re-measured
2026-08-22 under the caps: drafted mid-sections still clear 50% on every judge, so
the gate held. The other five edits, the rhythm quotas and the repetition check are
untouched.
