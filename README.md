# article-writer

Automated writer of social-science articles in the style of
[Revista La Cueva](https://revistalacueva.wordpress.com/). It goes from topic to a
reviewed final version, with real, verified bibliography, run past AI detectors.

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env             # and put your key in AW_API_KEY
python main.py --setup           # your interests, the reference corpus and the style guide
python ingest_facultad.py        # optional: add your university papers to the corpus
python main.py --refresh-style   # and rebuild the guide over the enlarged corpus
```

`--setup` first shows your interests (see below) and lets you rewrite them; then it
downloads the 16 reference articles to `corpus/`, measures the author's stylometric
fingerprint (`style_fingerprint.json`) and has the PRO model write the working style
guide (`style_guide.md`). It runs once.

`ingest_facultad.py` walks `D:/facultad` (or the path you pass) and adds your
monographs, take-home exams and reports to `corpus_facultad/`. It filters in three
steps: extension (you write in Word, the readings are PDFs), file name
(`Antropología 24-9.docx` is dated class notes, not an essay) and content — FLASH reads
a sample and decides whether it is you arguing or someone else's text, notes or a
transcript. `--dry-run` shows the classification without writing anything.

`corpus_facultad/` is **style reference only**: it never enters the bibliography and
is never cited. The bibliography comes from the online connectors and from `library/`,
which are separate directories on purpose.

For English articles (the default), `python build_corpus_en.py --guide` builds
`corpus_en/` from human-written English prose (arXiv, Gutenberg, English works in
`library/`) plus its fingerprint and style guide.

### Interests

Topics come from `interests.txt`: one area per line, lines starting with `#` are
ignored. It is personal and not committed; a fresh clone does not have it, so the first
`--setup` copies `interests.example.txt` (the author's list, as an example) and asks
whether you want to replace it. Type one area per line and an empty line to finish;
Enter right away keeps it as is. You can also edit the file by hand at any time, and
`--setup` offers it again every time you run it.

Each topic round draws four areas from that list, searches the news for them (the
first term of each line is the query) and asks PRO for five topics from different
areas, without repeating the subject of any article already written (the twelve most
recent are listed separately, so truncation never drops them). Political economy may
be a lens, not the axis of everything: an article about dinosaurs or aircraft stands
on its own.

## Usage

```bash
python main.py                                  # interactive wizard
python main.py --fmt medium --mode auto         # end to end, no questions
python main.py --topic "Cybersyn today" --exact-topic --fmt long
python main.py --topic "science fiction and value" --mode assisted
python main.py --resume                         # resume the latest run (every stage is cached)
python main.py --resume output/20260820-slug    # resume a given run
python main.py --detect text.md                 # only run the AI detectors
python main.py --spanish --fmt medium           # article in Spanish (English is the default)
```

### Flags

| Flag | What it does |
|---|---|
| `--fmt` | length and genre (table below) |
| `--mode auto\|assisted` | end to end, or asking at topic, outline, corrections and approval |
| `--topic`, `--exact-topic` | topic hint; with `--exact-topic`, that topic and no other |
| `--resume [folder]` | resume a run; without a folder, the latest |
| `--no-library` | do not stop to ask for missing books |
| `--rounds N` | PRO review rounds |
| `--detector-rounds N`, `--threshold N`, `--no-detector` | rewrites against the detectors, highest tolerated score (0 human, 100 machine), or no detectors |
| `--drafter pro\|flash` | who drafts the sections (FLASH by default; PRO reads more human and costs more) |
| `--english`, `--spanish` | article language; English by default, calibrated against `corpus_en/` |
| `--setup`, `--refresh-style` | interests + corpus + style guide; or only rebuild the guide |
| `--detect file.md` | only the detectors, over a file |
| `--publish no\|draft\|auto\|live` | upload as draft, live if approved and under the threshold, or always live |
| `--provider`, `--backups A,B`, `--pro`, `--flash`, `--models` | provider chain and models (see Models) |
| `--wizard` | the wizard asks even when other flags are given |
| `--continuous N`, `--every MIN` | N articles in a row (0 = never stop); `--topic` guides the first new article, then topics are picked automatically; `--every` sets the pause |

**You set the topic.** With `--topic` you give a hint and the system proposes five
angles to choose from. With `--topic --exact-topic` it writes about that and nothing
else: PRO only adds a hypothesis, a question and a reading plan. In assisted mode you
can adjust the hypothesis before research starts.

### Formats

| `--fmt` | Words | What it is |
|---|---|---|
| `short` | 1,200–1,600 | short intervention piece |
| `medium` | 3,000–4,000 | theory-for-general-readers article |
| `long` | 7,000–9,000 | long essay with critical apparatus |
| `paper` | 10,000–13,000 | academic paper with abstract, hypothesis and conclusion |
| `thesis` | 40,000–80,000 | thesis in chapters |
| `book` | 70,000–120,000 | book-length theoretical essay |
| `debate` | 3,500–4,500 | explicit confrontation between rival positions |

Length is a **range, not a target**: the outline splits the budget across sections
and the drafter decides how many and how long (500 to 10,000 words, deliberately
uneven). Each format also carries its own register (a `paper` and an intervention
piece do not sound alike), which overrides the generic style rules.

### Modes

- `auto` — picks the topic, researches, writes, reviews and approves with no
  intervention. It does not list the proposed topics and takes one at random (always
  the first one was always the model's most predictable idea).
- `assisted` — asks you at four points: topic choice, outline, applying the review's
  corrections, and final approval.

## How it works

```
topic (PRO)  →  research  →  outline (FLASH, audited by PRO)
             →  drafting section by section (FLASH)
             →  peer review (PRO)  →  surgical corrections (PRO)
             →  AI detectors + rewriting (FLASH + judges from other families)
             →  final approval (PRO)
```

Every stage is saved in `output/<date>-<slug>/`:

| File | Contents |
|---|---|
| `01_topic.json` | topic, hypothesis, theoretical tension, why now |
| `02_plan.json` / `02_dossier.json` / `02_faltantes.json` | queries, sources, books that could not be obtained |
| `03_plan.json` | prior architecture (only `thesis` and `book`): units, function and budget |
| `03_outline.json` | outline with word budget and sources per section |
| `04_sec01.md`… | each drafted section (cache: delete it to redo it). Long sections are split by subsection: `04_sec02_01.md`, and into parts if they still do not fit: `04_sec02_01a.md` |
| `04_draft.md`, `04_draft_humanizado.md`, `04_draft_corregido.md` | draft, humanized version, post-humanizing correction |
| `05_final.md` | **the article** |
| `06_review.json`, `07_detector.json`, `08_approval.json` | review, detectors, verdict |

## Bibliography

Sources consulted, all without an API key:

- **Papers**: OpenAlex, Crossref, DOAJ (good Spanish-language coverage), Semantic
  Scholar, arXiv.
- **Full text**: Unpaywall (legal open access) and, for what sits behind a paywall,
  Sci-Hub. Sci-Hub mirrors are almost always behind a JS challenge, so it usually
  fails; that is not a bug. When no catalogue resolves a work, the search ends on the
  open web with headless Chrome (Selenium), the one route that still gets through
  where search engines block keyless HTTP.
- **Current affairs**: Google News RSS (Spanish and English) and GDELT.
- **Books**: your Calibre library first (queried by title, never scanned), then Open
  Library, Project Gutenberg, archive.org and Library Genesis; whatever lands is saved
  to `library/` for the next run. Journal articles are resolved by `fetch_paper()`
  (Crossref/arXiv → DOI → Unpaywall → Sci-Hub), and Anna's Archive is only a link in
  the list — Cloudflare blocks it.
- **Theory archives**: Monthly Review, Viewpoint, Brooklyn Rail, Spectre, Historical
  Materialism, libcom.org, and Revista La Cueva itself.
- **Your library**: drop `.epub`, `.pdf`, `.docx`, `.txt`, `.html` (or Calibre formats
  like `.azw3`, converted with `ebook-convert`) into `library/` and they are indexed
  automatically.

### The library gate

Before writing a single line, the system compares the works the plan declared
essential against the ones it actually obtained **in full text** (a two-line abstract
does not count). For whatever is missing it stops and shows you, work by work, the
direct-download candidates it found plus seven fallback links —Anna's Archive,
Library Genesis, Z-Library, Marxists Internet Archive, Google Books, Internet Archive,
Open Library—. Drop the files in `library/`, press Enter, and it checks again; you can
also paste a URL and it downloads it on the spot. Up to three rounds. Type `skip` to
go on without them. In `auto` mode it prints the report and carries on without
blocking. With `--no-library` it does not even ask.

### Citations: APA 7 and nothing invented

Citations follow **APA 7** in the article's language: `(Postone, 2006)`,
`(Postone, 2006, p. 302)`, `(Kurz & Jappe, 2016)` in English or `(Kurz y Jappe, 2016)`
in Spanish, `(Marx et al., 1867)`, narrative citations when the author is the subject,
and an alphabetical **References** list with italic titles and DOIs.

**No citation is invented.** The model may only cite keys present in the dossier;
`research.verify_citations()` checks every `(Author, year)` in the text against the
real sources, before and after rewriting, and sends anything unbacked back for
correction.

## Prose review and detectors

Drafting and rewriting share editorial criteria in English and Spanish: clarity,
continuity of the argument, precision and idiomatic syntax. Corpus samples guide the
voice; their facts and citations are not sources for the article. The metrics
describe the style and do not impose quotas of short sentences, subordinate clauses,
parentheses, enclitics or punctuation marks.

Each new section is checked before it is saved: length within the requested range and
the right language. A reply outside ±10% gets a retry; after that, the right-language
reply closest to the budget is kept if it lands between 70% and 150%. A plainly cut or
wrong-language section is retried on the main provider and then handed to each backup
in the chain; the run stops only when every link failed, and nothing incomplete is
cached. Review and correction receive the available dossier passages so claims can be
checked against the sources.

Rewriting receives the context of the neighbouring blocks. Each proposal must keep the
citations, pages, figures, verbatim quotations and note markers, and stay within ±10%
of the original length. If it alters any of them, the previous block is kept. This
check does not verify meaning on its own: the fidelity of the claims still needs
editorial and documentary review.

The report separates three signals:

1. Local stylometry: diagnostics for revising the prose, not an approval decision.
2. LLM judges: a fixed panel (`claude-opus-5-5` on `claude`, `gpt-6.1-sol` on
   `oauth`), over the opening, middle and end of long texts. If every judge fails,
   the drafting model judges instead.
3. External detectors: the configured services and, in English, the optional local
   classifier if available.

The threshold applies to the worst valid score among judges and external detectors.
With no valid answers the result is undetermined, never a pass. The best measured
version is kept and the report records its round, language and SHA-256 hash. If a
later correction changes the text, it is measured again. Automatic publishing requires
editorial approval and an explicit result under the threshold; skipping the detectors
does not count as passing them.

**Language.** English is the default (`--spanish` for Spanish). English uses
`corpus_en/`, the English guide and English-specific drafting criteria, and also
translates the labels, date and bibliography the program generates. Without an English
corpus or guide it still writes with the general criteria. Calls to CLI writers are
isolated from the project's coding instructions.

**Console and prompts are in English** in both languages. Every console message,
wizard question and interactive prompt (`[y/N]`, `[e]dit / [p]ublish / [n]o`,
`[w]ait / [c]hange / [a]bort`) is English, and so are the instructions the models
receive; the article's language is set by a directive appended to each prompt
(`pipeline._lang()`), so `--spanish` still writes Rioplatense Spanish. JSON field names
and `output/` file names are unchanged, so old runs resume as before. The
Spanish-specific judge and rewriter (`humanize.JUDGE_PROMPT`, `REWRITE_PROMPT`) and the
Spanish guide builder (`style.BUILD_PROMPT`) stay in Spanish: they only run under
`--spanish`.

Scores are review signals, not proof of authorship. A low score does not guarantee
that another detector will accept the text, nor that its claims are correct.

## Models

| Role | Model | Used for |
|---|---|---|
| PRO | the main provider's | topic, outline audit, review, corrections, final approval |
| FLASH | the main provider's | queries, outline, drafting, rewriting |
| Judges | `claude-opus-5-5`, `gpt-6.1-sol` | detecting generated text |

Each backup in the chain answers with its own models (`--models`, `AW_MODELS`):

| Provider | Default PRO / FLASH |
|---|---|
| `claude` | `claude-opus-5-5` / `sonnet` |
| `hyper`, `go` | `qwen3.8-flash` / `deepseek-v4.1-flash` |
| `zen` | `glm-5.3-flash` / `deepseek-v4.1-flash` |
| `grok` | `grok-4` / `grok-4-fast` |
| `g4f` | `deepseek-v4-pro` / `glm-5.3` |
| `oauth` | `gpt-6.1-sol` / `gpt-6-luna` |

Change them in `.env` (`AW_BACKEND`, `AW_MODEL_PRO`, `AW_MODEL_FLASH`, `AW_MODELS`),
with flags (`--provider/--backups/--pro/--flash/--models`) or in the interactive
wizard (first question). `AW_BACKEND` is an **ordered chain** of providers: the call
goes out through the first and walks to the next if it does not answer. `claude` is
the Claude Code CLI in print mode (it runs on your subscription, not a metered key)
and is *exclusive*: it only answers for models in its own catalogue. In an attended
run, when the whole chain fails you are asked to wait 30s, change provider/model, or
abort.

### OpenCode fallback

If the main provider does not answer after every retry, the call goes out through the
`opencode` CLI against the `opencode-go` provider, which serves the same families (and
some hyper lacks, like `glm-5.3`). It is automatic; turn it off with
`AW_OPENCODE_FALLBACK=0`. In the menus and in `AW_BACKEND` the provider is called `go`
(the old name `opencode` is still accepted).

Every call goes through `book writer`'s `AIService` (the workspace's shared AI suite;
sibling folder `../book writer` or `AW_BOOK_WRITER`): each link in the chain uses that
provider's configuration in book writer, asks for the model's full output allowance,
retries a truncated reply with double the budget, and the last link of an unattended
run waits out a usage limit instead of dying.

The project ships an `opencode.json` with the `hyper` provider already configured and
`CLAUDE.md` as instructions, in case you want to work on the repo from there:

```bash
export AW_API_KEY=sk-hyper-...
opencode        # from article-writer/
```

## Lamplight

The game (library, historical campaign and Godot client) was split out on 2026-09-23
to [`../lamplight`](../lamplight/README.md). It imports `llm`, `pipeline`, `research`
and `style` from this folder: if their public API changes, run its tests too.

## Verification

```bash
python test_article_writer.py    # deterministic checks, no network or API key
python -m pytest -q -p no:cacheprovider   # alternative, if you have pytest installed
python bench_bilingual.py --backend hyper  # real test, spends the configured API
```

The bilingual test saves drafts, edited versions and reports in
`output/bilingual-check-<date>/`. It uses an explicitly fictional case to check
language, draft length, citations and content preservation during rewriting. Review
may shorten repeated or unbacked material; the style rewrite must stay within ±10% of
the reviewed version. An LLM editor scores naturalness, clarity, coherence, precision
and fidelity, and requires at least 4/5 on each. Detector results are reported
separately and are not presented as proof of quality or of human authorship.
