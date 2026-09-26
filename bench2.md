# Bench 2026-08-24 — humanizers, detectors and drafting models, ES + EN

Harness: `bench2.py` (`gen` → `hum` → `score` → `report`), raw data in `bench/scores.json`,
every text in `bench/*.md`. 66 texts, ~1.200 words each, one topic (automation of work
and the fate of value), so nothing below is a topic effect.

Seven drafting models: **opus**, **sonnet** (claude CLI), **gpt-5.6-terra** (oauth proxy),
**qwen3.8-max**, **kimi-k3**, **deepseek-v4-pro** (hyper) and **x-preview-f-free**
(OpenCode Zen through the `opencode` CLI — the free tier answers with no
`OPENCODE_API_KEY`). Four judges, one call per text per judge, worst window: opus,
sonnet, gpt-5.6-terra, kimi-k3 — one per family.

The human floor is not an assumption: six real 1.200-word windows of `corpus/`
(the author) and six of `corpus_en/` (real arXiv papers and Gutenberg social theory).

## 1. The human floor separates cleanly

Every judge, over the twelve real human windows:

| judge | on the author (ES) | on real English academic prose |
|---|---|---|
| opus | 3–12 | 2–4 |
| sonnet | 3–12 | 2–15 |
| kimi-k3 | 4–10 | 2–12 |
| gpt-5.6-terra | 12–28 | 1–8 |

| | local | worst judge |
|---|---|---|
| author's real prose (6 windows) | 0–23 | ≤28 |
| real English academic prose (6 windows) | 0–21 | ≤15 |
| every generated text (54) | 0–68 | ≥68 |

Worst judge over a *human* window never passes **28**; worst judge over the best
generated text is **68**. The four judges agree on humans and disagree on machines,
which is the shape a working gate has.

`gpt-5.6-terra` runs ~10 points hotter than the others on Spanish humans (12–28 where
the rest sit at 3–12) and is still the sharpest separator. Read its ordering, not its
absolute number. **kimi-k3 is the best-calibrated of the four** — the tightest human
range in both languages — and it is the only non-Anthropic, non-OpenAI read in the pool.

**Self-judging bias is visible.** `guided_opus_es` reads 18% to opus and 78% to terra;
`guided_sonnet_es` reads 22% to sonnet and 91% to terra. A model is soft on its own
prose, which is the argument for keeping the judge pool wider than the drafting model.

## 2. No third-party humanizer beats ours (Spanish). In English ours *hurts*.

Same input for every candidate: `raw_sonnet_es` / `raw_sonnet_en`.
`similitud` is the character-level ratio against that input — how much the tool
actually did.

### Spanish (input `raw_sonnet_es`: local 29.8, worst judge **84**)

| candidate | similitud | local | worst judge |
|---|---|---|---|
| **ours**, rewriter = sonnet | 0.44 | 17.2 | **78** |
| **ours**, rewriter = deepseek-v4-pro (production's FLASH) | — | 29.8 | 82 |
| texthumanize 0.35 | 0.98 | 17.7 | 82 |
| lynote standard pipeline | 0.25 | 56.6 | 82 |
| translation chain (es→fi→es) | 0.34 | 33.0 | 93 |
| humano 2.0.1 | 0.99 | 20.1 | 99 |

### English (input `raw_sonnet_en`: local 11.8, worst judge **78**)

| candidate | similitud | local | worst judge |
|---|---|---|---|
| texthumanize 0.35 | **1.00** | 11.8 | **78** (= the input, untouched) |
| lynote standard pipeline | 0.02 | 24.3 | 80 |
| **ours**, rewriter = deepseek-v4-pro | — | 2.2 | 82 |
| translation chain (en→fi→en) | 0.02 | 9.1 | 88 |
| **ours**, rewriter = sonnet | 0.37 | 3 | **95** |
| humano 2.0.1 | 0.99 | 4.4 | 99 |

**Which model does the rewriting is itself a variable, and it flips by language.**
In Spanish sonnet rewrites better than deepseek (78 vs 82); in English deepseek is far
less damaging than sonnet (82 vs 95). Neither *improves* on the English input.

What the numbers mean, candidate by candidate:

- **texthumanize 0.35.0** returned the English input **byte for byte** (similitud
  1.000) and changed 2% of the Spanish one. Its `syntax_rewriting` stage raises
  `Unsupported language: 'es'. Use 'en', 'ru', 'uk', or 'de'` — the «25 languages»
  claim covers the cosmetic stages only. Re-measured, same verdict as 2026-08-23.
- **humano 2.0.1** (new since the last survey) is an English filler injector. Over
  Spanish prose it produced «La fábrica textil que visité en 2019 no tenía casi nadie.
  **Here's the deal -** dos operarios recorrían…». It made the text *worse* on every
  judge, in both languages, and it is the worst candidate in the table.
- **The translation chain** (lynote's method, hopping through Finnish) rewrites
  everything and breaks Spanish grammar doing it: Finnish has no articles, and the
  return leg drops them — «En 2019 visité fábrica textil». Meaning survives; the
  language does not.
- **lynote's «standard pipeline»** (chain + LLM rewrite at temperature 1.3) is the
  only third-party method that beats ours anywhere: 80 vs 95 in English. It gets there
  by discarding the text and writing a new one (similitud 0.02), which for us is
  disqualifying — every citation would have to be re-verified against a document that
  no longer shares a sentence with the dossier-grounded draft.
- **`ai-text-humanizer` 1.0.0** does not build (`BackendUnavailable: Cannot import
  'setuptools.build_meta'`, also with `--no-build-isolation`). English and Persian only
  per its own description.
- Not on PyPI, clone-only, not tested: `llmstrip`, `StealthHumanizer`,
  `AI-Text-Humanizer-App`, `harshaneel/humanize` (a rule-based prompt skill, and
  already the source of `ENGLISH_TELLS`).

**The English regression is the finding worth acting on.** `humanize._rewrite` in
`LANG="en"` drives the local score to 3 with sonnet and 2.2 with deepseek — the two
best numbers in the table — while pushing the worst judge from 78 to 95 and 82. It is
optimising the stylometric floor and manufacturing tells doing it: exactly the failure
the Spanish quotas were capped for on 2026-08-23, except `REWRITE_PROMPT_EN` never got
that pass. Two rewriters, same direction, so it is not one model's quirk.

## 3. Not one downloadable classifier is usable

Every published AI-text classifier with meaningful downloads, worst 512-token window,
same texts:

| classifier | on real human prose | on generated text | verdict |
|---|---|---|---|
| `Hello-SimpleAI/chatgpt-detector-roberta` (wired in today) | 0.0–14.6 | 0.0–91.7, **under 5% on 52 of 54** | near-constant zero |
| `desklib/ai-text-detector-v1.01` | 2.3–**87.1** | 0.1–100, under 5% on 21 of 54 | overlaps; false-positives the author at 87% |
| `desklib/ai-text-detector-academic-v1.01` | 3.0–**90.0** | 0.1–100, under 5% on 18 of 54 | worse; 90% on a real arXiv paper |
| `Oxidane/tmr-ai-text-detector` | 47–98 | 2.4–98 | anti-correlated |
| `yaya36095/xlm-roberta-text-detector` | **100.0** on all 12 | 100.0 on 52 of 54 | constant |
| `arincon/roberta-base-autextification` | 5.5 | 5.5 | constant |
| `arincon/roberta-base-openai-detector-autextification` | 99.8 | 99.8 | constant |

- The only **Spanish** MGT fine-tunes that exist are the two AuTexTification (IberLEF
  2023) ones, and both emit a constant. The benchmark's generators were BLOOM-era;
  nothing transfers. Nothing has been published since.
- `desklib` leads the RAID benchmark and is the best of the seven, and it still reads
  one of the author's own published articles as 87% machine while reading
  `raw_sonnet_es` as 0.4%. As a gate it would reject the target and pass the problem.
- `Hello-SimpleAI/chatgpt-detector-roberta`, the one `humanize._hf_detector()` loads
  in English mode, answers under 5% on 52 of the 54 generated texts. It fires on blatant
  assistant register (`raw_gpt-5.6-terra_en`, 91.7) and on nothing else: it was trained
  on HC3 question-answer pairs, not on essays. It has **zero false positives** across
  twelve human windows, so it costs nothing to keep — but it is not a signal, and the
  500 MB it downloads buys two hits, both on `raw_gpt-5.6-terra_en`-grade prose.
- Binoculars needs a pair of 7B models resident; it is English-focused and was already
  ruled out on 2026-08-23. Not re-tested.

**Conclusion unchanged: the LLM-judge gate plus corpus-calibrated stylometry is still
the state of the art available to us, in both languages.** Re-test when a Spanish
detector actually ships.

## 4. Drafting models, ranked by worst judge

Three prompt conditions per model: `raw` (bare instruction), `styled` (compact style
block — verbatim samples — plus the counted rhythm quotas) and `guided` (the full
`style_block`: PRO-written style guide, fingerprint and samples, i.e. what the drafting
stage really sends).

Worst of the four judges. Lower is better.

x-preview's kimi-k3 column could not be measured (hyper *and* zen both out of credit by
then), so the table below is the worst of **opus, sonnet and gpt-5.6-terra** for every
model — apples to apples. Adding kimi back moves only gpt-5.6-terra EN (88→91).

| model | ES raw | ES styled | ES guided | **ES best** | EN raw | EN styled | EN guided | **EN best** |
|---|---|---|---|---|---|---|---|---|
| **opus** | 68 | 68 | 78 | **68** | 78 | 88 | 78 | **78** |
| sonnet | 84 | 82 | 91 | **82** | 78 | 91 | 82 | **78** |
| gpt-5.6-terra | 92 | 82 | 84 | **82** | 93 | 88 | 92 | **88** |
| x-preview-f-free | 88 | 86 | 89 | **86** | 92 | 90 | 92 | **90** |
| kimi-k3 | 88 | 96 | 88 | **88** | 88 | 91 | 88 | **88** |
| deepseek-v4-pro | 93 | 97 | 88 | **88** | 96 | 92 | 96 | **92** |
| qwen3.8-max | 93 | 96 | 96 | **93** | 93 | 96 | 92 | **92** |

- **Opus wins both languages by a clear margin** — 68 where the runner-up is 82 — and is
  the only model whose prose ever reads under 25 to a judge other than itself
  (`guided_opus_es`: opus 18, sonnet 22).
- **The hyper trio is the bottom of the table, and that is where the pipeline drafts.**
  `qwen3.8-max` is PRO and `deepseek-v4-pro` is FLASH in a default `.env`, and they are
  the two worst drafters measured (93 and 88 at best in Spanish, 92 for both in
  English). `kimi-k3` ties deepseek. Every one of the three is 20+ points behind opus in
  Spanish. **If a run can afford `AW_BACKEND=claude`, it should use it for drafting.**
- **gpt-5.6-terra is a weak drafter and the strictest judge at once.** Its raw English
  is among the most machine-sounding texts in the bench (93, local 39.4, and one of only
  two texts `hello-roberta` ever caught).
- **The quotas without a style guide are actively harmful, and worst for the weak
  models.** `styled` is worse than `raw` in EN for every model (opus 78→88, sonnet
  78→91) and in ES for the whole hyper trio (deepseek 93→97, qwen 93→96, kimi 93→96).
  `guided` — the full block, style guide included — is what repairs it: kimi 96→88,
  deepseek 97→88, `guided_opus_en` back to 78 at local 0. Same lesson as the Spanish
  capping: an unbounded quota manufactures its own tells, and a model that cannot
  absorb the quotas obeys them mechanically.
- **The local stylometric score is not a gate, and this bench is the clearest proof
  yet.** `styled_qwen3.8-max_en` scores local **0** — a perfect stylometric match — and
  **96** to the judges. `styled_deepseek-v4-pro_en` is 0 and 92. Four texts score local
  0 and none of them is under 78.
- **Nothing generated clears the 25 threshold.** The best is 68 (ES) against a human
  ceiling of 28. Caveat: these are standalone sections with no dossier, so no citations
  and no data density, and the judges name «datos concretos, citas» as their main human
  signal. Do not read the 68 as the pipeline's real number — read the *ordering*.

- **x-preview-f-free lands mid-table**: 86 in Spanish, 90 in English — better than
  qwen and deepseek, well behind opus. It is also the slowest thing in the bench by a
  wide margin (96–497s per text through the CLI, and 1909s on one styled prompt before
  the contamination below was removed).

### `opencode.json` was contaminating every CLI completion

The first x-preview run answered the **English** prompt in **Spanish**, and its
`raw_en` text scored a suspiciously good 63. The cause was not the model.

`opencode.json` carries a top-level `"instructions": ["CLAUDE.md"]`, and **that applies
to every agent, `redactor` included** — the note in CLAUDE.md saying redactor has a
«one-line system prompt» is wrong. Asked directly, the redactor agent answered:

> Yes — working dir `D:\python\article-writer`: automated Spanish social-science
> article writer in the voice of Augusto Germán Rehfeldt, plus parent `D:\python`
> workspace rules and my own caveman/ponytail session modes.

So every completion on the `go`/CLI route is prefixed with ~44 KB telling the model
this project writes Spanish articles for one named author. In Spanish that is invisible
to harmless; against `--english` it overrides the task outright.

Removing the key fixes it — the same prompt then came back in English — and it is also
most of the latency: `raw_es` went 261s → 96s and `styled_es` 1909s → 465s. A per-agent
`"instructions": []` does **not** override it; only the top-level key does, and there is
no `--config` flag on `opencode run`. `opencode.json` was restored untouched after the
bench; the six x-preview texts above were regenerated with the key stripped so they are
comparable to the other six models, none of which ever saw CLAUDE.md.

Session leakage was ruled out separately: two sequential `opencode run` calls do not
share context ("What word did I ask you to remember?" → "NONE.").

### Gaps

`kimi-k3`'s read is missing on six `styled_*` texts and on all six x-preview ones:
hyper ran out of credits a second time mid-scoring, and OpenCode Zen answers
`Insufficient balance` for everything except its `-free` tier. The other three judges
scored them all, and the model table above is computed over those three for every model
so nothing is compared unevenly. Re-run `python bench2.py score` with credits to fill it.

**The `opencode` CLI route works, but slowly and only on models with balance.**
`opencode-go/kimi-k3` burned the full 900s on a judge-sized prompt (~8k characters, well
under the documented 47k ceiling) and returned «transcripción vacía»; the same model on
the `opencode/` (zen) prefix answers `Insufficient balance` in 5s, so the silent empty
transcript was very likely a billing failure the CLI never surfaced. `x-preview-f-free`
runs fine on the same route — the free tier is the part with credit. Nothing streams
there (one fresh process per completion), so a slow reasoning model needs more than the
old hardcoded 900s: `AW_OPENCODE_TIMEOUT` now raises it, and one x-preview prompt took
1909s.

## Reproduce

```bash
python bench2.py gen     # 42 texts: 7 models x 2 langs x 3 prompt conditions
python bench2.py hum     # 6 humanizers over raw_sonnet_{es,en}
python bench2.py score   # local + 4 judges + 7 classifiers, cached per text
#   AW_OPENCODE_TIMEOUT=3000 is needed for the x-preview rows (CLI route, no streaming)
python bench2.py report
```

---

# Bench 2026-08-25/26 — real sections, prompt A/B, and how much of the score is noise

Harnesses: `bench_secciones.py` (real pipeline output), `bench_reglas.py` (prompt A/B on
one section), `bench_ruido.py` (the same text judged repeatedly). Raw data in
`bench/secciones.json`, `bench/reglas.json`, `bench/ruido.json`; every text in `bench/`.
All of it on the 20260824 `--english` run (`paper`, 9 sections, drafted by sonnet
through `claude -p`), judged with the same four judges and the same 1.200-word windows
as the tables above.

## 1. The 68 was not a bench artifact — dossier grounding buys nothing

`bench2.md`'s caveat was that its 54 texts were standalone sections with no dossier, no
citations and no page markers, while the judges name «datos concretos, citas» as their
main human signal. Measured directly on seven real, dossier-grounded, `[p. N]`-anchored
sections:

| sección | citas | local | opus | sonnet | terra | peor |
|---|---|---|---|---|---|---|
| sec01 | 0 | 0.7 | 72 | 90 | 88 | 90 |
| sec02 | 4 | 0 | 72 | 82 | 78 | 82 |
| sec03 | 10 | 22.2 | 62 | 30 | 58 | **62** |
| sec05 | 5 | 3 | 55 | 62 | 78 | 78 |
| sec06 | 12 | 3 | 72 | 78 | 92 | 92 |
| sec07 | 4 | 3 | 72 | 88 | 92 | 92 |
| sec08 | 0 | 0.3 | 22 | 82 | 92 | 92 |

Median worst judge **90**, against 78 for the standalone `guided_sonnet_en` of the
previous bench. Real pipeline output is not better than the bench's; it is slightly
worse. **corr(citations, worst judge) = −0.42**: the two sections with zero citations
read 90 and 92, the one with twelve reads 92, and the best-reading section has ten. The
judges' stated reason is not their decision function, and `min_citas` buys nothing on
this axis.

## 2. The local stylometric score points the wrong way

**corr(local score, worst judge) = −0.85** over those seven sections. The section the
floors like least (sec03, local 22.2) is the one the judges like most (62); the two that
satisfy the floors (sec08 0.3, sec02 0) read 92 and 82. This is the third independent
measurement in the same direction — `styled_qwen3.8-max_en` at local 0 and judge 96, and
the English `_rewrite` regression (local 2.2–3, judge 82–95) are the other two.

Acted on: `local_score()` now returns `issues_texto` alongside `issues`, and
`humanize()` feeds **only** `issues_texto` to the rewriter. The five pure-floor messages
(burstiness, mean sentence length, long-sentence ratio, ttr, paragraph cv) still count
towards the score and are no longer instructions. What the rewriter still hears are
concrete, quoted defects: repeated 6-grams, cliché hits, structural tells, colon
density, antithesis density.

## 3. Four prompt interventions, all null

One section (sec08), one model (sonnet), temperature 0.9, identical dossier, synopsis and
`evitar` — the prompt is the only variable. Worst judge, per sample:

| arm | samples | mediana |
|---|---|---|
| prompt actual | 78, 90, 90, 94 | **90** |
| sin las siete reglas «humano, no máquina» | 90, 93 | 91.5 |
| sin las siete y sin las cuotas de ritmo | 92, 93 | 92.5 |
| **con aparato al pie, densidad documental y textura tipográfica** | 88, 90, 95 | **90** |
| **magro**: sin style block, sin guía, sin muestras, sin reglas (58k → 27k caracteres) | 88, 94 | 91 |
| el prompt actual escrito por **opus** en vez de sonnet | 91, 92 | 91.5 |

- **The seven rules stay.** Removing them made the section slightly *worse*, not better,
  so the sec03/sec08 contrast (62 obeying none, 92 obeying all) is not caused by the rule
  block. The rhythm quotas are the same story.
- **The apparatus block is a null result on the judges.** It works as designed — the
  drafts come back with numbered footnotes carrying real bibliographic detail («Revista
  Espaço Acadêmico, v. 21 (2021), pp. 67-78, published 1 June 2021»), bracketed German
  ([Abspaltung]), exact dates, institutions and roles — and the judges do not care. It is
  kept because a documented article is better than an undocumented one, not because it
  moves this number.
- **Cutting the prompt in half is also null, and that is the informative one.** The
  `magro` arm deletes the 23k-character style block (guide, fingerprint, corpus samples)
  and the whole 8k rule list, leaving the commission, the synopsis, the sources and four
  lines of instruction — 58.017 characters down to 26.961. Same 91. The hypothesis it
  was testing (that a wall of instruction pushes any model into a compliant,
  instruction-following register, which would explain why adding or removing 2k of rules
  changes nothing) is wrong: the register is not coming from the prompt at all.
- **And neither is the drafting model.** `opus` writing this same section from this same
  prompt reads 91 and 92 — no better than sonnet's 90. bench2's 68 for opus came from a
  *standalone* ~50-word prompt with no dossier; on the real task the model ranking does
  not survive. The «one measured lever» of the 2026-08-24 table does not transfer.

## 4. The gate is not noise, and the human floor is real

The same text judged three times, judge by judge:

| texto | opus | sonnet | terra | kimi |
|---|---|---|---|---|
| `human_en_1` (Postone, *Time, Labor and Social Domination*) | 3, 3, 3 | 3, 3, 3 | 1, 1, 1 | 4, 3, 3 |
| `human_en_1_limpio` (same, OCR artifacts removed) | 3, 3, 3 | 8, 3, 4 | 1, 2, 1 | 3, 3, 4 |
| `human_en_3` (Veblen) | 3, 2, 3 | 2, 2, 2 | 1, 0, 1 | 2, 3, 2 |
| `human_es_1` (el autor, scraped from WordPress) | 8, 8, 8 | 12, 8, 22 | 18, 18, 18 | 10, 10, 8 |
| `human_es_6` (el autor) | 4, 4, 4 | 6, 3, 8 | 12, 12, 12 | 8, 5, 7 |
| `reglas_viejo_1` (generated) | 72, 72, 72 | 82, 88, 72 | 88, 86, 78 | 82, 91, 90 |
| `reglas_nuevo_1` (generated) | 62, 62, 72 | 82, 82, 88 | 88, 86, 86 | 18, 84, 72 |
| `guided_sonnet_es` (generated) | 87, 72, 72 | 62, 82, 62 | 86, 82, 82 | — |

- **Judges are near-deterministic on human prose**: spread ≤3 points within a judge, and
  never above 22 across five human windows in two languages. The 78-vs-94 swing between
  two samples of the same arm is *generation* variance, not judging variance.
- **`corpus_en/`'s OCR debris is not what earns the low score.** `human_en_1` carries a
  soft hyphen at every line break («interpreta¬ tions») and hard 80-column wraps, which no
  model can emit; removing all of it changes the verdict by at most 5 points on one judge.
  The floor is measuring the writing.
- **`opus` is nearly a constant on generated text** (72 on eleven of the fourteen
  generated readings in this session) and 3-8 on human prose. It separates; it does not
  rank. `gpt-5.6-terra` is the most consistent discriminator in both languages.

## 5. Where this leaves the gap

Clean, contemporary, published human prose reads **8–22** to the worst of four judges.
Everything this pipeline produces reads **78–95**. The gap is ~65 points, reproducible to
±3, and untouched by every prompt-level intervention tried on 2026-08-25/26.

Six interventions, six nulls: the seven rules, the rhythm quotas, the apparatus block,
the documentary-density quota, halving the prompt, and swapping sonnet for opus. Every
arm lands between 88 and 95. **On this task the score is not a function of the prompt or
of the model**, which is a different claim from the 2026-08-24 table and supersedes it:
that table ranked models on standalone 50-word prompts, and the ranking does not survive
a real dossier-grounded section.

Two practical consequences.

**The default threshold is unreachable.** `--threshold 25` against a floor of 78 means
`humanize()` runs its four rounds, fails, and returns the best text it saw, every single
run. That is four rewrite passes plus twelve judge calls per article buying a number
that never passes. Either the gate is dropped to a *relative* one (worst judge no worse
than the drafted text, i.e. use the loop to avoid regressions rather than to reach a
target), or `--detector-rounds` is set to 0 and the LLM judges are read as a report
instead of a gate. This is a product decision and has been left alone.

**What has not been tried**, and is the only family of idea left with a mechanism behind
it: building sections around long verbatim source passages rather than around a plan, so
the page carries human sentences rather than paraphrase; and dropping the completeness
requirement, so a section stops covering everything its `contenido` lists. Both change
what is on the page rather than how the sentences are shaped, which is the only axis
that has ever separated the two distributions. `--drafter pro` (`AW_BORRADOR=pro`) is
in place either way: `draft()` had `llm.FLASH` hardcoded and there was no way to spend
PRO on the sections at all.

## 6. The rewrite loop, with and without the stylometric floors

`bench_reescritura.py`, over `guided_sonnet_es` (worst judge 90), two rounds each way,
rewriter = sonnet:

| | ronda 1 | ronda 2 |
|---|---|---|
| feeding `issues` (the floors included — what ships) | 88 | **72** |
| feeding `issues_texto` (floors dropped) | 88 | 88 |

One text per arm, so this settles nothing on its own — but it is a *direct* test and the
−0.85 correlation is observational, so the old behaviour stands. `local_score()` returns
both lists; `humanize()` still feeds `issues`. Note what the 72 is made of: opus 72,
sonnet 20, terra 68, kimi 10 — two judges reading a rewritten machine text as broadly
human, which is the only time that has happened in this session and is worth reproducing
before anything is built on it.

Also worth recording: the rewrite loop is the **only** thing measured this session that
moved the number at all (90 → 72 on one text). Six prompt- and model-level interventions
moved it by zero. If there is a lever left, it is in `_rewrite`, not in `SECTION_PROMPT`.


---

## Flash bake-off (2026-09-05): glm-5.3-flash vs the defaults

`flash_bench.py` drafts the same PLAIN_ES/PLAIN_EN prompts with each candidate on
hyper and scores them with `local_score` + kimi-k3 (glm-5.2 was measured too but
dropped from the table: it rates its own family's output 98 across the board).

| modelo | seg/prompt | local ES | local EN | juez ES | juez EN |
|---|---|---|---|---|---|
| deepseek-v4-flash | 13 | 78.6 | 37.3 | 96 | 96 |
| deepseek-v4-pro (actual FLASH) | 29 | 86.3 | 37.5 | 94 | 93 |
| glm-5.3-flash | 163 | 84.2 | **0** | 94 | **18** |
| qwen3.8-flash | 162 | 71.5 | 48.1 | 90 | 86 |

Verdict: **not the default yet.** glm-5.3-flash's English output is the only text in
either run that crossed the human floor (local 0, judge 18), but two caveats keep it
out of PRO/FLASH:

1. **~12× slower than deepseek** (163s vs 13s per 1200-word draft; qwen3.8-flash the
   same cost), and drafting is the highest-volume call in the pipeline.
2. Its Spanish output is unremarkable (local 84.2, judge 94 — same as everything
   else). One language winning is n=1.

Also operational: it is a reasoning model — small `max_tokens` caps return empty
`content` with the answer stuck in `reasoning_content`. Any caller that sets a tight
cap must raise it (flash_bench uses 16384).

Registered everywhere (`_HYPER_MODELS`, opencode.json, ai_config_hyper.json,
calibre providers, book-watch live listing); switch it in with
`AW_MODELS=hyper:qwen3.8-max/glm-5.3-flash` or `--flash glm-5.3-flash` when speed
stops mattering more than the EN win.
