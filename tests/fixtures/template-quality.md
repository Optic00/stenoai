# Built-in template quality cases

`template-quality.json` contains synthetic conversations for the four editable
built-ins, each in English and German. No recordings or personal meeting content
are used. `expectations` are a human review rubric, not substring tests.

## Splits

Every case has a `split`:

- `development` cases were read while the prompts were written and revised, and
  the regressions below were found on them. A good result on them only shows
  that known failures were addressed; it does not show that the prompts
  generalise.
- `held-out` cases were added on 2026-09-16, after the current prompt revision.
  They exercise the same rules with different situations and wording, no prompt
  text was taken from them, and no model has been run on them yet. They were
  written with knowledge of the prompt rules, so they are held out from tuning,
  not an independent test of which rules matter.

Keep the held-out split honest:

- Freeze the prompt change before generating held-out outputs.
- Never copy names, nouns, or phrases from any case into a shipped prompt. A
  prompt rule has to describe the general behaviour.
- If a held-out failure leads to a prompt change, move that case to
  `development` in the same change and add a new held-out case for the rule.
- Report development and held-out results separately.

## Running a comparison

A comparison is only production-equivalent when it sends what the app sends.
Compare the current and proposed prompts on the same host, with the same Ollama
version and model:

- **Model tag:** the tag the app sends, `src.config.resolve_runtime_tag(model)`
  for the configured summary model, not the canonical id. For the default
  `gemma4:e2b-it-qat` this is `gemma4:e2b-nvfp4` on Apple Silicon and
  `gemma4:e2b-it-qat` elsewhere.
- **Prompt:** `OllamaSummarizer._create_template_report_prompt(transcript,
  prompt, language)` with the case's `language` and no notes.
- **Request:** a streamed `/api/chat` call with one user message and
  `think=false`, as in `OllamaSummarizer._stream_direct`.
- **Options:** exactly `OllamaSummarizer._ollama_options()`, which is
  `{"num_ctx": resolve_num_ctx(model)}` (32768 for the Gemma 4 QAT models). Do
  not set `temperature`, `seed`, `top_p`, `top_k`, or `num_predict`: production
  leaves them at the model defaults, so a pinned seed or output cap measures a
  different configuration.
- **Repeats:** without a seed, outputs can differ between runs. Generate at
  least three outputs per case and prompt version, and review all of them. A
  blocking failure in any run counts.

Record the date, host OS and architecture, Ollama version, resolved model tag,
options, the commit of each prompt version, and the raw outputs keyed by case
`id` with the evaluation. A mocked completion or a matching heading is not a
quality assertion.

## Review

Check factual support, retained qualifications and disagreements, correct
speakers and action owners, requested versus accepted actions, output language,
and repetition. Read the transcript as well as the listed expectations. A
shorter report alone is not a pass. An omitted material disagreement, invented
commitment, or reassigned action should block release of the prompt change.

## Initial local comparison, 2026-09-05 (not production-equivalent)

This run used `gemma4:e2b-it-qat` with temperature 0, seed 42, `num_ctx` 8192,
and an output limit (`num_predict`) of 1100. The app's Ollama report request
sets no temperature, seed, or output limit, and sends `num_ctx` 32768 for this
model. The record does not name the host, so it is unknown whether the model
tag matched the one the app sends there. The run covered only the eight
`development` cases, before their speaker names were replaced, and compared the
previous shipped prompts with a proposed revision that predates the current
prompts. Its results are useful leads, not evidence about production quality.

All 16 before/after calls completed without truncation. Reports were generally
more compact, and the demo outputs separated the demonstrated capability from
the planned and the unsupported ones. It found four regressions:

- Both demo outputs assigned the requested follow-up documents to the vendor,
  although the transcript leaves the owner unconfirmed.
- The German 1:1 omitted the disagreement about priorities, which the baseline
  retained.
- The English standup omitted the accepted offer of help, which the baseline
  retained; the German standup repeated the unowned item.
- The English sales report omitted the explicit absence of a purchase
  commitment.

The current prompts add general rules for these failure classes: an owner only
after explicit acceptance, a section that gives each person's position in a
disagreement, accepted help listed under the person who agreed and unowned work
listed once, and a commitment status that is always reported. They have not
been evaluated with a model, and no outputs for them exist. They should not ship
on green unit or CRUD tests alone; run the comparison above on both splits
first. Existing user overrides and the locked Standard template are unchanged.
