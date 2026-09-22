# Qwen3.5-9B TriviaQA detailed local-search protocol

## Scope

`qwen35-triviaqa-search-augmented@2` is a separate answer-blind diagnostic over
the same frozen 128-case TriviaQA IID panel as the question-only and v1 search
tracks. It keeps the adapter-free Qwen3.5-9B model, tokenizer, chat template,
non-thinking decoding profile, official alias EM/F1 scorer, and seed unchanged.
It does not enter training or modify `idea.tex`.

The released RC record's passage preview is removed at its final `Question:`
boundary. Both corpus construction and model generation receive only that bare
public question; task-local retrieval is the only source of supporting context.

## Rich private corpus

For each public question, an ephemeral Codex `gpt-5.6-sol` process at `xhigh`
effort produces a 1,200–3,000-character answer-blind research dossier, exactly
eight distinct Wikipedia queries, and six to twelve research entities. The
evaluator target and accepted aliases are unavailable to this process. Direct
answer wording is rejected.

The eight queries are searched independently through the English MediaWiki API.
Page selection proceeds round-robin across query result ranks rather than
allowing the first query to consume the page budget. At most twelve distinct
pages are associated with one task. Full plain-text articles are divided into
1,600-character passages with 200-character overlap and up to sixteen passages
per page. Dossier chunks and an entity passage are indexed alongside Wikipedia.
After generation and article retrieval, a private final pass removes every
literal official target alias from task-local titles and passage text. Labels
are used only by this removal pass; they are unavailable to Codex, Wikipedia
query selection, retrieval ranking, and Qwen generation.

The private SQLite FTS5 index is partitioned by task ID. A model search returns
at most eight match-centered snippets, with no more than two from one source
document. All corpus text, questions, targets, aliases, plans, provenance,
traces, and per-task predictions remain outside Git.

## Generation and scoring

The model must search before answering. Every generation is instructed to emit
one line containing `Search: <query>` or `Final answer: <short answer>`. The
parser extracts exactly one unambiguous labeled command line even if the model
surrounds it with prose; unlabeled responses, multiple commands, and mixed
search/final commands remain definitive candidate failures and are neither
repaired nor retried while a search is permitted. The richer corpus makes one
search call available. After that budget is exhausted, the existing
`short-answer-last-line@1` rule accepts the last non-empty output line as the
final candidate when no labeled command is present. This fallback is never
enabled before retrieval. Only infrastructure operations may be attempted up
to three times.

The unchanged official TriviaQA normalizer computes EM and token F1 after
generation. The result is reported separately from v1 and question-only scores;
changing the retrieval corpus and prompt makes it a diagnostic, not a more exact
reproduction of the paper-direct protocol.
