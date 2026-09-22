# Qwen3.5-9B TriviaQA local-search protocol

## Scope

This is an independent, adapter-free diagnostic track for the frozen 128-case
SkillFlow TriviaQA IID panel. It does not replace or rescore the question-only
Direct-Qwen result, does not enter the TTB training path, and does not change the
method in `idea.tex`. The executable binding is
`configs/evaluation/qwen35_triviaqa_search_augmented.yaml`.

The serving route is the same frozen `Qwen/Qwen3.5-9B` base model, tokenizer,
chat template, non-thinking decoding profile, and seed as the direct-reference
TriviaQA run. No LoRA, adapter, skill, learned retriever, or answer-bearing
artifact is loaded.

## Frozen corpus construction

Each task has a private retrieval partition built before generation from only
its public question:

1. An ephemeral Codex `gpt-5.6-sol` process at `xhigh` effort receives one
   question through standard input. It emits exactly three Wikipedia queries
   and a 300–800-character related background note. It never receives the
   target or aliases and is instructed not to state a final answer.
2. The three queries are sent directly to the English MediaWiki API. At most
   four distinct pages are retained for one task. The page ID, revision ID,
   revision timestamp, title, and canonical URL are captured in the private
   provenance record.
3. Plain article text is split into 1,200-character passages with 150-character
   overlap and at most six passages per page. The Codex note is an additional
   passage.
4. Passages are indexed in a private SQLite FTS5 database. Retrieval is always
   constrained by task ID, returns at most five passages, and permits at most
   two hits from one source document.

Raw Wikipedia can naturally contain facts relevant to an answer. “Indirect”
means that neither the target nor accepted aliases are used to generate queries,
select pages, index passages, rank hits, or construct model prompts. Licensed
questions, targets, aliases, private notes, article text, retrieval traces, and
per-task results remain outside Git.

## Generation contract

The first Qwen turn contains the question but no retrieved text and must emit
exactly `Search: <query>`. After each search, the observation is appended to the
same trajectory. A later turn may emit another search or exactly
`Final answer: <short answer>`. There are at most three searches and four model
calls; after the search budget is exhausted, only a final answer is accepted.
Malformed commands and empty final answers are definitive candidate failures
and are not repaired or retried.

Only generation, retrieval, or scorer infrastructure failure is retryable, with
at most three attempts for one operation. An operation retry keeps the same
request and trajectory; it cannot regenerate a successful model response,
retry a candidate failure, or change the corpus. The final answer is scored
after generation with the unchanged official-alias TriviaQA normalization, EM,
and token F1 implementation.

## Reporting

The public result reports the frozen count, EM, F1, infrastructure failures,
invalid-candidate count, search-use rate, mean search count, empty-hit rate,
throughput, and wall time. It is labeled
`qwen35-triviaqa-search-augmented@1` and must be displayed separately from the
question-only baseline. A partial run is diagnostic and cannot emit a formal
score.
