# Step81 training-only format-zero review

Owner-requested scoring change after the complete Step80 checkpoint. The
training remains paused until resumed; no earlier rewards are regraded.

- Enable `format_review_from_step: 81` in the continued six-domain config and
  select `--allow-format-review` with the complete Step80 snapshot. This saves
  an explicit scoring-condition boundary and retains F/B/Z, Adam, posterior,
  library, source cursor and historical evidence.
- First run the unchanged native evaluator. Only a native zero caused by a
  malformed final action, failed submission admission, or rejected terminal
  projection is eligible. A normally admitted wrong answer or failing executable
  solution does not get another chance. Positive native rewards are unchanged.
- The terminal-only judge is `lab-gpt-5.6-luna`, medium, through the existing
  Flowsteer durable route. It inspects the actual final answer/action and private
  reference, never generates a replacement answer or a corrected action. It
  must affirm both **format-only failure** and **correct complete content**.
  Approval sets training reward to 1 and binary success to true. This is not
  reported as an official native benchmark pass.
- Only the final candidate is reviewed; do not search earlier actions or
  reasoning for a favorable answer. ALFWorld remains governed by actual native
  environment success. IID/OOD evaluation remains unchanged.
- Store original reward/success/native metrics, review verdict and rationale,
  API identity and metering privately. Requests are journaled before dispatch;
  uncertain calls are not resent. Missing or malformed Judge responses are
  infrastructure failures, not negative labels.
- TTB consumes the effective terminal reward and posterior uses its effective
  success only for real skill calls. Original action tokens, parse errors,
  observation history, structure/admission/execution counts stay unchanged.
  There is no extra objective, skill-call reward or actor-side answer feedback.
- W&B receives only aggregates: effective reward/success, original native
  reward/success, review/approval counts, reward gain and review token counts.
  Step1–80 and Step81+ are distinct scoring segments, not an unchanged native
  accuracy curve. Raw tasks, references and review explanations are not uploaded.

This change deliberately changes the training reward condition. An LLM verdict
can be mistaken; the separate original-score series remains necessary.
