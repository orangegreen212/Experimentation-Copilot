# Experimentation at Scale

Original summary of publicly-discussed practices from large-scale
experimentation platforms (Booking.com-style). General industry
knowledge, not reproduced from any specific source.

## Guardrail Metrics

Beyond the primary metric, large platforms track a small set of
guardrail metrics on every experiment (latency, error rate, revenue
per user) regardless of what the experiment is nominally about. A
variant that wins on the primary metric but degrades a guardrail is
not an automatic ship — the guardrail regression needs its own
sign-off. This catches the common failure mode of "conversion went up
because the page got faster to load due to a bug that also broke
something else."

## Experimentation Velocity vs. Statistical Rigor

Running thousands of experiments per year creates pressure to move
fast, which trades off against rigor (larger minimum sample sizes,
longer runtimes, more guardrail checks). A common resolution is
tiering experiments by blast radius: small UI tweaks get lighter
review, pricing or checkout-flow changes get the full guardrail suite
and a mandatory minimum runtime.

## Holdout Groups

A holdout is a group of users deliberately excluded from a shipped
change (or from a whole category of changes) over a longer period, to
measure the cumulative long-term effect of many individually-small
wins. Individually significant experiments sometimes have zero or
even negative combined long-term effect (e.g. from ad-fatigue-style
user annoyance) — holdouts are the main tool for catching that.

## Multiple Comparisons

Running many experiments (or checking many metrics within one
experiment) inflates the false-positive rate — at a 5% significance
threshold, testing 20 independent metrics yields roughly one false
positive by chance alone. Platforms mitigate this with a single
pre-declared OEC per experiment (see the A/B testing fundamentals
doc), and by treating secondary/exploratory metrics as
hypothesis-generating rather than decision-making.
