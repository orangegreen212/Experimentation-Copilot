# Practical Experimentation at Scale

Original summary of publicly-documented experimentation practices
described in Airbnb's engineering writing on their experimentation
platform (Airbnb-style guidance). General statistical/product
methodology, not reproduced text. Sections here deliberately avoid
overlapping with topics already covered by kohavi.md (Novelty/Primacy)
and booking.md (Guardrail Metrics, Multiple Comparisons) — see those
files for those topics.

## Segment-Level Heterogeneity

An experiment's headline effect is an average across all users in the
sample, and an average near zero can hide a meaningfully positive
effect for one segment cancelled out by a meaningfully negative effect
for another (e.g. new vs. returning users, or mobile vs. desktop).
Slicing the result by pre-registered segments — decided before looking
at the data, to avoid multiple-comparisons false positives from
post-hoc slicing — can reveal whether a "no effect" result is actually
a "different effects for different users" result, which changes the
recommendation from "don't ship" to "ship with targeting."

## Iteration Speed vs. Experiment Rigor

Running many small experiments quickly compounds product improvements
faster than running fewer, larger, more rigorously-designed ones — but
only if each individual test still respects a minimum sample size and
runtime; shipping a test that ran for two days on low traffic just to
keep velocity up produces noise dressed up as a decision. A practical
balance many high-velocity teams use is a shared minimum-runtime floor
(commonly one full week, to average out day-of-week effects) below
which a test is not read as conclusive regardless of how the p-value
happens to land on a given day. This is the core tension teams
balancing shipping speed against rigorous testing have to resolve
explicitly, not leave implicit.
