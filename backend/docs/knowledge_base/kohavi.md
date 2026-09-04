# A/B Testing Fundamentals

Original summary of widely-taught online controlled experimentation
concepts (Kohavi-style trustworthy experimentation practice). No text
is reproduced from any specific book — these are general, public
statistical and methodological concepts.

## Overall Evaluation Criterion (OEC)

An experiment needs one primary metric decided BEFORE the experiment
starts — the Overall Evaluation Criterion. Picking the metric after
seeing results (or picking whichever metric looks best) is a form of
p-hacking and invalidates the statistical guarantees of the test.
Guardrail metrics (e.g. page load time, error rate) are monitored
alongside the OEC to catch unintended harm even when the OEC improves.

## Sample Ratio Mismatch (SRM)

SRM is when the observed traffic split between variants doesn't match
the intended allocation (e.g. 58/42 instead of 50/50). It's detected
with a chi-square goodness-of-fit test. SRM is a critical trust
signal: it usually means the randomization or logging pipeline is
broken, and any effect estimate from an SRM-failing experiment should
be discarded rather than interpreted, no matter how significant it
looks — a broken randomization mechanism can fabricate arbitrary
effects that have nothing to do with the actual variant.

## Twyman's Law

"Any figure that looks interesting or different is usually wrong." A
surprisingly large effect size is more often a bug (tracking error,
SRM, a caching issue) than a real breakthrough. Large effects deserve
extra scrutiny, not extra excitement.

## Novelty and Primacy Effects

A new feature's effect can look artificially strong in the first days
(users notice change and click out of curiosity — novelty effect) or
artificially weak (loyal users resist a changed workflow — primacy
effect). Both fade over time, so short experiments risk mistaking a
transient effect for the steady-state effect. Running an experiment
long enough to observe at least one full behavioral cycle mitigates
this.

## Minimum Detectable Effect (MDE) and Power

Power analysis answers "how small an effect could this sample size
reliably detect?" before or after running an experiment. An
underpowered experiment finding no significant result tells you
nothing about whether a real (smaller) effect exists — absence of
evidence isn't evidence of absence. The MDE should be decided before
running the experiment, informed by what effect size is actually
worth shipping for.
