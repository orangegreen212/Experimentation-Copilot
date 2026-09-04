# Experiment Decision Policies

Original summary of general, publicly-taught experimentation decision
practice. These are interpretation policies for a human/LLM reading
ALREADY-COMPUTED statistics (p-values, confidence intervals, effect
sizes, power) — they are not a substitute for running the numbers, and
no policy here should be used to derive a statistic that wasn't
computed deterministically upstream.

## SHIP Policy

<!-- category: decision_policy; concept: ship; document_type: decision_rule; priority: high -->
A result is a reasonable candidate for a GO recommendation when all of
the following hold: the experiment is VALID (no SRM, no critical
data-quality failure, no conflicting variant assignment), the primary
metric shows a statistically significant effect, the effect moves in
the intended direction, the effect size clears practical significance
(it's large enough to matter to the business, not just large enough
to be detectable), and guardrail metrics have not regressed beyond
their acceptable threshold. A positive, significant result that fails
any one of these conditions does not automatically warrant a GO — see
the other policies below.

## DO NOT SHIP (NO-GO) Policy

<!-- category: decision_policy; concept: no_go; document_type: decision_rule; priority: high -->
NO-GO applies when the experiment is valid but the evidence does not
support shipping: the primary metric shows no statistically
significant effect, or it is significant but the variant is worse than
control, or it is significant and better but the effect size does not
clear practical significance. A statistically significant negative
result is a clear NO-GO regardless of guardrail status. Do not upgrade
a NO-GO to caution or ship on the basis of a secondary metric alone —
secondary metrics inform interpretation, they do not override the
primary metric's gate.

## GO WITH CAUTION Policy

<!-- category: decision_policy; concept: go_with_caution; document_type: decision_rule; priority: high -->
GO WITH CAUTION applies when the primary metric result is positive and
statistically significant but something else in the picture is mixed
or unresolved — most commonly a guardrail metric has regressed, or
practical significance could not be established from available
threshold information (a genuine indeterminacy in the data, not simply
"no pre-registered target was given"). This is a conditional
recommendation: it should be paired with a specific reason (which
guardrail regressed, and by how much) rather than issued as a vague
hedge. Whether "caution" ultimately means shipping with monitoring or
holding off is a severity judgment for the reader, informed by how
large the guardrail regression is relative to its own threshold.

## INVALID Policy

<!-- category: decision_policy; concept: invalid; document_type: decision_rule; priority: high -->
INVALID applies when a critical validity check fails: Sample Ratio
Mismatch (SRM), a critical data-quality failure (e.g. outlier
contamination severe enough to be flagged critical), or conflicting
variant assignment (the same user recorded in more than one variant).
When any of these hold, no ship/no-ship recommendation can be made
from the data, no matter how significant or large the observed effect
looks — a broken randomization or assignment mechanism can fabricate
an arbitrary effect that has nothing to do with the actual variants
being compared. The correct action is to fix the underlying pipeline
issue and rerun the experiment, not to interpret the numbers as-is.
Stop the analysis rather than attempting to salvage a recommendation.

## INSUFFICIENT EVIDENCE / UNDERPOWERED Policy

<!-- category: decision_policy; concept: underpowered; document_type: decision_rule; priority: high -->
This is the single most commonly confused policy, so it is stated
explicitly: "no statistically significant effect was found" and
"there is insufficient evidence to establish an effect" are NOT the
same claim, and an underpowered experiment must never be reported as
if it demonstrated "the treatment has no effect." A non-significant
result from a study with low statistical power (achieved power well
below what was targeted for the analysis) is a study that could not reliably detect
even a real, business-relevant effect if one existed — the null result
is uninformative about whether that effect is present, not proof of
its absence. Absence of evidence is not evidence of absence. The
correct framing is: "no effect was detected, and this experiment did
not have enough statistical power to rule one out" — never "the
treatment does not work." The appropriate action for an underpowered
null result is typically to collect more data (larger sample, longer
duration) rather than to conclude the experiment and reject the
change. A well-powered null result, by contrast, IS informative: if
achieved power is at or above target and the result is still not
significant, that is real (if still probabilistic) evidence against a
meaningful effect.

## Conflicting Variant Assignment

<!-- category: validity; concept: conflicting_assignment; document_type: methodology; priority: high -->
Conflicting variant assignment occurs when the same user (or unit of
randomization) is recorded as belonging to more than one variant — for
example, seen in both control and treatment across sessions due to a
bug in the assignment/logging pipeline, a shared device, or an
identifier collision. This breaks the fundamental guarantee that each
unit's outcome reflects exactly one treatment condition, and it can
bias the estimated effect in either direction depending on how the
overlap correlates with the outcome. Like SRM, this is a validity
failure, not a statistical one — it should stop the analysis (INVALID)
rather than be adjusted for post hoc, because the direction and size
of the resulting bias generally cannot be estimated from the same
compromised data.

## Experiment Contamination

<!-- category: validity; concept: contamination; document_type: methodology; priority: medium -->
Contamination is when treatment "leaks" into the control group (or
vice versa) through a channel outside the randomization — shared
households, social sharing of a feature, caching/CDN effects that
serve the wrong variant, or organizational processes that apply a
policy change to everyone regardless of assigned group. Contamination
dilutes the measured effect toward zero (making a real effect look
smaller or non-significant) rather than fabricating a false positive,
which is what makes it easy to miss: the experiment can look
statistically clean while still understating the true effect. Suspect
contamination when a plausible sharing/network channel exists between
variants and the observed effect is smaller than product intuition or
a pilot would suggest.

## Missing Data

<!-- category: validity; concept: missing_data; document_type: methodology; priority: medium -->
Missing outcome data (e.g. users who didn't complete a funnel step, or
logging gaps for one variant) is only safe to ignore when it's missing
completely at random — with roughly equal, unrelated-to-treatment rates
across variants. If the missingness rate itself differs meaningfully
between control and treatment (a form of differential attrition), that
difference is itself worth treating with the same suspicion as SRM:
the treatment may be causing certain users to drop out of measurement
entirely, and analyzing only the users who remained can produce a
badly biased effect estimate even when the remaining data looks
statistically clean.
