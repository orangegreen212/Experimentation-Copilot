# Statistical Interpretation Guidance

Original summary of general, publicly-taught statistical interpretation
concepts for reading already-computed experiment results. Nothing here
computes a statistic — it explains how to read one that a deterministic
Python routine already produced.

## What a p-value Means

<!-- category: statistics; concept: p_value; document_type: methodology; priority: high -->
The p-value is the probability of observing a difference at least as
extreme as the one measured, IF the null hypothesis (no true
difference between variants) were actually true. A small p-value
(conventionally below the significance threshold, e.g. 0.05) means the
observed data would be unlikely under "no effect," which is evidence
against the null — not proof the treatment works, and not the
probability that the null hypothesis is true. A p-value says nothing
about effect size: a tiny, practically meaningless effect can still
produce a very small p-value with enough sample size, which is exactly
why statistical significance and practical significance must be
checked separately.

## Confidence Intervals

<!-- category: statistics; concept: confidence_interval; document_type: methodology; priority: high -->
A confidence interval (CI) gives a range of plausible values for the
true effect, not just a single point estimate. A 95% CI means that if
the experiment were repeated many times, about 95% of the intervals
constructed this way would contain the true effect. A CI that excludes
zero is consistent with (though not identical to) a statistically
significant result at the corresponding threshold. The WIDTH of the CI
matters as much as whether it excludes zero: a wide interval spanning
both practically trivial and practically large effects means the
result, even if significant, is too imprecise to act on with
confidence, and more data would narrow it.

## Effect Size

<!-- category: statistics; concept: effect_size; document_type: methodology; priority: high -->
Effect size is the magnitude of an observed difference — how large
the shift between variants actually is — independent of whether that
shift is statistically distinguishable from noise. A result can be
statistically significant with a negligible magnitude (common with
very large user counts) or fail to reach significance despite a
respectable magnitude (common with small or underpowered
experiments). Reporting a p-value alone without also reporting how
large the shift actually was, or reporting the shift without its
uncertainty range, is an incomplete picture — a decision should weigh
both together, not treat "significant" as a synonym for
"big."

## Relative vs. Absolute Lift

<!-- category: statistics; concept: lift; document_type: methodology; priority: medium -->
Absolute lift is the raw difference between variants (e.g. treatment
conversion rate minus control conversion rate, in percentage points).
Relative lift expresses that same difference as a percentage of the
baseline (e.g. a 0.5 percentage-point absolute lift on a 2% baseline is
a 25% relative lift). Relative lift can make a small absolute change
sound dramatic on a low baseline, and understate a large absolute
change on a high baseline — always check which framing is being used,
and prefer reporting both together rather than relative lift alone
when the baseline itself is small or unstable.

## Statistical Power and Type I / Type II Errors

<!-- category: statistics; concept: power; document_type: methodology; priority: high -->
A Type I error is a false positive: concluding there's an effect when
there truly isn't one — its rate is controlled by the significance
threshold (alpha, typically 5%). A Type II error is a false negative:
failing to detect a real effect that does exist — its rate (beta)
depends on sample size, the true effect size, and variance in the
data. Statistical power is 1 minus the Type II error rate — the
probability of correctly detecting a real effect of a given size, if
one exists. Low power means a high chance of a Type II error: missing
a real, meaningful effect and wrongly treating it as "no effect."
Power should ideally be checked before running an experiment (to size
the sample correctly) and always considered when interpreting a
non-significant result after the fact.

## Post-Hoc vs. Pre-Registered MDE

<!-- category: statistics; concept: mde; document_type: methodology; priority: high -->
MDE can be computed two different ways, and the report should never
conflate them. Pre-registered MDE is decided BEFORE the experiment
runs, from a target sample size and power — this describes what
business-relevant lift is worth chasing. Post-hoc MDE is computed
AFTER the experiment, from the sample size actually observed — this
describes only what the study, as run, COULD have picked up, not a
business threshold anyone decided on in advance. A large post-hoc MDE
relative to a plausible real lift is a warning sign that the study was
structurally unable to pick up a meaningful change, which means
treating its non-significant result as "the treatment does nothing"
would be a mistake — see the Insufficient Evidence / Underpowered
policy.

## Practical Significance

<!-- category: statistics; concept: practical_significance; document_type: methodology; priority: high -->
Practical significance asks whether an observed, statistically
significant effect is large enough to be worth the cost of shipping —
engineering effort, complexity, risk — regardless of whether it's
statistically distinguishable from zero. A statistically significant
result that is not practically significant should generally not be
shipped on its own merits; the "Statistical Significance Is Not
Practical Significance" methodology note applies directly here. When
no pre-registered practical-significance threshold exists, this should
be reported as "not assessed" rather than silently treated as either
pass or fail.
