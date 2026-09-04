# Statistical Significance and Practical Significance

Original summary of publicly-documented statistical methodology
described in Optimizely's and Google's experimentation/statistics
guidance (Optimizely/Google-style). General statistical methodology,
not reproduced text. Sections here deliberately avoid overlapping with
topics already covered by booking.md (Multiple Comparisons) — see that
file for that topic.

## Statistical Significance Is Not Practical Significance

A result can be statistically significant (the p-value is below the
chosen alpha) while the observed change is too small to matter for the
business — a 0.1% relative lift measured with enough traffic can reach
p < 0.05 without being worth the engineering cost of shipping it.
Conversely, a promising-looking change can fail to reach significance
simply because the sample size was too small, not because the true
impact is zero. Two separate questions must both be answered before a
ship decision: is the change statistically distinguishable from noise
(significance), and is it, if real, large enough to justify acting on
(practical significance, typically judged against the pre-registered
minimum detectable effect).

## One-Sided vs. Two-Sided Tests

A two-sided test (the default and generally the safer choice) asks
whether the variant differs from control in either direction; a
one-sided test asks only whether it's better, at the cost of being
unable to detect and report a real negative effect on that metric. A
one-sided test should only be used when a large negative effect
genuinely would not change the ship decision differently than a null
result — which is rare — otherwise it can mask a regression the
guardrail metrics were supposed to catch.

## A/A Tests as a Sanity Check

Running an A/A test — splitting traffic into two groups that both get
the exact same experience — should show a significant difference only
about as often as the chosen alpha (roughly 1 in 20 times at
alpha=0.05). Running one periodically against the live experimentation
pipeline is a cheap way to catch instrumentation bugs (mis-triggered
tracking, a broken randomization unit) before they corrupt a real
experiment's result — a pipeline that fails its own A/A test can't be
trusted to correctly measure a real A/B test either.
