# Statistical Trustworthiness

Original summary of publicly-documented statistical rigor practices
(Microsoft ExP-style trustworthy experimentation). General statistical
knowledge, not reproduced text.

## Choosing Between Parametric and Non-Parametric Tests

Student's and Welch's t-tests assume the sampling distribution of the
mean is approximately normal — which, by the Central Limit Theorem,
holds even for skewed underlying data once the sample size is large
enough (a common rule of thumb is n >= 30 per arm). For smaller
samples, checking normality directly (e.g. with a Shapiro-Wilk test)
before choosing a t-test is safer; when normality doesn't hold, a
non-parametric test like Mann-Whitney U (which compares distributions
via rank rather than mean, and makes no distributional assumption)
avoids a potentially invalid p-value.

## Welch's t-test vs. Student's t-test

Student's t-test assumes both groups have equal variance; Welch's
t-test does not. Since Welch's test loses very little power when
variances genuinely are equal, and produces an invalid p-value when
they aren't, defaulting to Welch's test is standard modern practice —
there's rarely a good reason to use Student's t-test over it.

## Binary Metrics: Chi-square vs. Fisher's Exact Test

For comparing conversion rates (or any binary outcome) between two
groups, a chi-square test of independence on the 2x2 contingency table
is standard. Chi-square is a large-sample approximation, though — when
any expected cell count in the contingency table falls below about 5,
the approximation becomes unreliable, and Fisher's exact test (which
computes an exact p-value regardless of sample size) should be used
instead.

## Confidence Intervals Are Not Optional

A p-value alone answers "is there evidence of an effect," while a
confidence interval answers "how big could the effect plausibly be."
A statistically significant result with a wide confidence interval
spanning both a trivial and a huge effect size is much weaker evidence
for shipping than a significant result with a tight interval — the
p-value alone doesn't communicate this difference.
