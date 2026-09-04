# Variance Reduction Techniques

Original summary of publicly-documented variance reduction methods
used in large-scale experimentation (Netflix/Microsoft-style CUPED
literature). General statistical methodology, not reproduced text.

## CUPED (Controlled-experiment Using Pre-Experiment Data)

CUPED reduces the variance of a metric by adjusting for a
pre-experiment covariate correlated with the outcome — most commonly,
the same metric measured before the experiment started. The
adjustment is `Y_cuped = Y - theta * (X - mean(X))`, where theta is
chosen to minimize the adjusted variance (`theta = Cov(Y,X) / Var(X)`).
Because it doesn't change the expected value of the metric, only its
variance, CUPED-adjusted results remain an unbiased estimate of the
treatment effect — it just makes the confidence interval tighter for
the same sample size, meaning experiments can either run shorter or
detect smaller effects than they otherwise could.

CUPED only helps when the covariate is meaningfully correlated with
the outcome — for a genuinely new metric with no pre-experiment
history (a brand-new feature's usage, for instance), there's no valid
covariate and CUPED should be skipped rather than applied to an
unrelated proxy.

## Sequential Testing Considerations

Checking a p-value every day and stopping as soon as it crosses 0.05
("peeking") inflates the false-positive rate far above the nominal
alpha — the true false-positive rate of naive repeated peeking can
exceed 30% even when the null hypothesis is true. Valid approaches are
either committing to a fixed sample size decided in advance, or using
a sequential testing method (e.g. group sequential designs, or
always-valid p-values) explicitly designed to control the false
positive rate under repeated looks.

## Bootstrap Confidence Intervals

For metrics whose sampling distribution doesn't have a convenient
closed-form confidence interval (e.g. the median, or complex ratio
metrics), a percentile bootstrap — resampling the observed data with
replacement many times and taking the percentiles of the resulting
statistic's distribution — gives a distribution-free confidence
interval. It requires no normality assumption, at the cost of more
computation than a parametric interval.
