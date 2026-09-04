# Experiment Fundamentals

Original summary of general, publicly-taught A/B testing fundamentals.

## Randomization and Control vs. Treatment

<!-- category: fundamentals; concept: randomization; document_type: methodology; priority: high -->
A/B testing works by randomly assigning units (usually users) to a
control group (the existing experience) and one or more treatment
groups (the new experience being tested), so that — on average — the
only systematic difference between groups is the thing being tested.
Randomization is what licenses a causal claim ("the treatment caused
the observed difference") rather than a merely correlational one. If
assignment isn't actually random, or breaks down partway through
(a rollout that isn't randomized, a feature flag that changes
mid-experiment, users who can opt in/out non-randomly), any resulting
effect estimate can be confounded by whatever caused the non-random
split.

## Secondary Metrics

<!-- category: fundamentals; concept: metric_roles; document_type: methodology; priority: medium -->
Secondary metrics are tracked alongside the primary metric (the
Overall Evaluation Criterion — see kohavi.md) to provide supporting
diagnostic context: they help explain WHY the primary metric moved, or
didn't, without being what the ship/no-ship decision itself gates on.
A significant, favorable secondary result should never override a
non-significant or unfavorable primary result — that would be
equivalent to silently swapping in a different OEC after seeing the
data, the exact p-hacking risk the OEC concept exists to prevent. They
are most useful for building a causal story around the primary
finding ("checkout conversion rose because the new layout reduced cart
abandonment specifically"), never for arguing around a primary metric
that didn't move.

## Sample Ratio Mismatch (SRM) — Fundamentals

<!-- category: validity; concept: srm; document_type: methodology; priority: high -->
SRM is when the observed traffic split between variants doesn't match
the intended allocation (e.g. 58/42 instead of 50/50), detected with a
chi-square goodness-of-fit test against the intended ratio. SRM is a
critical trust signal, not a statistical nuance: it usually means the
randomization or logging pipeline is broken somewhere (bot traffic
hitting one variant differently, a redirect bug, differential
client-side logging failures), and any effect estimate from an
SRM-failing experiment should be discarded entirely rather than
interpreted with caveats — a broken randomization mechanism can
fabricate an arbitrary effect that has nothing to do with the actual
variant being tested.

## Randomization Problems Beyond SRM

<!-- category: validity; concept: randomization_problems; document_type: methodology; priority: medium -->
Not every randomization failure shows up as SRM. Interference between
units (a treatment given to one user affecting outcomes for another,
e.g. via a social feed or shared marketplace inventory), unit-of-
randomization mismatches (randomizing by session when the metric is
measured per-user), and time-based confounds (running treatment and
control in different calendar periods rather than concurrently) can
all violate the assumptions randomization is supposed to guarantee
without ever producing a detectable SRM. These are generally caught by
experiment design review rather than by a post-hoc statistical test,
which is why documenting the unit of randomization and concurrency of
exposure matters as much as running the chi-square check.
