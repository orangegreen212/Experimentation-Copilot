# Advanced Experiment Methodology

Original summary of general, publicly-taught experimentation methodology
for more advanced analysis situations.

## Underpowered Experiments — Analysis Implications

<!-- category: statistics; concept: underpowered; document_type: methodology; priority: high -->
An underpowered experiment is one whose sample size was too small (or
whose runtime too short) to reliably detect the effect size that
would actually matter for the decision — achieved power falls well
below the target (commonly 80%). Beyond the ship/no-ship framing (see
the Decision Policies document), underpowered analysis has practical
consequences: segment-level and secondary-metric results will be even
more underpowered than the primary metric (smaller subgroup, same or
worse noise), so a "significant" subgroup finding in an
already-underpowered overall experiment deserves extra skepticism, not
extra confidence, and should be treated as hypothesis-generating for a
future, properly-sized test rather than as a standalone finding.

## Multiple Comparisons in Segmentation Analysis

<!-- category: statistics; concept: multiple_comparisons_segmentation; document_type: methodology; priority: high -->
Segmentation analysis (breaking the primary result down by device,
geography, new vs. returning user, etc.) multiplies the number of
statistical tests being run against the same experiment. At a 5%
significance threshold, testing 20 independent segments yields roughly
one "significant" result by chance alone even if the treatment truly
has no effect in any segment — this is the multiple comparisons
problem applied specifically to segment cuts. A significant subgroup
result found by scanning many segments (rather than one segment
specified as a hypothesis before looking at the data) should be
treated as exploratory, reported with a note about how many segments
were examined, and ideally validated with a correction (e.g.
Bonferroni, Benjamini-Hochberg) or a dedicated follow-up experiment —
not presented with the same confidence as the pre-registered primary
result.

## Subgroup Sample Size

<!-- category: statistics; concept: subgroup_sample_size; document_type: methodology; priority: medium -->
Every segment cut divides the overall sample, so a segment's effective
sample size can be a small fraction of the total — a segment
representing 10% of traffic has roughly a tenth of the primary
analysis's statistical power, often falling well under the sample size
needed to detect even the same effect size the overall test was
powered for. Before treating a segment result as informative, check
whether that segment's own sample size come anywhere close to
supporting the same MDE as the overall analysis; if not, a null
segment result is uninformative for the same reason an underpowered
overall experiment is (see the Insufficient Evidence / Underpowered
policy), and a "significant" segment result with a very small n
deserves the multiple-comparisons skepticism above.

## CUPED Interaction with Segmentation and Reporting

<!-- category: statistics; concept: cuped_segmentation; document_type: methodology; priority: medium -->
When a CUPED-adjusted primary result is broken down by segment, each
segment's adjustment coefficient should ideally be re-fit within that
segment rather than reused from the overall population — a covariate
relationship strong enough to shrink variance overall can differ, or
even reverse sign, within a subgroup. A report that shows an
overall CUPED-adjusted number next to unadjusted segment cuts (or vice
versa) without labeling which is which risks the reader comparing two
figures that aren't on the same footing. As a variance-reduction
technique, an adjusted estimate should never be interpreted as a
different treatment effect than the unadjusted one — only as the same
effect measured more precisely.

## Peeking, Stopping Rules, and Sequential Testing

<!-- category: statistics; concept: peeking; document_type: methodology; priority: high -->
"Peeking" is checking a fixed-horizon experiment's results repeatedly
before its planned sample size is reached and stopping as soon as the
result looks significant. This inflates the true false-positive rate
well above the nominal significance threshold, because each additional
look is another chance for random noise to cross the threshold — an
experiment peeked at ten times can have a false-positive rate several
times higher than the stated 5%, even though each individual p-value
was computed correctly. The fix is either to commit to a fixed sample
size/duration decided in advance and only look once at the end, or to
use a sequential testing method (e.g. always-valid p-values, group
sequential designs) explicitly designed to control the false-positive
rate under repeated looks. A p-value from a non-sequential test that
was peeked at repeatedly should not be reported as if it controls
error at the stated significance level.

## Experiment Duration and Stopping Rules

<!-- category: methodology; concept: stopping_rules; document_type: methodology; priority: medium -->
A pre-registered stopping rule (a fixed sample size or a fixed
duration, ideally covering at least one full business cycle — e.g. a
full week to capture weekday/weekend behavior) should be decided
before the experiment launches and adhered to, independent of how the
results look partway through. Stopping early because a result looks
favorable (a form of peeking) or extending an experiment past its
planned end because the result is "almost significant" both distort
the statistical guarantees the test was designed to provide, and
should be flagged as a limitation when they occur.
