"""
Sample Ratio Mismatch (SRM) check — Stage 3.

Tests whether observed traffic allocation between arms matches the
expected ratio (usually 50/50), using a chi-square goodness-of-fit
test. A significant deviation (p < alpha) suggests the randomization
or assignment mechanism is broken and downstream results shouldn't be
trusted — this is exactly what mock-data.ts's LOW_CONFIDENCE_REPORT
demonstrates (58/42 split, p < 0.001).

Pure function: (counts, expected_ratio) -> SRMResult. No DataFrame
handling here — that's the caller's job (quality_node in Stage 5+).
"""

from __future__ import annotations

from app.core.lazy_import import LazyModule

# Deferred: only needed once check_srm()/check_srm_multi_arm() run the
# chi-square test, not at import time — see app/core/lazy_import.py.
scipy_stats = LazyModule("scipy.stats")

from app.core.config import stats_thresholds
from app.schemas.quality import QualityCheck, SRMResult


def check_srm(
    observed_control: int,
    observed_variant: int,
    expected_ratio: float = 0.5,
    alpha: float | None = None,
) -> SRMResult:
    """
    Chi-square goodness-of-fit test for a 2-arm SRM check.

    `expected_ratio` is the expected share of the CONTROL arm (0.5 for
    a standard 50/50 split). `alpha` defaults to
    `stats_thresholds.srm_alpha` if not provided.
    """
    if observed_control < 0 or observed_variant < 0:
        raise ValueError("Observed counts must be non-negative.")
    total = observed_control + observed_variant
    if total == 0:
        raise ValueError("Cannot run SRM check with zero total observations.")
    if not (0.0 < expected_ratio < 1.0):
        raise ValueError("expected_ratio must be strictly between 0 and 1.")

    alpha = stats_thresholds.srm_alpha if alpha is None else alpha

    expected_control = total * expected_ratio
    expected_variant = total * (1 - expected_ratio)

    chi2_statistic, p_value = scipy_stats.chisquare(
        f_obs=[observed_control, observed_variant],
        f_exp=[expected_control, expected_variant],
    )

    passed = bool(p_value >= alpha)

    return SRMResult(
        observed_control=observed_control,
        observed_variant=observed_variant,
        expected_ratio=expected_ratio,
        chi2_statistic=float(chi2_statistic),
        p_value=float(p_value),
        passed=passed,
        alpha=alpha,
    )


def check_srm_multi_arm(observed_counts: list[int], alpha: float | None = None) -> SRMResult:
    """SRM check for k-arm experiments with equal expected allocation."""
    if len(observed_counts) < 2 or any(c < 0 for c in observed_counts) or sum(observed_counts) == 0:
        raise ValueError("At least two non-negative arm counts with a positive total are required.")
    alpha = stats_thresholds.srm_alpha if alpha is None else alpha
    total = sum(observed_counts)
    expected = [total / len(observed_counts)] * len(observed_counts)
    chi2_statistic, p_value = scipy_stats.chisquare(f_obs=observed_counts, f_exp=expected)
    return SRMResult(
        observed_control=observed_counts[0],
        observed_variant=sum(observed_counts[1:]),
        expected_ratio=1.0 / len(observed_counts),
        chi2_statistic=float(chi2_statistic),
        p_value=float(p_value),
        passed=bool(p_value >= alpha),
        arm_counts=list(observed_counts),
        alpha=alpha,
    )


def srm_result_to_quality_check(result: SRMResult) -> QualityCheck:
    """
    Fold an SRMResult into the frontend-facing QualityCheck row.

    Formatting (percentages, p-value display) happens here — this is
    still pure deterministic Python, not LLM territory, matching
    mock-data.ts's detail text style:
      "Observed split 50.1% / 49.9% vs expected 50/50 (p = 0.62)"
      "Observed split 58% / 42% vs expected 50/50 (p < 0.001) — FAILED"
    """
    p_display = "< 0.001" if result.p_value < 0.001 else f"= {result.p_value:.2f}"
    suffix = "" if result.passed else " — FAILED"

    threshold_note = f" [decision threshold: alpha = {result.alpha:g}]" if result.alpha else ""

    if result.arm_counts and len(result.arm_counts) > 2:
        total = sum(result.arm_counts)
        shares = "/".join(f"{100 * c / total:.1f}%" for c in result.arm_counts)
        expected = 100 / len(result.arm_counts)
        detail = (
            f"Observed {len(result.arm_counts)}-arm split {shares} "
            f"vs expected {expected:.1f}% per arm (p {p_display}){suffix}{threshold_note}"
        )
    else:
        total = result.observed_control + result.observed_variant
        control_pct = 100 * result.observed_control / total
        variant_pct = 100 * result.observed_variant / total
        expected_control_pct = 100 * result.expected_ratio
        expected_variant_pct = 100 - expected_control_pct
        detail = (
            f"Observed split {control_pct:.1f}% / {variant_pct:.1f}% "
            f"vs expected {expected_control_pct:.0f}/{expected_variant_pct:.0f} "
            f"(p {p_display}){suffix}{threshold_note}"
        )

    return QualityCheck(
        label="Sample Ratio Mismatch (SRM)",
        passed=result.passed,
        detail=detail,
    )
