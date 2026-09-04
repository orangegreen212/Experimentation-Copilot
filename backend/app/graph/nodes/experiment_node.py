"""
Experiment node — orchestrates app/stats/hypothesis_tests.py,
power_analysis.py, and variance_reduction.py. No math here — only
wiring the already-resolved columns/masks to those modules, and
deciding (from Settings) whether to attempt CUPED/bootstrap.
"""

import time

import numpy as np

from app.core.dataset_store import get_dataset
from app.core.logging import get_node_logger
from app.graph.state import GraphState
from app.stats.dataset_classifier import (
    DatasetClassificationError,
    deduplicate_by_user,
    humanize_metric_label,
    resolve_control_label,
)
from app.stats.hypothesis_tests import compute_multi_arm_stat_results, compute_stat_result, select_test
from app.schemas.segmentation import SegmentationResult
from app.schemas.stratification import (
    StratificationEligibility,
    StratificationIneligibilityReason,
    StratificationResult,
    StratificationStatus,
)
from app.stats.power_analysis import compute_power_analysis, format_mde, format_sample_size_note
from app.stats.segmentation import run_segmentation_analysis
from app.stats.stratification import run_stratified_analysis
from app.stats.variance_reduction import apply_cuped_to_experiment, bootstrap_ci_for_difference

log = get_node_logger("Experiment")


def experiment_node(state: GraphState) -> GraphState:
    df = get_dataset(state["dataset_id"])
    columns = state["experiment_columns"]
    settings = state["settings"]

    if columns is None:
        # Defensive — validation_node normally catches this first since
        # it always runs before experiment_node in the graph, but this
        # guards against any future routing path that reaches
        # experiment_node directly.
        raise DatasetClassificationError(
            "Cannot run the hypothesis test — this dataset does not have a recognizable "
            "user/variant/metric column structure for A/B analysis."
        )

    # Deduplicated the same way validation_node does — see that
    # module's comment. Ensures the sample size used for the
    # hypothesis test and power analysis matches DatasetInfo.users
    # exactly, rather than silently counting duplicate rows as extra
    # observations.
    df = deduplicate_by_user(df, columns.user_col)

    variant_values = df[columns.variant_col].dropna().unique().tolist()
    control_label = resolve_control_label(df, columns.variant_col)
    multi_arm = len(variant_values) > 2

    control_mask = df[columns.variant_col] == control_label
    variant_mask = ~control_mask

    control_metric = df.loc[control_mask, columns.metric_col]
    variant_metric = df.loc[variant_mask, columns.metric_col]

    variance_reduction = None
    if settings.cuped and multi_arm:
        # Existing CUPED implementation is explicitly two-arm. Do not
        # silently apply a pooled treatment adjustment to a multi-arm
        # experiment because that would change the estimand.
        log.info("[Experiment] CUPED skipped — multi-arm CUPED requires per-arm covariate adjustment.")
    elif settings.cuped:
        _cuped_start = time.monotonic()
        log.info("[Experiment] starting CUPED")
        adj_control, adj_variant, vr_result = apply_cuped_to_experiment(
            df, columns.metric_col, control_mask, variant_mask
        )
        log.info("[Experiment] CUPED completed in %.2fs", time.monotonic() - _cuped_start)
        variance_reduction = vr_result
        if adj_control is not None:
            control_metric, variant_metric = adj_control, adj_variant
            log.info(
                "[Experiment] CUPED applied — variance reduced %.1f%%",
                vr_result.variance_reduction_pct,
            )
        else:
            log.info("[Experiment] CUPED skipped — %s", vr_result.method)

    metric_label = humanize_metric_label(columns.metric_col)
    if multi_arm:
        arms = {
            str(label): df.loc[df[columns.variant_col] == label, columns.metric_col]
            for label in variant_values
        }
        stat_results = compute_multi_arm_stat_results(
            arms, columns.metric_type, metric_label, control_label=control_label
        )
        test_selection = None
    else:
        test_selection = select_test(control_metric, variant_metric, columns.metric_type)
        stat_results = [compute_stat_result(
            control_metric, variant_metric, columns.metric_type, metric_label, test_selection
        )]

    bootstrap_ci_check = None
    bootstrap_iterations = None
    if settings.bootstrap:
        statistic = "mean" if columns.metric_type.value != "binary" else "mean"
        bootstrap_iterations = 10000
        ci_lower, ci_upper = bootstrap_ci_for_difference(
            control_metric, variant_metric, statistic=statistic, iterations=bootstrap_iterations
        )
        bootstrap_ci_check = (ci_lower, ci_upper)
        log.info(
            "[Experiment] Bootstrap CI cross-check (%d iterations): [%.4f, %.4f]",
            bootstrap_iterations, ci_lower, ci_upper
        )

    # For multi-arm experiments, power/MDE is anchored to the control vs the
    # strongest adjusted-significant treatment (or the strongest observed
    # treatment when the omnibus test is null). This keeps the existing
    # two-arm power schema meaningful without pretending that a single
    # omnibus MDE exists.
    if multi_arm:
        pairwise = [r for r in stat_results if not r.is_omnibus and r.arm is not None]
        candidates = [r for r in pairwise if r.significant] or pairwise
        winner = None
        if candidates:
            def _relative_effect(result):
                import re
                match = re.search(r"[-+]?\d+(?:\.\d+)?", result.delta)
                return float(match.group()) if match else float("-inf")
            positive = [r for r in candidates if _relative_effect(r) > 0]
            winner = max(positive or candidates, key=_relative_effect)
            stat_results = [
                r.model_copy(update={"is_winner": r.comparison == winner.comparison})
                for r in stat_results
            ]
            winner_arm = winner.arm
            winner_series = df.loc[df[columns.variant_col].astype(str) == str(winner_arm), columns.metric_col]
        else:
            winner_series = variant_metric
        power_result = compute_power_analysis(control_metric, winner_series, columns.metric_type)
        mde = abs(power_result.minimum_detectable_effect_relative)
        import re
        enriched = []
        for result in stat_results:
            if result.is_omnibus:
                enriched.append(result)
                continue
            match = re.search(r"[-+]?\d+(?:\.\d+)?", result.delta)
            effect = abs(float(match.group())) if match else None
            practical = bool(effect is not None and not np.isnan(mde) and effect >= mde) if effect is not None else None
            enriched.append(result.model_copy(update={"practical_significant": practical}))
        stat_results = enriched
        log.info(
            "[Experiment] Multi-arm analysis completed — omnibus p=%.4f, %d pairwise comparisons",
            stat_results[0].p_value,
            len(pairwise),
        )
    else:
        power_result = compute_power_analysis(control_metric, variant_metric, columns.metric_type)
        log.info(
            "[Experiment] %s completed — p=%.4f, significant=%s",
            test_selection.test_type.value,
            stat_results[0].p_value,
            stat_results[0].significant,
        )

    # Phase 5 — segmentation. Deliberately two-arm only for this phase:
    # a per-segment breakdown of a 3+ arm omnibus/pairwise result would
    # multiply the multiple-comparisons surface (segments x arms) well
    # beyond what the Holm correction here is designed for, and no
    # requirement asked for multi-arm segmentation. Runs against the
    # SAME deduplicated `df` and raw `columns.metric_col` used for the
    # rest of validation/experiment — never the CUPED-adjusted arrays,
    # since those are no longer aligned with segment membership.
    _seg_start = time.monotonic()
    log.info("[Experiment] starting segmentation")
    if multi_arm:
        segmentation_result = SegmentationResult(
            ran=False,
            reason="Segmentation is not run for multi-arm (3+ variant) experiments in this phase.",
            usable_dimensions=[],
            skipped_dimensions=[],
            dimension_results=[],
            min_segment_size=0,
        )
    else:
        segmentation_result = run_segmentation_analysis(
            df=df,
            user_col=columns.user_col,
            variant_col=columns.variant_col,
            control_label=control_label,
            metric_col=columns.metric_col,
            metric_type=columns.metric_type,
        )
        log.info(
            "[Experiment] Segmentation — ran=%s, usable_dimensions=%d",
            segmentation_result.ran,
            len(segmentation_result.usable_dimensions),
        )
    log.info("[Experiment] segmentation completed in %.2fs", time.monotonic() - _seg_start)

    # TRUE stratified analysis (as distinct from segmentation above) —
    # only computed when the UI explicitly selected it (see
    # AnalysisSettings.analysis_mode / planner_strategy.py's
    # plan_from_explicit_settings). Two-arm only, same restriction as
    # segmentation and for the same reason (a 3+ arm stratified
    # combination multiplies the arms x strata surface without a
    # requirement asking for it). Always runs against the SAME raw,
    # deduplicated `df` and `columns.metric_col` segmentation uses —
    # never the CUPED-adjusted arrays — since stratification and CUPED
    # are orthogonal techniques (see stats/stratification.py's module
    # docstring), not mutually exclusive alternatives.
    stratification_result = None
    if settings.analysis_mode == "stratified" and settings.stratification_column:
        if multi_arm:
            stratification_result = StratificationResult(
                status=StratificationStatus.RAN,
                stratification_column=settings.stratification_column,
                eligibility=StratificationEligibility(
                    stratification_column=settings.stratification_column,
                    eligible=False,
                    reason=(
                        "Stratified analysis is not supported for multi-arm (3+ variant) "
                        "experiments in this phase."
                    ),
                    ineligibility_reason=StratificationIneligibilityReason.MULTI_ARM_NOT_SUPPORTED,
                ),
                estimate=None,
            )
            log.info("[Experiment] Stratified analysis skipped — multi-arm not supported.")
        else:
            eligibility, estimate = run_stratified_analysis(
                df=df,
                variant_col=columns.variant_col,
                control_label=control_label,
                metric_col=columns.metric_col,
                metric_type=columns.metric_type,
                stratification_col=settings.stratification_column,
                metric_label=metric_label,
            )
            stratification_result = StratificationResult(
                status=StratificationStatus.RAN,
                stratification_column=settings.stratification_column,
                eligibility=eligibility,
                estimate=estimate,
            )
            log.info(
                "[Experiment] Stratified analysis — column=%s, eligible=%s",
                settings.stratification_column,
                eligibility.eligible,
            )

    return {
        **state,
        "test_selection": test_selection,
        "stat_results": stat_results,
        "power_analysis": power_result,
        "variance_reduction": variance_reduction,
        "bootstrap_ci_check": bootstrap_ci_check,
        "bootstrap_iterations": bootstrap_iterations,
        "segmentation_result": segmentation_result,
        "stratification_result": stratification_result,
        "_mde_display": format_mde(power_result.minimum_detectable_effect_relative),
        "_sample_size_note": format_sample_size_note(power_result),
    }
