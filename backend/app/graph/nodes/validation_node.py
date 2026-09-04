"""
Validation node — orchestrates app/stats/srm.py + quality_checks.py.
All statistical logic lives in those modules; this node only wires
them to the resolved experiment columns and folds results into the
frontend-facing QualityCheck list.
"""

from app.core.dataset_store import get_dataset
from app.core.logging import get_node_logger
from app.graph.state import GraphState
from app.schemas.quality import QualityCheck
from app.schemas.statistics import MetricType
from app.stats.dataset_classifier import (
    DatasetClassificationError,
    analyze_duplicate_users,
    deduplicate_by_user,
    resolve_control_label,
)
from app.stats.hypothesis_tests import _LARGE_SAMPLE_THRESHOLD
from app.stats.quality_checks import (
    check_normality,
    check_nulls,
    check_outliers,
    combine_outlier_checks,
    normality_result_to_quality_check,
)
from app.stats.srm import check_srm, check_srm_multi_arm, srm_result_to_quality_check

log = get_node_logger("Validation")


def validation_node(state: GraphState) -> GraphState:
    df = get_dataset(state["dataset_id"])
    columns = state["experiment_columns"]

    if columns is None:
        reason = state.get("experiment_columns_error")
        raise DatasetClassificationError(
            reason
            if reason
            else (
                "Cannot validate data quality — this dataset does not have a recognizable "
                "user/variant/metric column structure for A/B analysis (it may be a "
                "funnel/event-log dataset with no separate outcome column)."
            )
        )

    # Duplicate-user rows are checked and reported BEFORE dedup (so the
    # QualityChecks reflect what was actually in the file), then the
    # dataset is deduplicated for every downstream count — SRM,
    # outliers, normality, and (in experiment_node) the hypothesis test
    # and power analysis. This is what keeps "290,584 users" (the
    # classifier's unique-user count) and "N users observed" (used in
    # the stats) from ever disagreeing again — they were computed two
    # different ways (nunique() vs raw row count) before this fix.
    #
    # SEVERITY SPLIT (see DuplicateUserAnalysis docstring): a duplicate
    # is harmless if it's the same user in the same variant — dedup and
    # move on. But if the SAME user_id appears under DIFFERENT variants,
    # that's not a duplicate, it's a broken assignment pipeline (a
    # variant crossover), and — like an SRM failure — it can fabricate
    # an arbitrary effect size. That case is treated as severe enough
    # to halt the pipeline before Experiment runs (see route_after_validation).
    dup_analysis = analyze_duplicate_users(df, columns.user_col, columns.variant_col, columns.metric_col)

    if dup_analysis.duplicate_row_count == 0:
        duplicate_rows_check = QualityCheck(label="Duplicate User Rows", passed=True, detail="No duplicate user_id rows")
    else:
        duplicate_rows_check = QualityCheck(
            label="Duplicate User Rows",
            passed=True,
            detail=(
                f"{dup_analysis.duplicate_row_count:,} duplicate user_id row(s) found — "
                f"deduplicated (kept first occurrence) before analysis"
            ),
        )

    quality_checks_extra = [duplicate_rows_check]

    if dup_analysis.has_severe_conflict:
        quality_checks_extra.append(
            QualityCheck(
                label="Duplicate User Variant Conflicts",
                passed=False,
                detail=(
                    f"{dup_analysis.conflicting_variant_users:,} user(s) found assigned to MORE THAN ONE "
                    f"variant — this is not a harmless duplicate, it indicates a broken randomization/"
                    f"assignment pipeline. Results below cannot be trusted until this is fixed."
                ),
            )
        )
    elif dup_analysis.conflicting_metric_users > 0:
        quality_checks_extra.append(
            QualityCheck(
                label="Duplicate User Metric Conflicts",
                passed=False,
                critical=True,
                detail=(
                    f"{dup_analysis.conflicting_metric_users:,} user(s) have duplicate rows in the SAME "
                    f"variant but with DIFFERENT metric values — the first occurrence was kept, but "
                    f"which value is correct is ambiguous. Investigate the data source."
                ),
            )
        )

    df = deduplicate_by_user(df, columns.user_col)

    control_mask = df[columns.variant_col] == resolve_control_label(df, columns.variant_col)
    variant_mask = ~control_mask

    control_metric = df.loc[control_mask, columns.metric_col]
    variant_metric = df.loc[variant_mask, columns.metric_col]

    arm_labels = df[columns.variant_col].dropna().unique().tolist()
    if len(arm_labels) > 2:
        arm_counts = [int((df[columns.variant_col] == label).sum()) for label in arm_labels]
        srm_result = check_srm_multi_arm(arm_counts)
    else:
        srm_result = check_srm(observed_control=int(control_mask.sum()), observed_variant=int(variant_mask.sum()))
    quality_checks = [srm_result_to_quality_check(srm_result), *quality_checks_extra]

    nulls_check = check_nulls(df, [columns.metric_col])
    quality_checks.append(nulls_check)

    outlier_control = check_outliers(control_metric, metric_type=columns.metric_type)
    outlier_variant = check_outliers(variant_metric, metric_type=columns.metric_type)
    quality_checks.append(combine_outlier_checks(outlier_control, outlier_variant))

    # Normality is only meaningful for continuous metrics; for binary
    # metrics the test selector's own chi-square/Fisher's logic handles
    # validity, so we skip adding a normality row to avoid a
    # nonsensical "normality" check on a 0/1 column.
    if columns.metric_type != MetricType.BINARY:
        normality = check_normality(control_metric, variant_metric)
        # Must mirror hypothesis_tests.py's own n>=30-per-arm threshold
        # exactly — this flag exists purely to say "was this result
        # actually used to pick the test", so it has to match the real
        # decision rule, not an independently-chosen number.
        large_sample_rule_applied = (
            len(control_metric.dropna()) >= _LARGE_SAMPLE_THRESHOLD
            and len(variant_metric.dropna()) >= _LARGE_SAMPLE_THRESHOLD
        )
        quality_checks.append(
            normality_result_to_quality_check(normality, large_sample_rule_applied=large_sample_rule_applied)
        )

    log.info(
        "[Validation] SRM %s (p=%.3f) — %d/%d quality checks passed",
        "passed" if srm_result.passed else "FAILED",
        srm_result.p_value,
        sum(1 for qc in quality_checks if qc.passed),
        len(quality_checks),
    )

    return {
        **state,
        "quality_checks": quality_checks,
        "srm_result": srm_result,
        "has_conflicting_variant_duplicates": dup_analysis.has_severe_conflict,
        # Exact count for report/UI text (e.g. "1,895 users have conflicting
        # variant assignments") — the boolean above still drives the actual
        # validity gate (route_after_validation) unchanged; this is purely
        # an additional, already-computed fact for reporting.
        "conflicting_variant_user_count": dup_analysis.conflicting_variant_users,
    }
