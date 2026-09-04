"""
Dataset classifier — Stage 2.

PURE, DETERMINISTIC, NO LLM. Takes a pandas DataFrame, returns a
`DatasetInfo`. This is the foundation everything else depends on: if
this misidentifies the variant column or user column, every downstream
stat is wrong. Keep this module boring and well-tested.

Heuristics (applied together, per project decision):

  1. STRUCTURAL — column name matching against known vocabularies for
     "raw event log" columns vs "per-user outcome" columns.
  2. QUANTITATIVE — rows-per-user ratio. Raw event logs have many rows
     per user; aggregated/user-level data has ~1 row per user.
  3. FALLBACK — if the two heuristics disagree, or no user-id-like
     column is found at all, classify as UNKNOWN. We still attempt to
     report variants/users on a best-effort basis.

The two heuristics are combined with a simple voting scheme rather than
one overriding the other, so a CSV that "looks aggregated" by column
names but actually has a high rows-per-user ratio doesn't get silently
misclassified.
"""

from __future__ import annotations

import re

import pandas as pd

from app.schemas.dataset import DatasetInfo, DatasetType, ExperimentUnitLevel, RatioMetricCandidate, UnitIdentifierType
from app.schemas.statistics import MetricType

# --- column-name vocabularies -------------------------------------------

_USER_ID_CANDIDATES = ["user_id", "userid", "user", "customer_id", "visitor_id", "id"]
_VARIANT_CANDIDATES = [
    "variant", "group", "arm", "bucket", "treatment", "cohort",
    "experiment_group", "test_group", "ab_group", "ab_variant", "variant_group",
    "variant_name", "group_name",
    # CRM/marketing experiment naming (e.g. Hillstrom-style datasets):
    # a "segment" column here means treatment-arm assignment, not a
    # post-hoc analytical segmentation dimension — those are unrelated
    # concepts that happen to share a common English word.
    "segment", "treatment_group", "campaign_group",
]
# Known holdout/no-treatment VALUE labels (not column names — see
# _VARIANT_CANDIDATES above) for resolve_control_label()'s step 2.
# Normalized against whitespace/hyphens/underscores collapsed to a
# single space and lowercased, so "No-E-Mail", "no_email", and
# "No E-Mail" all match the same "no e mail" / "no email" entries.
_HOLDOUT_LABELS = {
    "control", "ctrl", "holdout", "no email", "no e mail",
    "no campaign", "no treatment", "untreated",
}
_EVENT_COLUMN_CANDIDATES = ["event_name", "event_type", "event", "action"]
_TIMESTAMP_COLUMN_CANDIDATES = ["timestamp", "event_time", "created_at", "occurred_at", "time"]
_OUTCOME_COLUMN_CANDIDATES = [
    "converted", "conversion", "purchased", "is_converted",
    "order_value", "revenue", "value", "amount", "click", "clicked",
    # Plural/aggregate-count variants of the same outcomes above — the
    # per-user candidates already existed; these are the equivalent
    # names used in pre-aggregated (variant | users | conversions)
    # summary datasets.
    "conversions", "purchases", "clicks",
    # Generic outcome name used by user-level experiment datasets.
    # Kept after the existing candidates so all legacy datasets retain
    # exactly the same priority.
    "metric",
]

# Column names that plausibly represent the per-arm sample-size
# denominator in a pre-aggregated (variant | users | conversions) summary
# dataset, as opposed to individual-level rows. Used ONLY for dataset-
# shape recognition and honest capability messaging — never to fabricate
# per-user rows or feed a new aggregated statistical engine.
_AGGREGATE_SIZE_CANDIDATES = ["users", "sample_size", "n", "impressions", "visitors", "sessions"]

# Explicit CUPED/pre-experiment covariate names. These are never primary
# outcomes; they are inputs to variance reduction.
_CUPED_COVARIATE_CANDIDATES = [
    "pre_experiment_metric", "pre_experiment_value", "pre_metric",
    "pre_metric_value", "baseline", "baseline_metric", "baseline_value",
    "covariate", "cuped_covariate",
]

# Naming vocabulary a guardrail metric column plausibly matches.
# Naming heuristic ONLY:
# this never runs a statistical guardrail check (that's a separate,
# not-yet-wired computation — see ExperimentReport.guardrail_results),
# it only flags which of the ALREADY-detected available_metrics look
# like they'd plausibly serve as a guardrail (something you watch for
# regressions, not the thing you're trying to move) rather than the
# primary success metric.
_DEMOGRAPHIC_ATTRIBUTE_KEYWORDS = ["age", "gender", "sex", "dob", "birth_year"]

_GUARDRAIL_LOWER_IS_BETTER_KEYWORDS = [
    "latency", "load_time", "response_time", "error_rate", "error",
    "bounce_rate", "bounce", "crash_rate", "crash", "unsubscribe",
    "complaint", "refund", "churn", "cancellation", "cancel", "page_load",
]

# Business/monetary and engagement guardrails (real coverage gap — the
# list above was ops/reliability-only and missed common revenue and
# engagement guardrails like `purchase_amount`, `revenue_gbp`,
# `session_duration`, `spend`, `views`, `visit`, `add_to_cart`,
# `first_order_value`, `order_value`). Kept in a SEPARATE list from
# `_GUARDRAIL_LOWER_IS_BETTER_KEYWORDS` — unlike bounce/churn/error,
# an increase in these is good, not bad, and `infer_guardrail_direction`
# below depends on that split to set `higher_is_better` correctly.
# Matched with word-boundary rules (see `_keyword_matches_column`
# below), not plain substring, so a generic short token like "visit"
# never false-positives on an unrelated identifier column like
# `visitor_id`.
_GUARDRAIL_HIGHER_IS_BETTER_KEYWORDS = [
    "revenue", "purchase", "spend", "order_value", "aov",
    "session_duration", "session_time", "views", "visit",
    "add_to_cart", "first_order",
]

_GUARDRAIL_METRIC_KEYWORDS = _GUARDRAIL_LOWER_IS_BETTER_KEYWORDS + _GUARDRAIL_HIGHER_IS_BETTER_KEYWORDS

# A column is a plausible STRATIFICATION dimension candidate if it's
# a low-cardinality, non-numeric-looking grouping column that isn't
# already playing one
# of the structural roles (user id / variant / metric / event /
# timestamp / covariate). This is a display-only candidate list for
# Dataset Classification, never a promise of eligibility — the actual
# causal-stratification eligibility gate is
# app.stats.stratification.check_stratification_eligibility.
_MAX_STRATIFICATION_CANDIDATE_CARDINALITY = 20


# rows-per-user ratio above this => treat as raw event-level data
_RAW_ROWS_PER_USER_THRESHOLD = 1.5

# Synthetic identifier column name used ONLY when the multi-signal grain
# classifier (see `classify_experiment_unit_level`) has established that
# a dataset with no explicit identifier is genuinely unit-level (one row
# = one randomized experimental unit). Reuses every existing user_col-
# keyed code path (dedup, duplicate-conflict analysis, SRM, segmentation)
# unchanged, rather than introducing a parallel "no id" branch through
# the rest of the pipeline. Every value is unique by construction (a
# fresh 0..N-1 range), so dedup/duplicate-analysis against it are
# provable no-ops.
_IMPLICIT_UNIT_ID_COLUMN = "__experiment_unit_row_id__"

# CRM/marketing pre-treatment (safe segmentation) vocabulary. These are
# customer attributes that exist independent of / before randomization.
# Checked BEFORE the post-treatment vocabulary below, so a column like
# "previous_purchase_count" (a pre-treatment history attribute) is never
# misclassified via a loose substring match on "purchase" (a
# post-treatment outcome word) — see `classify_column_treatment_timing`.
_PRE_TREATMENT_KEYWORDS = [
    "age", "recency", "historical_spend", "history", "customer_type",
    "channel", "region", "newbie", "previous_purchase_count",
    "previous_purchase", "prior_purchase", "tenure", "signup_date",
    "account_age", "zip_code", "zipcode", "loyalty_tier",
    "customer_since", "gender", "income", "baseline",
]

# CRM/marketing post-treatment (unsafe segmentation) vocabulary — these
# are outcomes measured AFTER randomization/treatment and must never be
# used as a segmentation dimension for treatment-effect analysis (doing
# so introduces post-treatment bias / conditions on a descendant of the
# treatment itself).
_POST_TREATMENT_KEYWORDS = [
    "conversion", "converted", "visit", "visited", "click", "clicked",
    "open", "opened", "campaign_revenue", "current_spend",
    "current_order", "response", "responded", "purchase", "purchased",
    "signup", "signed_up", "revenue", "spend", "order_value",
    "customer_value", "delivered", "sent", "add_to_cart",
]

# CRM funnel stages, in canonical order, for the display-only
# `DatasetInfo.funnel_metrics` field. Only reported when 2+ stages are
# present in a single (unit-level) dataset — this is separate from
# `app.stats.funnel_classifier`, which handles genuine event-log
# funnels (one row per event, inferred step order from timestamps).
_FUNNEL_STAGE_VOCAB: list[tuple[str, list[str]]] = [
    ("sent", ["sent"]),
    ("delivered", ["delivered"]),
    ("opened", ["opened", "open"]),
    ("clicked", ["clicked", "click"]),
    ("visited", ["visited", "visit"]),
    ("converted", [
        "converted", "conversion", "purchase", "purchased",
        "signup", "signed_up", "response", "responded",
    ]),
]
_MIN_FUNNEL_STAGES = 2


class DatasetClassificationError(ValueError):
    """Raised when the CSV is too malformed to classify at all (e.g. empty)."""


def attach_implicit_unit_id(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Materialize the SAME synthetic row-id column, with the exact same
    construction (name-collision-avoiding column name, fresh 0..N-1
    range), that `detect_experiment_columns` builds internally when
    `ExperimentColumns.is_implicit_unit` is True.

    `detect_experiment_columns` only returns column NAMES, not the
    dataframe it built them against — its synthetic column lived on a
    local `working_df` copy that the caller never receives. Any caller
    that intends to persist/reuse the dataset under
    `columns.user_col` (e.g. classifier_node storing it under
    `state["dataset_id"]` for every downstream node to re-fetch) MUST
    call this first and persist ITS return value, or every downstream
    lookup of `df[columns.user_col]` raises KeyError — the synthetic
    column would exist only in classification's own transient copy.

    Returns `(df_with_column, column_name)` — `column_name` matches
    `ExperimentColumns.user_col` exactly when `is_implicit_unit` is
    True, so callers can pass it straight through unchanged.
    """
    implicit_col = _IMPLICIT_UNIT_ID_COLUMN
    while implicit_col in df.columns:
        implicit_col = f"_{implicit_col}"
    out = df.copy()
    out[implicit_col] = range(len(out))
    return out, implicit_col


def _find_column(columns: list[str], candidates: list[str]) -> str | None:
    """Case-insensitive exact match against a candidate vocabulary, in priority order."""
    lowered = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _detect_user_column(df: pd.DataFrame) -> str | None:
    return _find_column(list(df.columns), _USER_ID_CANDIDATES)


def _detect_variant_column(df: pd.DataFrame) -> str | None:
    return _find_column(list(df.columns), _VARIANT_CANDIDATES)


def enrich_with_assignment(
    df: pd.DataFrame,
    assignment_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Deterministically merge a separate experiment-assignment dataset
    (e.g. ``user_id | variant``) onto the primary dataset, so a primary
    dataset with no variant column of its own (e.g. ``user_id |
    order_value``) can still be classified once each user's arm is
    known from a separate assignment export.

    - ``assignment_df is None`` -> `df` is returned completely
      unchanged (identity) — every existing single-file caller is
      unaffected.
    - Otherwise: detect the user column in BOTH frames and the variant
      column in `assignment_df`, using the SAME `_detect_user_column` /
      `_detect_variant_column` vocabularies used everywhere else in
      this module (no separate/duplicated heuristic). Only a column
      that matches the recognized variant vocabulary can become the
      variant column — a business field such as `plan`,
      `billing_frequency`, `order_type`, `subscription_type`, or
      `product` is NEVER inferred as a variant just because it happens
      to sit in the assignment file.
    - ONLY the user column and the variant column are imported from
      `assignment_df` — every other assignment-file column (e.g. an
      `assigned_at` timestamp) is dropped before the merge, so it can
      never leak into metric/CUPED detection downstream.
    - `assignment_df` must be one-row-per-user; a duplicate user
      assignment raises `DatasetClassificationError` rather than
      silently picking a row (a duplicate assignment is a broken
      randomization export, not something to resolve implicitly).
    - Merges with `how="left", validate="many_to_one"`: every row of
      the primary dataset is preserved (including raw/event-level data
      with multiple rows per user); users with no assignment-file entry
      get NaN in the new variant column rather than being dropped.
    """
    if assignment_df is None:
        return df

    primary_user_col = _detect_user_column(df)
    assignment_user_col = _detect_user_column(assignment_df)
    assignment_variant_col = _detect_variant_column(assignment_df)

    if primary_user_col is None:
        raise DatasetClassificationError(
            "Cannot merge assignment data: the primary dataset has no recognizable "
            "user identifier column (e.g. user_id, customer_id, visitor_id)."
        )
    if assignment_user_col is None:
        raise DatasetClassificationError(
            "Cannot merge assignment data: the assignment dataset has no recognizable "
            "user identifier column (e.g. user_id, customer_id, visitor_id)."
        )
    if assignment_variant_col is None:
        raise DatasetClassificationError(
            "Cannot merge assignment data: the assignment dataset has no recognizable "
            "variant/group column (e.g. variant, group, arm, bucket, treatment, cohort)."
        )

    if assignment_variant_col in df.columns and assignment_variant_col != primary_user_col:
        # A real name collision with a non-key column would otherwise be
        # silently resolved by pandas into `<name>_x`/`<name>_y`, which
        # _detect_variant_column would then fail to recognize at all —
        # fail loudly and specifically instead.
        raise DatasetClassificationError(
            f"Cannot merge assignment data: the primary dataset already has a column "
            f"named '{assignment_variant_col}', which would collide with the "
            "assignment dataset's variant column of the same name."
        )

    # Import ONLY the user column + variant column — nothing else from
    # the assignment file is ever allowed to reach the merged dataset.
    slim_assignment = assignment_df[[assignment_user_col, assignment_variant_col]].copy()

    duplicate_assignments = int(slim_assignment.duplicated(subset=[assignment_user_col]).sum())
    if duplicate_assignments > 0:
        raise DatasetClassificationError(
            f"Cannot merge assignment data: found {duplicate_assignments} duplicate "
            f"user assignment row(s) in '{assignment_user_col}' — each user must appear "
            "exactly once in the assignment dataset. Fix the assignment export before retrying."
        )

    merged = df.merge(
        slim_assignment,
        how="left",
        left_on=primary_user_col,
        right_on=assignment_user_col,
        validate="many_to_one",
    )
    if assignment_user_col != primary_user_col:
        # Drop the assignment file's own (now-redundant) join-key column
        # — only the user column and variant column were ever supposed
        # to be imported, never a second copy of the user id under a
        # different name.
        merged = merged.drop(columns=[assignment_user_col])
    return merged


def _detect_metric_column(df: pd.DataFrame, exclude: set[str]) -> str | None:
    """
    Find the primary outcome/metric column.

    Selection order:
      1. Existing explicit outcome vocabulary (legacy behavior preserved).
      2. If no named outcome exists, use a single eligible numeric column.
         This supports generic names such as ``metric`` without guessing
         when multiple numeric outcomes exist.
      3. Explicit CUPED/pre-experiment covariates are never selected as outcome.
    """
    candidates = [c for c in df.columns if c not in exclude]

    # Preserve legacy behavior and priority.
    matched = _find_column(candidates, _OUTCOME_COLUMN_CANDIDATES)
    if matched is not None:
        return matched

    covariates = {c.lower() for c in _CUPED_COVARIATE_CANDIDATES}
    eligible_numeric = [
        c for c in candidates
        if c.lower() not in covariates
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    # Safe fallback only when there is no ambiguity.
    return eligible_numeric[0] if len(eligible_numeric) == 1 else None



def _select_metric_column(
    df: pd.DataFrame, exclude: set[str], preferred_metric: str | None = None
) -> tuple[str | None, str]:
    """
    Choose the primary metric column AND explain the choice, deterministically.

    This must never silently default to "the first column" or hardcode
    conversion rate — the reason string is a first-class output, not an
    afterthought, so the report can state plainly why a metric was (or
    wasn't) picked and whether any competing metric existed at all.

    Selection order:
      1. If the request text names a metric that matches exactly one of
         the dataset's available numeric outcome columns (by column name
         or humanized label), that column wins — the analyst asked for it.
      2. Otherwise fall back to the existing deterministic outcome-column
         priority list (`_OUTCOME_COLUMN_CANDIDATES`), unchanged from
         before this fix, so untouched datasets classify exactly as they
         did previously.

    `preferred_metric` is matched against plain column/label text only —
    never sent to an LLM — so this stays a deterministic classifier fact,
    not an interpretation.
    """
    numeric_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    numeric_cols = [c for c in numeric_cols if c.lower() not in _EVENT_COLUMN_CANDIDATES + _TIMESTAMP_COLUMN_CANDIDATES]

    if preferred_metric and preferred_metric.strip():
        prompt = preferred_metric.lower()
        matches = [
            c for c in numeric_cols
            if c.lower() in prompt or humanize_metric_label(c).lower() in prompt
        ]
        if len(matches) == 1:
            col = matches[0]
            return col, (
                f"Selected {humanize_metric_label(col)} because the analysis request "
                f"explicitly referenced it."
            )
        # Zero or multiple ambiguous matches -> fall through to the
        # deterministic default rather than guessing.

    default_col = _detect_metric_column(df, exclude=exclude)
    if default_col is None:
        return None, "No recognizable primary metric column was found in this dataset."

    others = [humanize_metric_label(c) for c in numeric_cols if c != default_col]
    if not others:
        return default_col, (
            f"Selected {humanize_metric_label(default_col)} as the primary metric — "
            f"no competing outcome metrics were available in this dataset."
        )
    return default_col, (
        f"Selected {humanize_metric_label(default_col)} as the primary metric by the "
        f"deterministic outcome-column priority, since the request did not specify one. "
        f"Other available metric(s) — {', '.join(others)} — were not selected as primary; "
        f"ask about one of them by name to analyze it instead."
    )


def detect_available_metrics(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    """Return numeric outcome candidates for dataset-specific metric selection.

    This is deliberately broader than `_detect_metric_column`: the latter picks
    one metric for experiment execution, while this function exposes all plausible
    numeric outcomes so a methodology/metric-selection question can explain why
    one should be primary without pretending the dataset is an A/B test.

    Demographic/identity attributes (`_DEMOGRAPHIC_ATTRIBUTE_KEYWORDS` —
    e.g. `age`, `gender`) are deliberately excluded here even though
    they're numeric or low-cardinality: a user's age is a baseline
    characteristic, not an experiment outcome, so it must never be
    offered as an "Additional Metric" or as a selectable Guardrail
    Metric (the guardrail multiselect in the UI is built directly from
    this function's output). This is the fix for the guardrail
    root-cause audit's Age finding — `Age` showed up as a guardrail
    not because the decision engine invented it, but because the UI
    let it be selected in the first place. Demographic columns are
    unaffected everywhere else — they remain fully available via
    `detect_stratification_candidates`, which is the correct path for
    a baseline/demographic dimension.
    """
    candidates = []
    for col in df.columns:
        if col in exclude:
            continue
        low = col.lower()
        if low in (
            _EVENT_COLUMN_CANDIDATES
            + _TIMESTAMP_COLUMN_CANDIDATES
            + _CUPED_COVARIATE_CANDIDATES
        ):
            continue
        if any(_keyword_matches_column(low, keyword) for keyword in _DEMOGRAPHIC_ATTRIBUTE_KEYWORDS):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            candidates.append(humanize_metric_label(col))
    return list(dict.fromkeys(candidates))


def detect_cuped_covariate(
    df: pd.DataFrame, exclude: set[str] | None = None
) -> str | None:
    """
    Detect an explicitly named numeric CUPED/pre-experiment covariate.

    Conservative by design: no arbitrary numeric-column guessing.
    """
    excluded = exclude or set()
    names = {name.lower() for name in _CUPED_COVARIATE_CANDIDATES}
    matches = [
        c for c in df.columns
        if c not in excluded
        and c.lower() in names
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    return matches[0] if len(matches) == 1 else None


def humanize_metric_label(column_name: str | None) -> str:
    """
    'converted' -> 'Conversion Rate', 'order_value' -> 'Order Value',
    None -> 'Unknown Metric'. Simple, deterministic string mapping —
    no LLM needed for this.
    """
    if column_name is None:
        return "Unknown Metric"

    known_labels = {
        "converted": "Conversion Rate",
        "conversion": "Conversion Rate",
        "is_converted": "Conversion Rate",
        "purchased": "Conversion Rate",
        "order_value": "Order Value",
        "revenue": "Revenue",
        "value": "Value",
        "amount": "Amount",
        "click": "Click Rate",
        "clicked": "Click Rate",
    }
    return known_labels.get(column_name.lower(), column_name.replace("_", " ").title())


def _structural_vote(df: pd.DataFrame, user_col: str | None) -> DatasetType | None:
    """Vote based on presence of event-log-like vs outcome-like columns.

    A dataset with a real user_id AND an outcome column (e.g. `user_id |
    variant | converted`) is exactly what raw, individual-level
    experiment data looks like — it is never an aggregated summary
    table. Aggregation can only be inferred in the ABSENCE of a
    per-unit identifier (handled separately in `classify_dataset`'s
    `user_col is None` branch below). So when `user_col` is
    present, this votes RAW_USER_LEVEL instead.
    """
    columns = [c.lower() for c in df.columns]
    has_event_cols = any(c in columns for c in _EVENT_COLUMN_CANDIDATES) or any(
        c in columns for c in _TIMESTAMP_COLUMN_CANDIDATES
    )
    has_outcome_cols = any(c in columns for c in _OUTCOME_COLUMN_CANDIDATES)

    if has_event_cols and not has_outcome_cols:
        return DatasetType.RAW_EVENT_LEVEL
    if has_outcome_cols and not has_event_cols:
        return DatasetType.RAW_USER_LEVEL if user_col is not None else DatasetType.AGGREGATED_AB_TEST
    return None  # ambiguous or neither — no structural signal


def _detect_aggregate_size_column(df: pd.DataFrame) -> str | None:
    """Find a per-arm sample-size/denominator column (e.g. 'users', 'sample_size')."""
    return _find_column(list(df.columns), _AGGREGATE_SIZE_CANDIDATES)


def _quantitative_vote(df: pd.DataFrame, user_col: str | None) -> DatasetType | None:
    """Vote based on rows-per-user ratio.

    One row per experimental unit is precisely the normal shape of raw
    user-level A/B test data (e.g.
    `user_id,timestamp,group,landing_page,converted` with ~294K rows
    for ~290K distinct users) — a low rows-per-user ratio alone does not
    imply "aggregated." A real, high-cardinality `user_col` is
    only possible when the data has NOT been aggregated away — genuine
    aggregation always discards the per-unit identifier. So whenever
    `user_col` is not None, this function can only ever distinguish
    between two flavors of RAW data (event-level vs user-level), never
    AGGREGATED — that classification is reserved for the `user_col is
    None` path in `classify_dataset`, which has its own
    dedicated aggregate-size-column evidence.
    """
    if user_col is None:
        return None
    n_users = df[user_col].nunique()
    if n_users == 0:
        return None
    rows_per_user = len(df) / n_users
    if rows_per_user >= _RAW_ROWS_PER_USER_THRESHOLD:
        return DatasetType.RAW_EVENT_LEVEL
    return DatasetType.RAW_USER_LEVEL


class ExperimentGrainResult:
    """
    Output of `classify_experiment_unit_level` — the evidence-backed
    answer to "what does one row represent, relative to the randomized
    experimental unit?". See that function's docstring for the full
    decision logic.
    """

    def __init__(
        self,
        level: ExperimentUnitLevel,
        unit_identifier: UnitIdentifierType,
        confidence: float,
        evidence: list[str],
        blocking_reason: str | None = None,
        identifier_column: str | None = None,
    ):
        self.level = level
        self.unit_identifier = unit_identifier
        self.confidence = confidence
        self.evidence = evidence
        self.blocking_reason = blocking_reason
        self.identifier_column = identifier_column


def _has_low_cardinality_outcome(df: pd.DataFrame, exclude: set[str]) -> str | None:
    """
    Find a column (outside `exclude`) that looks like a row-level
    binary/categorical outcome — either a recognized outcome-vocabulary
    column, or any 2-8 distinct-value column that isn't itself a
    plausible aggregate-count/identifier column. Used only as a UNIT-
    LEVEL signal, never to select the actual metric column (that stays
    `_select_metric_column`'s job).
    """
    named = _find_column([c for c in df.columns if c not in exclude], _OUTCOME_COLUMN_CANDIDATES)
    if named is not None:
        return named
    for col in df.columns:
        if col in exclude or col.lower() in _AGGREGATE_SIZE_CANDIDATES:
            continue
        n_distinct = df[col].dropna().nunique()
        if 2 <= n_distinct <= 8:
            return col
    return None


def classify_experiment_unit_level(df: pd.DataFrame) -> ExperimentGrainResult:
    """
    Multi-signal, deterministic classification of the dataset's GRAIN
    relative to the randomized experimental unit — event_level (many
    rows can belong to one unit; an explicit identifier is required),
    unit_level (one row IS one unit; an explicit identifier is
    optional), aggregate_level (rows are group-level counts), or
    unknown.

    This is intentionally a SEPARATE question from `DatasetType`
    (`_structural_vote` / `_quantitative_vote` above), which votes on
    raw-vs-aggregated shape using a scheme that assumes a user_col
    already exists. Most CRM/marketing exports have no customer id at
    all, and `DatasetType.UNKNOWN` alone can't distinguish "genuinely
    unanalyzable" from "perfectly fine unit-level data with no id
    column" — that honest distinction, with evidence, is this
    function's entire job.

    Decision order (first match wins — see module docstring for why a
    combined multi-signal approach, not one heuristic, is required):

      1. Explicit identifier present (`user_col` found):
         - event/timestamp column present, OR rows-per-unit >= the
           existing raw-event-log threshold -> EVENT_LEVEL,
           EXPLICIT_COLUMN (multiple rows can belong to the same unit).
         - otherwise (~1 row per id) -> UNIT_LEVEL, EXPLICIT_COLUMN.

      2. No explicit identifier:
         a. A variant/group column AND a per-arm sample-size/count
            column (e.g. 'users', 'sample_size') -> AGGREGATE_LEVEL,
            MISSING. Rows describe GROUPS, never individual units —
            this must never be treated as unit-level.
         b. An event/timestamp column is present -> EVENT_LEVEL,
            MISSING, with a `blocking_reason`: multiple rows may
            belong to the same customer, so an explicit identifier is
            required before reliable unit-level analysis. This is the
            protection that must NEVER be weakened.
         c. Otherwise, score the STRONG unit-level signals (variant
            column present; row-level binary/categorical outcome
            present; no event column; no timestamp column; no
            aggregate-count structure; row-level attribute columns
            present). If a variant column is present and enough
            signals agree, -> UNIT_LEVEL, IMPLICIT_ROW (one row = one
            randomized unit) with confidence scaled by signal count.
            Never assumed merely because an id is absent —
            every positive signal is itemized in `evidence`.
         d. Otherwise -> UNKNOWN, MISSING — not enough evidence either
            way; callers must not guess.
    """
    user_col = _detect_user_column(df)
    variant_col = _detect_variant_column(df)
    event_col = _find_column(list(df.columns), _EVENT_COLUMN_CANDIDATES)
    timestamp_col = _find_column(list(df.columns), _TIMESTAMP_COLUMN_CANDIDATES)
    aggregate_size_col = _detect_aggregate_size_column(df)

    # --- 1. Explicit identifier present ------------------------------
    if user_col is not None:
        n_units = int(df[user_col].nunique(dropna=True))
        rows_per_unit = (len(df) / n_units) if n_units else float("inf")
        evidence = [f"An explicit experiment-unit identifier ('{user_col}') was found"]

        if event_col is not None or timestamp_col is not None or rows_per_unit >= _RAW_ROWS_PER_USER_THRESHOLD:
            if event_col is not None:
                evidence.append(f"An event/action column ('{event_col}') was detected")
            if timestamp_col is not None:
                evidence.append(f"A timestamp column ('{timestamp_col}') was detected")
            if rows_per_unit >= _RAW_ROWS_PER_USER_THRESHOLD:
                evidence.append(f"Multiple rows were observed per unit (~{rows_per_unit:.1f} rows/unit)")
            evidence.append("Repeated observations per unit are possible")
            return ExperimentGrainResult(
                level=ExperimentUnitLevel.EVENT_LEVEL,
                unit_identifier=UnitIdentifierType.EXPLICIT_COLUMN,
                confidence=0.97,
                evidence=evidence,
                identifier_column=user_col,
            )

        evidence.append("Approximately one row was observed per unit")
        evidence.append("No event/action or timestamp column suggesting repeated observations was detected")
        return ExperimentGrainResult(
            level=ExperimentUnitLevel.UNIT_LEVEL,
            unit_identifier=UnitIdentifierType.EXPLICIT_COLUMN,
            confidence=0.95,
            evidence=evidence,
            identifier_column=user_col,
        )

    # --- 2a. Aggregate summary (variant + a per-arm count column) ----
    if variant_col is not None and aggregate_size_col is not None:
        return ExperimentGrainResult(
            level=ExperimentUnitLevel.AGGREGATE_LEVEL,
            unit_identifier=UnitIdentifierType.MISSING,
            confidence=0.91,
            evidence=[
                "Rows contain group-level counts",
                f"A per-arm sample-size/count column ('{aggregate_size_col}') was detected",
                f"A variant/group column ('{variant_col}') was detected at the group level",
            ],
        )

    # --- 2b. Event-level data with no identifier (must stay blocked) -
    if event_col is not None or timestamp_col is not None:
        evidence = ["Multiple event records/observations are possible for the same customer"]
        if event_col is not None:
            evidence.append(f"An event/action column ('{event_col}') was detected")
        if timestamp_col is not None:
            evidence.append(f"A timestamp column ('{timestamp_col}') was detected")
        evidence.append("No experiment-unit identifier column was found")
        return ExperimentGrainResult(
            level=ExperimentUnitLevel.EVENT_LEVEL,
            unit_identifier=UnitIdentifierType.MISSING,
            confidence=0.9,
            evidence=evidence,
            blocking_reason=(
                "An experimental-unit identifier is required to aggregate event-level "
                "observations to the randomized unit."
            ),
        )

    # A rate/ratio-style column (e.g. 'click_rate', 'conversion_ratio')
    # with no per-arm sample-size column is a strong signal of a
    # pre-aggregated summary missing its denominator, NOT individual-
    # level data (a real individual has a 0/1 'clicked', never a
    # 'click_rate') — never confidently call this UNIT_LEVEL. Let the
    # caller's existing aggregate/rate-specific error paths handle it
    # with a more specific, actionable message than a bare UNKNOWN would.
    has_rate_like_column = any(
        ("rate" in c.lower() or "ratio" in c.lower()) for c in df.columns if c != variant_col
    )
    if has_rate_like_column and variant_col is not None:
        return ExperimentGrainResult(
            level=ExperimentUnitLevel.UNKNOWN,
            unit_identifier=UnitIdentifierType.MISSING,
            confidence=0.4,
            evidence=[
                "A rate/ratio-style outcome column was detected",
                "No per-arm sample-size/count column was found to recover the underlying denominator",
            ],
            blocking_reason=(
                "A rate/ratio-style outcome without a sample-size denominator is ambiguous "
                "between a missing-denominator aggregate summary and individual-level data — "
                "it cannot be safely treated as one row per randomized unit."
            ),
        )

    # --- 2c. Score unit-level signals ---------------------------------
    signals: list[str] = []
    if variant_col is not None:
        signals.append("A variant/treatment assignment column was detected")
    outcome_col = _has_low_cardinality_outcome(df, exclude={c for c in [variant_col] if c is not None})
    if outcome_col is not None:
        signals.append(f"A row-level binary or categorical outcome ('{outcome_col}') was detected")
    signals.append("No event-type column was detected")
    signals.append("No timestamp/event-sequence column was detected")
    signals.append("No aggregate count/sample-size structure was detected")
    attribute_cols = [
        c for c in df.columns
        if c not in {variant_col, outcome_col} and c.lower() not in _AGGREGATE_SIZE_CANDIDATES
    ]
    if attribute_cols:
        signals.append("Row-level customer/experimental-unit attributes are present")

    # A variant column is a hard requirement for the unit-level
    # inference to fire at all — without it, there's no experiment
    # structure to speak of, only an unlabeled table (never
    # assume unit-level merely because an id is absent).
    if variant_col is not None and len(signals) >= 3:
        confidence = min(0.5 + 0.1 * len(signals), 0.95)
        return ExperimentGrainResult(
            level=ExperimentUnitLevel.UNIT_LEVEL,
            unit_identifier=UnitIdentifierType.IMPLICIT_ROW,
            confidence=confidence,
            evidence=signals,
        )

    return ExperimentGrainResult(
        level=ExperimentUnitLevel.UNKNOWN,
        unit_identifier=UnitIdentifierType.MISSING,
        confidence=0.3,
        evidence=signals or ["No recognizable experiment structure (variant/outcome columns) was found"],
        blocking_reason=(
            "Not enough evidence to determine whether one row represents one randomized "
            "experimental unit — no variant/treatment column and/or outcome column was found."
        ),
    )


def classify_column_treatment_timing(column_name: str) -> str:
    """
    Classify a single column NAME as 'pre_treatment' (safe to use as a
    segmentation dimension for treatment-effect analysis),
    'post_treatment' (measured after assignment — must NOT be used for
    segmentation, since conditioning on it introduces post-treatment
    bias), or 'unknown'.

    Pre-treatment vocabulary is checked FIRST and deliberately includes
    full phrases like 'previous_purchase_count' so it wins over the
    post-treatment substring 'purchase' for exactly that column — this
    is a naming heuristic, not infallible, but matches the explicit
    pre/post examples in the CRM experiment spec.
    """
    low = column_name.lower()
    if any(keyword in low for keyword in _PRE_TREATMENT_KEYWORDS):
        return "pre_treatment"
    if any(keyword in low for keyword in _POST_TREATMENT_KEYWORDS):
        return "post_treatment"
    return "unknown"


def detect_pre_treatment_segmentation_candidates(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    """Columns (outside `exclude`) classified as pre-treatment — safe segmentation candidates."""
    return [c for c in df.columns if c not in exclude and classify_column_treatment_timing(c) == "pre_treatment"]


def detect_post_treatment_exclusions(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    """Columns (outside `exclude`) classified as post-treatment — must never be used for segmentation."""
    return [c for c in df.columns if c not in exclude and classify_column_treatment_timing(c) == "post_treatment"]


def detect_funnel_metrics(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    """
    Detect CRM funnel-stage columns (e.g. sent -> delivered -> opened ->
    clicked -> converted) present in a single (typically unit-level)
    dataset, in canonical funnel order. Naming heuristic only, and only
    reported once 2+ stages are present — a single outcome column is
    just the primary metric, not a funnel.
    """
    found: list[str] = []
    for _stage_label, synonyms in _FUNNEL_STAGE_VOCAB:
        for col in df.columns:
            if col in exclude:
                continue
            if col.lower() in synonyms:
                found.append(col)
                break
    return found if len(found) >= _MIN_FUNNEL_STAGES else []


# Ratio-metric detection (numerator/denominator column pairs) — see
# `detect_ratio_metric_candidates` below. A ratio like
# conversions/users carries numerator+denominator information a
# single pre-computed continuous number (0.08, 0.095, ...) would
# lose; downstream code decides whether the existing binary/
# proportion path (using the raw counts) or a continuous path fits —
# this module only reports the plausible pair, same as every other
# "*_candidates" heuristic here.
_RATIO_RATE_NUMERATORS = {
    "conversions": "conversion_rate",
    "conversion_count": "conversion_rate",
    "purchases": "purchase_rate",
    "orders": "order_rate",
    "clicks": "click_rate",
    "opens": "open_rate",
    "signups": "signup_rate",
    "responses": "response_rate",
    "successes": "success_rate",
}
_RATIO_MONETARY_NUMERATORS = {"revenue", "spend", "gmv", "order_value_total", "total_spend"}
_RATIO_DENOMINATOR_CANDIDATES = {
    "users", "sessions", "impressions", "visits", "customers",
    "n", "exposures", "sample_size",
}


def detect_ratio_metric_candidates(df: pd.DataFrame, exclude: set[str]) -> list["RatioMetricCandidate"]:
    """
    Detect plausible ratio metrics (a numerator count/monetary column
    paired with a denominator exposure/population column both present
    in this dataset) by naming heuristic — e.g. a dataset shaped like

        variant | users | conversions

    reports conversion_rate = conversions/users; a dataset shaped like

        variant | revenue | users

    reports revenue_per_user = revenue/users. Naming/shape match ONLY:
    this never computes the ratio, never decides which statistical
    engine path is appropriate for it (a rate-style ratio like
    conversions/users usually keeps the existing binary/proportion
    path using the raw counts; a monetary ratio like revenue/users is
    usually a genuine continuous metric) — that decision is entirely
    downstream. Structural columns (id/variant/metric/event/timestamp/
    covariate — passed in via `exclude`) are never proposed as either
    side of a ratio.
    """
    present = {col.lower(): col for col in df.columns if col not in exclude}
    denominators = [present[d] for d in _RATIO_DENOMINATOR_CANDIDATES if d in present]
    if not denominators:
        return []

    candidates: list[RatioMetricCandidate] = []
    for numerator_key, metric_name in _RATIO_RATE_NUMERATORS.items():
        if numerator_key in present:
            numerator = present[numerator_key]
            denominator = denominators[0]
            candidates.append(RatioMetricCandidate(
                metric_name=metric_name, numerator=numerator, denominator=denominator,
                ratio_definition=f"{numerator} / {denominator}",
            ))
    for numerator_key in _RATIO_MONETARY_NUMERATORS:
        if numerator_key in present:
            numerator = present[numerator_key]
            denominator = denominators[0]
            denom_singular = denominator[:-1] if denominator.lower().endswith("s") else denominator
            candidates.append(RatioMetricCandidate(
                metric_name=f"{numerator}_per_{denom_singular}", numerator=numerator, denominator=denominator,
                ratio_definition=f"{numerator} / {denominator}",
            ))
    return candidates


def detect_stratification_candidates(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    """
    Low-cardinality, non-structural columns that could plausibly
    serve as a stratification dimension.

    Naming/shape heuristic ONLY, mirroring the existing conservative
    style of this module: a column qualifies if it (a) isn't already
    playing a structural role (user id / variant / metric / event /
    timestamp / covariate — all passed in via `exclude`), and (b) has
    between 2 and `_MAX_STRATIFICATION_CANDIDATE_CARDINALITY` distinct
    non-null values, so a free-text or near-unique column (e.g. an
    email address) is never proposed as a stratification dimension.
    This never asserts eligibility — see
    `app.stats.stratification.check_stratification_eligibility` for
    the actual gate.
    """
    candidates = []
    for col in df.columns:
        if col in exclude:
            continue
        if col.lower() in (_CUPED_COVARIATE_CANDIDATES):
            continue
        nunique = df[col].dropna().nunique()
        if 2 <= nunique <= _MAX_STRATIFICATION_CANDIDATE_CARDINALITY:
            candidates.append(col)
    return candidates


def _keyword_matches_column(column_lower: str, keyword: str) -> bool:
    """
    Word-boundary match, not plain substring: a keyword must appear as
    its own `_`-delimited token (or the whole column name), never as a
    fragment inside a longer unrelated word. Without this, a short
    generic keyword like "visit" would false-positive on `visitor_id`
    (a structural identifier, never a guardrail metric) purely because
    "visit" is a substring of "visitor". `_` is the only separator
    considered, matching this codebase's column-naming convention
    (snake_case) everywhere else in this module.
    """
    pattern = rf"(?:^|_){re.escape(keyword)}(?:_|\d|$)"
    return bool(re.search(pattern, column_lower))


def detect_guardrail_candidates(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    """
    Numeric columns whose NAME matches common guardrail-metric
    vocabulary
    (`_GUARDRAIL_METRIC_KEYWORDS`), e.g. `latency_ms`, `error_rate`,
    `churn_30d`, `purchase_amount`, `revenue_gbp`.

    Naming heuristic ONLY: this never runs the actual guardrail
    statistical check itself, and it is a SUGGESTION mechanism only —
    entirely independent of what the user actually requested. See
    `resolve_guardrail_metrics` below for the deterministic resolution
    of the user's OWN explicit guardrail request
    (`AnalysisSettings.guardrail_metrics`), a different concept that
    must never be conflated with this one. Returns humanized labels,
    matching the convention already used by `detect_available_metrics`.

    Matching uses `_keyword_matches_column` (word-boundary), not a
    plain substring check, to avoid false positives like `visitor_id`
    matching the "visit" keyword.
    """
    candidates = []
    for col in df.columns:
        if col in exclude:
            continue
        low = col.lower()
        matches = _lower_is_better_match(low) or any(
            _keyword_matches_column(low, keyword) for keyword in _GUARDRAIL_HIGHER_IS_BETTER_KEYWORDS
        )
        if matches and pd.api.types.is_numeric_dtype(df[col]):
            candidates.append(humanize_metric_label(col))
    return list(dict.fromkeys(candidates))


def build_metric_column_map(df: pd.DataFrame, exclude: set[str]) -> dict[str, str]:
    """
    Map humanized metric label (e.g. "Purchase Amount") -> the actual
    raw dataframe column it came from (e.g. "purchase_amount"), for
    every numeric outcome column not in `exclude` — same eligibility
    rule as `detect_available_metrics`, just exposing the reverse
    mapping that function throws away, so a resolved guardrail name
    can be traced back to the column a hypothesis test actually runs
    on.

    Deterministic 1:1 mapping: if two raw columns would humanize to
    the same label, the first one encountered wins (same de-dup
    behavior `detect_available_metrics` already has via
    `dict.fromkeys`) — this never silently merges two different
    columns under one label.
    """
    mapping: dict[str, str] = {}
    for col in df.columns:
        if col in exclude:
            continue
        low = col.lower()
        if low in (
            _EVENT_COLUMN_CANDIDATES
            + _TIMESTAMP_COLUMN_CANDIDATES
            + _CUPED_COVARIATE_CANDIDATES
        ):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            label = humanize_metric_label(col)
            mapping.setdefault(label, col)
    return mapping


def resolve_guardrail_metrics(
    requested: list[str],
    available_metrics: list[str],
    primary_metric_label: str | None = None,
):
    """
    Deterministic, EXACT-MATCH-ONLY (case/whitespace-insensitive)
    resolution of user-requested guardrail names against this
    dataset's actual available metric labels.

    Never fuzzy, never semantic — "Revenue" never resolves to
    "Purchase Amount" and "Bounce Rate" never resolves to "Bounced"
    just because they're related concepts; only a literal
    (normalized) string match counts. If the project later adds an
    explicit deterministic alias table, it plugs in here — there is
    none today, so none is invented.

    The primary metric is never eligible as its own guardrail (a
    metric can't guard itself), regardless of how it was matched.

    Duplicate requests (after trim/case-fold) are resolved once and
    de-duplicated in the returned list, in the order first requested.

    Returns a list[GuardrailResolution] (imported locally to avoid a
    schemas -> stats -> schemas import cycle at module load time).
    """
    from app.schemas.guardrails import GuardrailResolution

    normalized_available: dict[str, str] = {}
    for label in available_metrics:
        if primary_metric_label is not None and label == primary_metric_label:
            continue
        normalized_available[label.strip().lower()] = label

    resolutions: list[GuardrailResolution] = []
    seen: set[str] = set()
    for raw_name in requested:
        clean = (raw_name or "").strip()
        if not clean or clean.lower() in seen:
            continue
        seen.add(clean.lower())
        matched_label = normalized_available.get(clean.lower())
        resolutions.append(
            GuardrailResolution(
                requested_name=clean,
                resolved=matched_label is not None,
                resolved_metric_label=matched_label,
            )
        )
    return resolutions


def _lower_is_better_match(column_lower: str) -> bool:
    """
    Plain substring match against `_GUARDRAIL_LOWER_IS_BETTER_KEYWORDS`
    only — kept as the ORIGINAL (pre-existing, already covered by
    tests) matching rule for this specific vocabulary, e.g. so
    `bounced` still matches `bounce`. Word-boundary matching
    (`_keyword_matches_column`) is applied only to the NEWLY added
    vocabularies (`_GUARDRAIL_HIGHER_IS_BETTER_KEYWORDS`,
    `_DEMOGRAPHIC_ATTRIBUTE_KEYWORDS`), where a short generic token
    like "visit" genuinely risks a false positive on an unrelated
    identifier column (`visitor_id`) that this original vocabulary
    never had to worry about.
    """
    return any(keyword in column_lower for keyword in _GUARDRAIL_LOWER_IS_BETTER_KEYWORDS)


def infer_guardrail_direction(column_name: str) -> bool:
    """
    Deterministic "higher is better?" heuristic for a guardrail metric
    column, keyed off `_GUARDRAIL_LOWER_IS_BETTER_KEYWORDS` only — that
    list is, by definition, made up of "bad when it goes up" concepts
    (bounce rate, churn, error rate, latency, complaints, refunds,
    cancellations...). Any column matching it is lower_is_better; every
    other column — including anything matching
    `_GUARDRAIL_HIGHER_IS_BETTER_KEYWORDS` (revenue, purchase amount,
    session duration...) as well as anything matching neither list —
    defaults to higher_is_better=True, the same default every other
    metric in this system already assumes implicitly. Word-boundary
    matched via `_keyword_matches_column`, same as candidate detection.
    Purely a naming heuristic — never inferred from the observed
    magnitude, and never decided by the LLM.
    """
    low = column_name.lower()
    return not _lower_is_better_match(low)


def infer_metric_type(df: pd.DataFrame, metric_col: str) -> MetricType:
    """Public wrapper around `_infer_metric_type`, for callers outside this module (e.g. guardrail_node.py)."""
    return _infer_metric_type(df, metric_col)


def detect_covariate_candidates(df: pd.DataFrame, exclude: set[str] | None = None) -> list[str]:
    """
    ALL numeric columns matching the CUPED/pre-experiment covariate
    vocabulary
    (`_CUPED_COVARIATE_CANDIDATES`), for display purposes.

    Distinct from `detect_cuped_covariate`, which intentionally returns
    a single column (and only when exactly one match exists) because
    it feeds the CUPED variance-reduction computation, where an
    ambiguous match must not be silently guessed. This function has a
    different job — listing what the classifier noticed — so it
    returns every match rather than requiring uniqueness.
    """
    excluded = exclude or set()
    names = {name.lower() for name in _CUPED_COVARIATE_CANDIDATES}
    return [
        c for c in df.columns
        if c not in excluded and c.lower() in names and pd.api.types.is_numeric_dtype(df[c])
    ]


def detect_variant_values(df: pd.DataFrame, variant_col: str | None) -> list[str]:
    """
    Distinct values observed in the variant column, as strings, in a
    deterministic (sorted) order — same ordering convention as
    `resolve_control_label`'s fallback, so this list's order is stable
    across runs on the same data. Empty list if there's no variant
    column.
    """
    if variant_col is None:
        return []
    values = df[variant_col].dropna().unique().tolist()
    return sorted((str(v) for v in values), key=str)


def classify_dataset(
    df: pd.DataFrame,
    preferred_metric: str | None = None,
    assignment_df: pd.DataFrame | None = None,
) -> DatasetInfo:
    """
    Classify a dataset and extract summary info for the UI's detection
    banner. Deterministic — same input always produces the same output
    for a given `preferred_metric` (which itself must be plain request
    text matched against column names, never an LLM interpretation —
    see `_select_metric_column`).

    `assignment_df` (optional): a separate ``user_id | variant``-shaped
    dataset. When provided, it is deterministically merged onto `df`
    via `enrich_with_assignment` BEFORE any detection runs, so a
    primary dataset with no variant column of its own (e.g. ``user_id |
    order_value``) is still classified using the merged variant column.
    `assignment_df=None` (the default) leaves this function's behavior
    completely unchanged from before this parameter existed.
    """
    if df.empty or len(df.columns) == 0:
        raise DatasetClassificationError("Dataset is empty — cannot classify.")

    working_df = enrich_with_assignment(df, assignment_df)

    user_col = _detect_user_column(working_df)
    variant_col = _detect_variant_column(working_df)
    aggregate_size_col = _detect_aggregate_size_column(working_df) if user_col is None else None

    structural = _structural_vote(working_df, user_col)
    quantitative = _quantitative_vote(working_df, user_col)

    if user_col is None:
        # A dataset with no experiment-unit identifier can still
        # be a genuinely recognizable pre-aggregated A/B summary (e.g.
        # `variant | users | conversions`), rather than an unrecognizable
        # format. Recognize the SHAPE correctly here; whether the
        # downstream analysis pipeline can actually run on it is a
        # separate, honest question handled by detect_experiment_columns.
        if variant_col is not None and aggregate_size_col is not None and structural in (None, DatasetType.AGGREGATED_AB_TEST):
            dataset_type = DatasetType.AGGREGATED_AB_TEST
        else:
            dataset_type = DatasetType.UNKNOWN
    elif structural is not None and quantitative is not None:
        dataset_type = structural if structural == quantitative else DatasetType.UNKNOWN
    elif structural is not None:
        dataset_type = structural
    elif quantitative is not None:
        dataset_type = quantitative
    else:
        dataset_type = DatasetType.UNKNOWN

    variants = int(working_df[variant_col].nunique()) if variant_col is not None else 0
    if user_col is not None:
        users = int(working_df[user_col].nunique())
    elif aggregate_size_col is not None:
        # Report the real aggregate denominator (sum of per-arm
        # sample sizes), not the row count, which would undercount by
        # (arms - 1)x for a `variant | users | conversions`-shaped summary.
        users = int(working_df[aggregate_size_col].sum())
    else:
        users = int(len(working_df))

    exclude = {c for c in [user_col, variant_col] if c is not None}
    exclude |= {c for c in working_df.columns if c.lower() in _EVENT_COLUMN_CANDIDATES + _TIMESTAMP_COLUMN_CANDIDATES}
    metric_col, metric_selection_reason = _select_metric_column(working_df, exclude=exclude, preferred_metric=preferred_metric)
    metric_label = humanize_metric_label(metric_col)
    available_metrics = detect_available_metrics(working_df, exclude=exclude)

    # Everything below is additive and derived purely from
    # already-computed deterministic
    # facts (user_col, variant_col, metric_col, available_metrics)
    # plus the same column vocabularies used elsewhere in this module.
    # None of it changes dataset_type, variants, users, metric_label,
    # available_metrics, or metric_selection_reason above.
    variant_values = detect_variant_values(working_df, variant_col)
    additional_metrics = [m for m in available_metrics if m != metric_label]

    role_exclude = set(exclude)
    if metric_col is not None:
        role_exclude.add(metric_col)
    stratification_candidates = detect_stratification_candidates(working_df, exclude=role_exclude)
    guardrail_candidates = detect_guardrail_candidates(working_df, exclude=role_exclude)
    covariate_candidates = detect_covariate_candidates(working_df, exclude=role_exclude)

    # Experiment-unit-level classification (CRM/marketing datasets task)
    # — purely additive, computed once here from the same `working_df`
    # already used above. Never changes dataset_type/variants/users/
    # metric_label/available_metrics/metric_selection_reason.
    grain = classify_experiment_unit_level(working_df)
    pre_treatment_candidates = detect_pre_treatment_segmentation_candidates(working_df, exclude=role_exclude)
    excluded_post_treatment = detect_post_treatment_exclusions(working_df, exclude=role_exclude)
    # Deliberately uses `exclude` (structural columns only), NOT
    # `role_exclude` — the chosen primary metric is very often exactly
    # a funnel stage itself (e.g. Hillstrom's 'conversion' is both the
    # primary metric AND the funnel's final stage). Excluding it here
    # silently blinds this detector to the most common real case: it
    # would only ever see whichever OTHER stage columns exist besides
    # the one already picked as primary, so a real 2-stage funnel
    # (visit -> conversion) reports as if only 1 stage existed. Same
    # bug shape, and same fix, as ratio-candidate detection above.
    funnel_metrics = detect_funnel_metrics(working_df, exclude=exclude)
    # Deliberately excludes structural columns only (user_id/variant),
    # NOT metric_col — the chosen primary metric is very often exactly
    # the ratio's numerator (e.g. 'conversions' in a users/conversions
    # dataset), so excluding it here would silently blind this
    # detector to the most common real case.
    ratio_metric_candidates = detect_ratio_metric_candidates(working_df, exclude=exclude)

    return DatasetInfo(
        type=dataset_type,
        variants=variants,
        users=users,
        metric_label=metric_label,
        available_metrics=available_metrics,
        metric_selection_reason=metric_selection_reason,
        user_id_column=user_col,
        variant_column=variant_col,
        variant_values=variant_values,
        primary_metric=metric_label,
        additional_metrics=additional_metrics,
        stratification_candidates=stratification_candidates,
        guardrail_candidates=guardrail_candidates,
        covariate_candidates=covariate_candidates,
        experiment_unit_level=grain.level,
        unit_identifier=grain.unit_identifier,
        unit_level_confidence=grain.confidence,
        unit_level_evidence=grain.evidence,
        unit_level_blocking_reason=grain.blocking_reason,
        pre_treatment_segmentation_candidates=pre_treatment_candidates,
        excluded_post_treatment_columns=excluded_post_treatment,
        funnel_metrics=funnel_metrics,
        ratio_metric_candidates=ratio_metric_candidates,
    )


def describe_dataset_structure(dataset_info: "DatasetInfo", columns_resolved: bool) -> str:
    """
    Human-facing description of dataset STRUCTURE, for report/execution-
    step text — distinct from `dataset_info.type`, which stays exactly
    as classify_dataset() computed it (no change to that value, no
    change to the API contract).

    This exists to fix one specific contradiction: `classify_dataset`'s
    structural-vs-quantitative vote can land on DatasetType.UNKNOWN for
    a dataset that `detect_experiment_columns` nonetheless *fully*
    resolved (a real user_col + variant_col + metric_col, i.e.
    `columns_resolved=True`). In that case, showing the bare "Unknown /
    Unsupported Format" label next to a report that just ran SRM/
    quality checks/variant-conflict detection on that exact structure
    is misleading — the structure IS recognized, it's the coarse
    two-heuristic vote that's ambiguous (e.g. a dataset with both
    event-like and outcome-like columns).

    When `dataset_info.type` is anything other than UNKNOWN, or when
    `columns_resolved` is False (genuinely not resolvable — e.g.
    Dataset 3's no-user-id case), this returns the original
    `dataset_info.type.value` unchanged, so every dataset that already
    classifies cleanly keeps the exact same wording it had before.
    """
    if dataset_info.type != DatasetType.UNKNOWN or not columns_resolved:
        return dataset_info.type.value

    return (
        f"Aggregated/raw experiment data with identifiable variants and a "
        f"{dataset_info.metric_label.lower()} outcome (a user-level experiment "
        f"unit, variant column, and metric column were all resolved)"
    )


def describe_unit_count(dataset_info: "DatasetInfo") -> str:
    """
    Phrase the dataset's row count using terminology that matches how
    the experimental unit was actually established. Report
    text must never say "N users" when no customer/user identifier
    exists at all — e.g. a CRM export classified as unit-level via
    `implicit_row` (one row = one randomized unit, no customer_id
    column). Every other case (an explicit id column, or an
    unresolved/legacy dataset) keeps the existing "N users" wording
    unchanged.
    """
    if dataset_info.unit_identifier == UnitIdentifierType.IMPLICIT_ROW:
        return f"{dataset_info.users:,} experimental units (no explicit customer identifier available)"
    return f"{dataset_info.users:,} users"


def describe_unit_level_detection(dataset_info: "DatasetInfo") -> str:
    """
    Context-aware, user-facing explanation of the experiment-unit-level
    classification — replaces a bare, confusing "no
    experiment-unit identifier found" for datasets that are actually
    valid unit-level experiments, and gives event-level-without-id
    datasets a clear, honest explanation instead. Returns "" when
    neither case applies (nothing extra to say beyond the existing
    dataset description).
    """
    if (
        dataset_info.experiment_unit_level == ExperimentUnitLevel.UNIT_LEVEL
        and dataset_info.unit_identifier == UnitIdentifierType.IMPLICIT_ROW
    ):
        return (
            "**Unit-level experiment detected.**\n\n"
            "No explicit customer identifier was found, but the dataset appears to "
            "contain one row per randomized experimental unit. Row-level "
            "observations can therefore be used as experimental units.\n\n"
            f"Assignment: `{dataset_info.variant_column}`\n"
            f"Variants: {dataset_info.variants}\n"
            f"Primary outcome: `{dataset_info.metric_label}`\n"
            f"Experimental units: {dataset_info.users:,}"
        )
    if (
        dataset_info.experiment_unit_level == ExperimentUnitLevel.EVENT_LEVEL
        and dataset_info.unit_identifier == UnitIdentifierType.MISSING
    ):
        return (
            "**Event-level data detected, but the experimental unit cannot be identified.**\n\n"
            "Multiple observations may belong to the same customer, so an explicit "
            "customer/user identifier is required before reliable user-level "
            "experiment analysis can be performed."
        )
    return ""


class ExperimentColumns:
    """
    Resolved column roles + inferred metric type, needed by every
    downstream stats node (validation, experiment). Kept separate from
    `DatasetInfo` (which is the frontend-facing summary) because this
    carries actual column NAMES, an implementation detail the frontend
    never needs to see.
    """

    def __init__(
        self,
        user_col: str,
        variant_col: str,
        metric_col: str,
        metric_type: MetricType,
        is_implicit_unit: bool = False,
    ):
        self.user_col = user_col
        self.variant_col = variant_col
        self.metric_col = metric_col
        self.metric_type = metric_type
        # True when `user_col` is the synthetic implicit-row identifier
        # (see `_IMPLICIT_UNIT_ID_COLUMN`) rather than a real column from
        # the source data — i.e. this dataset was confirmed unit-level
        # (one row = one randomized unit) with no explicit customer id.
        # Purely informational for reporting/narrative; every stats
        # computation already treats `user_col` uniformly either way.
        self.is_implicit_unit = is_implicit_unit


_MONETARY_KEYWORDS = ["revenue", "value", "amount", "price", "aov", "spend"]


def detect_experiment_columns(
    df: pd.DataFrame,
    preferred_metric: str | None = None,
    assignment_df: pd.DataFrame | None = None,
) -> ExperimentColumns:
    """
    Resolve the user/variant/metric columns and infer the metric's
    `MetricType`, for use by validation_node / experiment_node. Raises
    if the dataset doesn't have the minimum structure an A/B analysis
    requires (user + variant columns) — this is a harder requirement
    than classify_dataset's best-effort reporting, since these nodes
    can't proceed at all without real column references.

    `preferred_metric` (typically the user's own request text) is used
    the same deterministic way as in `classify_dataset` — see
    `_select_metric_column` — so the column the hypothesis test actually
    runs on always matches what `DatasetInfo.metric_selection_reason`
    told the user would be analyzed.

    `assignment_df` (optional): same ``user_id | variant``-shaped
    dataset accepted by `classify_dataset` — merged onto `df` via
    `enrich_with_assignment` BEFORE user/variant/metric detection, so
    the resolved variant column can come from the merged assignment
    table when the primary dataset has none of its own.
    `assignment_df=None` (the default) leaves this function's behavior
    completely unchanged from before this parameter existed.
    """
    working_df = enrich_with_assignment(df, assignment_df)

    user_col = _detect_user_column(working_df)
    variant_col = _detect_variant_column(working_df)
    used_implicit_unit = False
    if user_col is None:
        # CRM/marketing unit-level dataset support — reuse the SAME
        # multi-signal grain classifier used by `classify_dataset`
        # (single source of truth, per the project's "reuse the
        # existing architecture" rule) to decide whether the absence of
        # an id column is actually fine (a genuine unit-level dataset,
        # one row = one randomized unit) or a real blocker.
        grain = classify_experiment_unit_level(working_df)

        if grain.level == ExperimentUnitLevel.UNIT_LEVEL and grain.unit_identifier == UnitIdentifierType.IMPLICIT_ROW:
            # Confirmed unit-level: attach a synthetic, unique-by-
            # construction row-id column so every existing user_col-
            # keyed code path downstream (dedup, duplicate-conflict
            # analysis, SRM, segmentation) keeps working completely
            # unchanged — dedup against a column with no duplicate
            # values by construction is a provable no-op, so this never
            # silently drops or fabricates rows. Do NOT invent a
            # user_id to work around a MISSING/blocked identifier
            # (Case A below) — this branch only fires once the grain
            # classifier has already gathered real evidence.
            working_df, implicit_col = attach_implicit_unit_id(working_df)
            user_col = implicit_col
            used_implicit_unit = True
        else:
            # Distinguish honest failure reasons instead of one
            # generic message:
            #   (a) a recognizable pre-aggregated A/B summary (variant +
            #       a sample-size denominator column) was found, but this
            #       pipeline's experiment inference requires individual-
            #       level observations to run the existing hypothesis-test
            #       machinery — this is a capability limitation, not a
            #       malformed dataset, and must be reported as such, never
            #       as a misleading "no user_id"/malformed-data error.
            #   (b) a variant column and an apparent rate-only outcome
            #       (e.g. "click_rate") were found, but with no sample-size
            #       column at all, so not even the aggregate denominator is
            #       known — this is a distinct, more specific problem than
            #       (c) below (an outright missing/unresolvable experiment
            #       unit — either genuinely event-level data with no id,
            #       or not enough structure to tell).
            # Do NOT build a new aggregated statistical engine to make (a)
            # "work" — see module docstring.
            if variant_col is not None:
                aggregate_size_col = _detect_aggregate_size_column(working_df)
                if aggregate_size_col is not None:
                    arm_count = working_df[variant_col].nunique()
                    total_n = int(working_df[aggregate_size_col].sum())
                    raise DatasetClassificationError(
                        f"Aggregated A/B data was recognized ({arm_count} arms, {total_n:,} total "
                        f"users via '{aggregate_size_col}'), but the current experiment-analysis "
                        "path requires individual-level (one-row-per-user) observations to run its "
                        "hypothesis tests. Re-export this data with one row per user (or per "
                        "randomized unit) to analyze it here."
                    )
                has_rate_like_column = any(
                    "rate" in c.lower() or "ratio" in c.lower() for c in working_df.columns if c != variant_col
                )
                if has_rate_like_column:
                    raise DatasetClassificationError(
                        "A variant column and a rate-style outcome were found, but this dataset has "
                        "no sample-size column (e.g. 'users', 'sample_size', 'n') to determine how "
                        "many observations each rate represents, so even the aggregate denominator "
                        "cannot be recovered. Add a sample-size column per arm, or provide "
                        "individual-level observations instead."
                    )
            if grain.level == ExperimentUnitLevel.EVENT_LEVEL:
                # Case A (event-level): this is real, potentially useful
                # event-level/behavioral data — it just isn't analyzable
                # as a user-level A/B experiment without an explicit
                # identifier. Do NOT invent a user_id to work around
                # this — this protection must never be weakened.
                raise DatasetClassificationError(
                    "Event-level data detected, but the experimental unit cannot be "
                    "identified. Multiple observations may belong to the same "
                    "customer, so an explicit experiment-unit identifier (e.g. "
                    "user_id, customer_id, visitor_id) is required before reliable "
                    "unit-level experiment analysis can be performed — row count "
                    "alone is not a substitute for a real experiment-unit ID."
                )
            # Case A (unknown grain): no experiment-unit identifier, and
            # not enough recognizable experiment structure (variant +
            # outcome columns) to safely treat each row as one
            # randomized unit either. Do NOT invent a user_id or assume
            # unit-level to work around this.
            raise DatasetClassificationError(
                "No experiment-unit identifier (e.g. user_id, customer_id, visitor_id) "
                "was found, and this dataset does not have enough recognizable "
                "experiment structure (a variant/treatment column and an outcome "
                "column) to safely treat each row as one randomized experimental "
                "unit. Add an explicit identifier, or provide a variant column plus "
                "an outcome column, to analyze this data — row count alone is not a "
                "substitute for a real experiment-unit ID."
            )
    if variant_col is None:
        raise DatasetClassificationError(
            "Cannot run experiment analysis: missing a recognizable variant/group column."
        )

    arm_values = working_df[variant_col].dropna().unique().tolist()
    if len(arm_values) < 2:
        # Case B: the experiment unit is real, but there's only one
        # treatment arm present (e.g. every row has group="exp") — so
        # there's nothing to compare it against. Do NOT invent a second
        # variant/control group to make this "work".
        only_value = arm_values[0] if arm_values else "(no values)"
        raise DatasetClassificationError(
            f"Experiment unit identified ({user_col}), but only one treatment arm "
            f'is present ("{variant_col}" = {only_value!r} for every row). A '
            "control-vs-variant comparison cannot be performed without a second arm."
        )

    exclude = {user_col, variant_col}
    exclude |= {c for c in working_df.columns if c.lower() in _EVENT_COLUMN_CANDIDATES + _TIMESTAMP_COLUMN_CANDIDATES}
    metric_col, _reason = _select_metric_column(working_df, exclude=exclude, preferred_metric=preferred_metric)
    if metric_col is None:
        raise DatasetClassificationError(
            "Cannot run experiment analysis: no recognizable outcome/metric column found."
        )

    metric_type = _infer_metric_type(working_df, metric_col)
    return ExperimentColumns(
        user_col=user_col,
        variant_col=variant_col,
        metric_col=metric_col,
        metric_type=metric_type,
        is_implicit_unit=used_implicit_unit,
    )


def _infer_metric_type(df: pd.DataFrame, metric_col: str) -> MetricType:
    values = df[metric_col].dropna()
    unique_values = set(values.unique().tolist())
    if unique_values <= {0, 1} or unique_values <= {0.0, 1.0}:
        return MetricType.BINARY
    if any(keyword in metric_col.lower() for keyword in _MONETARY_KEYWORDS):
        return MetricType.CONTINUOUS_MONETARY
    return MetricType.CONTINUOUS_GENERAL


def resolve_control_label(df: pd.DataFrame, variant_col: str) -> str:
    """
    Resolve which value in the variant column represents "control".

    Priority order:
    1. Exact case-insensitive match on 'control'.
    2. A known holdout/no-treatment label (see `_HOLDOUT_LABELS`) —
       CRM/marketing experiments commonly hold out a "No E-Mail" /
       "No Campaign" arm instead of literally naming it "control".
       Without this step, a real CRM dataset like Hillstrom's
       (arms: "Mens E-Mail", "No E-Mail", "Womens E-Mail") would fall
       all the way through to the alphabetical fallback below, which
       picks "Mens E-Mail" — a treatment arm — as "control", making
       every downstream pairwise comparison use the wrong baseline.
    3. The first-seen value, alphabetically — unchanged fallback so
       datasets using arbitrary labels (e.g. 'A'/'B') still resolve
       deterministically.

    Shared by validation_node and experiment_node — both need the same
    control/variant split of the same dataset.
    """
    values = df[variant_col].dropna().unique().tolist()
    for v in values:
        if str(v).strip().lower() == "control":
            return v
    for v in values:
        normalized = re.sub(r"[\s_-]+", " ", str(v).strip().lower())
        if normalized in _HOLDOUT_LABELS:
            return v
    return sorted(values, key=str)[0]


def count_duplicate_user_rows(df: pd.DataFrame, user_col: str) -> int:
    """
    Number of EXTRA rows beyond one-per-user (e.g. a dataset with 5
    duplicate user_id rows returns 5, not the number of affected
    users). Used both for the duplicate-rows QualityCheck and to
    decide whether deduplication is needed before computing sample
    size, so "290,584 users" (the classifier's unique-user count) and
    "N users observed" (used in power analysis) never disagree — they
    were computed two different ways (nunique() vs row count) before
    this fix, which is exactly the kind of inconsistency that erodes
    trust in a decision-support tool.
    """
    return int(df.duplicated(subset=[user_col]).sum())


def deduplicate_by_user(df: pd.DataFrame, user_col: str) -> pd.DataFrame:
    """
    Keeps the FIRST row per user_id. Applied consistently by every
    node that computes sample-size-dependent statistics, so "sample
    size" always means the same thing (unique users) everywhere in the
    report — matching DatasetInfo.users from classify_dataset(), which
    was already nunique()-based.
    """
    return df.drop_duplicates(subset=[user_col], keep="first")


class DuplicateUserAnalysis:
    """
    Classifies duplicate user_id rows into three severities, per the
    explicit rule: a duplicate is harmless if it's the same user
    assigned to the same variant (arbitrary which row you keep); it is
    a SEVERE quality issue if the same user_id is assigned to
    DIFFERENT variants (this isn't a duplicate, it's a broken
    randomization/assignment pipeline — a variant crossover is exactly
    the kind of thing that can fabricate an arbitrary effect size, the
    same trust concern SRM exists to catch); and it's a lesser,
    disclosed-but-non-blocking issue if the same user+variant has
    conflicting metric values (still ambiguous which value is
    "correct," but doesn't corrupt the randomization itself).
    """

    def __init__(self, duplicate_row_count: int, conflicting_variant_users: int, conflicting_metric_users: int):
        self.duplicate_row_count = duplicate_row_count
        self.conflicting_variant_users = conflicting_variant_users
        self.conflicting_metric_users = conflicting_metric_users

    @property
    def has_severe_conflict(self) -> bool:
        return self.conflicting_variant_users > 0


def analyze_duplicate_users(df: pd.DataFrame, user_col: str, variant_col: str, metric_col: str) -> DuplicateUserAnalysis:
    """
    See `DuplicateUserAnalysis` docstring for the severity rule.
    `conflicting_metric_users` is only computed among users WITHOUT a
    variant conflict, to avoid double-counting the same root cause
    under two different labels.
    """
    duplicate_row_count = count_duplicate_user_rows(df, user_col)

    variant_counts_per_user = df.groupby(user_col)[variant_col].nunique()
    conflicting_variant_users = int((variant_counts_per_user > 1).sum())

    single_variant_user_ids = variant_counts_per_user[variant_counts_per_user == 1].index
    single_variant_df = df[df[user_col].isin(single_variant_user_ids)]
    metric_counts_per_user = single_variant_df.groupby(user_col)[metric_col].nunique()
    conflicting_metric_users = int((metric_counts_per_user > 1).sum())

    return DuplicateUserAnalysis(
        duplicate_row_count=duplicate_row_count,
        conflicting_variant_users=conflicting_variant_users,
        conflicting_metric_users=conflicting_metric_users,
    )
