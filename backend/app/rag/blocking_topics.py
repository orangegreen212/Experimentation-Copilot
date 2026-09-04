"""
Decision-aware / blocking-reason-aware evidence topics

PROBLEM THIS FIXES: `knowledge_base_node.py` retrieves against a
generic, always-the-same "core review concepts" query (SRM, MDE,
power, significance, segmentation, ...) so it can score well against
*some* dataset-driven review even before anyone knows whether the
experiment will turn out VALID or INVALID (see that module's docstring
— it runs concurrently with `validation`, so it structurally CANNOT
know the decision yet). That's fine when the experiment is VALID: any
of those generic methodology topics may genuinely be relevant context.

It stops being fine once the experiment is INVALID. At that point
there is exactly one story the report needs to tell: *why* it's
INVALID. A chunk about Minimum Detectable Effect scoring reasonably
against the generic query is not evidence for an Outlier-Detection
failure just because it was retrieved — it's evidence for a question
nobody is asking anymore.

This module is the single, deterministic place that answers "given
this specific blocking failure, what topic must the shown evidence
actually be about?" It is intentionally kept OUT of
`app/graph/nodes/knowledge_base_node.py` (which only ever sees the
dataset, never the finished decision — see that module's "PARALLEL-
SAFE RETURN" docstring) and lives instead alongside
`app/graph/report_generator.py`'s evidence-assembly step, which is the
first point in the pipeline that has BOTH the retrieved candidates and
the final decision in hand.

Nothing here computes or changes `ExperimentValidity` / `Decision` —
`blocking_topic()` below only ever reads back WHICH of
`experiment_validity()`'s own three INVALID conditions fired, in the
exact same order that function already checks them, so this can never
diverge from or duplicate that logic.
"""

from __future__ import annotations

from app.schemas.quality import QualityCheck

# --- canonical topic keys -------------------------------------------------

SRM = "SRM"
CONFLICTING_VARIANT_ASSIGNMENT = "CONFLICTING_VARIANT_ASSIGNMENT"
OUTLIER_DETECTION = "OUTLIER_DETECTION"
# Generic bucket for any OTHER critical quality-check failure this
# project adds in the future that isn't specifically outlier detection
# — deliberately narrow (just "data quality" terms) rather than
# guessing at a more specific topic this module doesn't know about.
CRITICAL_DATA_QUALITY = "CRITICAL_DATA_QUALITY"

# Human-readable label for each topic key — used both for the
# retrieval-topic name in logs/tests and as the "blocking issue" name
# shown in the report when no relevant evidence is found. For the
# CRITICAL_DATA_QUALITY generic bucket, callers should prefer the real
# `QualityCheck.label` they already have over this generic name — see
# `blocking_topic()` below, which does exactly that.
TOPIC_LABELS: dict[str, str] = {
    SRM: "Sample Ratio Mismatch (SRM)",
    CONFLICTING_VARIANT_ASSIGNMENT: "Conflicting Variant Assignment",
    OUTLIER_DETECTION: "Outlier Detection",
    CRITICAL_DATA_QUALITY: "Critical Data Quality Issue",
}

# The actual retrieval-relevance keywords per topic (spec section 2).
# Matched as case-insensitive substrings against a candidate chunk's
# heading + content — see `chunk_matches_topic()`. Deliberately does
# NOT include "MDE", "power", "confidence interval", "bootstrap", or
# "practical significance" for any of these three topics: those are
# the generic methodology topics that must NOT dominate evidence for a
# validity-blocking failure (spec section 1).
TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    SRM: (
        "sample ratio mismatch",
        "srm",
        "allocation imbalance",
        "randomization",
    ),
    CONFLICTING_VARIANT_ASSIGNMENT: (
        "conflicting variant assignment",
        "multiple variants per user",
        "treatment assignment integrity",
        "randomization integrity",
    ),
    OUTLIER_DETECTION: (
        "outlier detection",
        "outliers",
        "outlier",
        "extreme observations",
        "extreme observation",
        "extreme value",
        "influential observations",
        "influential observation",
        "data quality",
        "robust analysis",
        "robust statistics",
        "heavy-tailed",
        "heavy tailed",
    ),
    CRITICAL_DATA_QUALITY: (
        "data quality",
        "data integrity",
    ),
}


def topic_key_for_quality_label(label: str) -> str:
    """
    Maps a `QualityCheck.label` (Stage 3 — app/stats/quality_checks.py)
    to one of the canonical topic keys above. Only "Outlier Detection"
    is special-cased today, since it's the one label that currently
    ever sets `critical=True` (see `check_outliers`) — any other/future
    critical quality-check label falls back to the generic
    `CRITICAL_DATA_QUALITY` bucket rather than being silently treated
    as "no topic" (which would let generic methodology leak back in).
    """
    normalized = (label or "").strip().lower()
    if "outlier" in normalized:
        return OUTLIER_DETECTION
    return CRITICAL_DATA_QUALITY


def blocking_topic(
    *,
    srm_passed: bool,
    has_conflicting_variant_duplicates: bool,
    quality_checks: list[QualityCheck],
) -> tuple[str, str, tuple[str, ...]] | None:
    """
    Identifies the SPECIFIC reason an INVALID experiment is INVALID,
    for evidence-retrieval purposes only.

    IMPORTANT: this function does not decide validity itself, and the
    caller is expected to already know `experiment_validity(...) ==
    ExperimentValidity.INVALID` before calling it (see
    `report_generator._decision_blocking_topic`, the only caller).
    It reads back the SAME `srm_passed` / `has_conflicting_variant_
    duplicates` / `quality_checks` fields `experiment_validity()`
    itself branches on, checked in that SAME priority order (SRM,
    then conflicting assignment, then critical quality checks) — so
    this can never identify a "reason" `experiment_validity()` didn't
    actually use, and never needs to duplicate that function's logic
    to stay in sync with it.

    Returns `(topic_key, human_label, keywords)`, or `None` if none of
    these three specific conditions hold (a defensive default — the
    caller should treat `None` as "don't restrict retrieval," never as
    "restrict to nothing").
    """
    if not srm_passed:
        return SRM, TOPIC_LABELS[SRM], TOPIC_KEYWORDS[SRM]

    if has_conflicting_variant_duplicates:
        return (
            CONFLICTING_VARIANT_ASSIGNMENT,
            TOPIC_LABELS[CONFLICTING_VARIANT_ASSIGNMENT],
            TOPIC_KEYWORDS[CONFLICTING_VARIANT_ASSIGNMENT],
        )

    critical_failure = next(
        (qc for qc in quality_checks if (not qc.passed) and qc.critical), None
    )
    if critical_failure is not None:
        key = topic_key_for_quality_label(critical_failure.label)
        # Use the check's own real label (e.g. "Outlier Detection") as
        # the human-facing name rather than the generic bucket name,
        # even when it fell into the CRITICAL_DATA_QUALITY bucket —
        # the report should always be able to name the actual check
        # that failed.
        return key, critical_failure.label, TOPIC_KEYWORDS[key]

    return None


def chunk_matches_topic(heading: str, content: str, keywords: tuple[str, ...]) -> bool:
    """Case-insensitive substring match of any keyword against heading + content."""
    haystack = f"{heading or ''} {content or ''}".lower()
    return any(keyword in haystack for keyword in keywords)
