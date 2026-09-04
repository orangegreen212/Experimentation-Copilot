"""
Decision node — the only node that touches report TEXT generation.
Assembles `ReportFacts` from everything upstream nodes computed, then
delegates to `get_report_generator()` (Strategy pattern — see
graph/report_generator.py). This node has zero knowledge of whether
the underlying generator is template-based or LLM-based.

Phase 2 — this node is also the single place `hypothesis_evaluation`
is computed (via the pure-Python `evaluate_hypothesis()`, never a
LangGraph node of its own — a dedicated node isn't warranted for one
deterministic function call) and the single place both `hypothesis`
and `hypothesis_evaluation` are stamped onto the final
`ExperimentReport`, AFTER the report generator returns, via
`model_copy`. This deliberately bypasses every ReportGenerator
entirely: no template branch and no LLM prompt ever sees or
constructs these two fields, so there's exactly one place to verify
they're populated correctly rather than N generator branches to keep
in sync.
"""

import time

from app.core.logging import get_node_logger
from app.graph.report_generator import ReportFacts, build_emergency_fallback_report, get_report_generator
from app.graph.state import GraphState
from app.schemas.guardrails import GuardrailRequestState, GuardrailResolution
from app.stats.dataset_classifier import resolve_guardrail_metrics
from app.stats.decision_audit import build_decision_audit_trail
from app.stats.decision_narrative import build_decision_narrative
from app.stats.decision_support import build_decision_support
from app.stats.hypothesis_evaluator import evaluate_hypothesis
from app.schemas.stratification import DiagnosticStratification, StratificationResult, StratificationStatus

log = get_node_logger("Decision")


def _resolve_guardrail_facts(state: GraphState) -> tuple[list[str], list[GuardrailResolution], GuardrailRequestState]:
    """
    `guardrail_node` only runs on the "experiment actually ran" path
    (see graph_builder.py) — the SAME scope restriction segmentation/
    stratification already have. When the pipeline short-circuits
    BEFORE `experiment` (SRM failure,
    conflicting-variant duplicates, a critical quality failure, or the
    planner excluding "experiment" from this run), `guardrail_node`
    never executes, so `state["guardrail_request_state"]` etc. are
    simply absent — exactly the same kind of gap `_build_not_run_stratification`
    (above) exists to close for stratification.

    This reconstructs an honest resolution from already-computed facts
    ONLY (`settings.guardrail_metrics` + `dataset.available_metrics` —
    both set long before validation runs) — it never runs a new
    statistical test and never invents `guardrail_results`; a request
    that was never resolved against real columns correctly stays
    unevaluated (guardrail_status stays NOT_AVAILABLE), it just no
    longer gets silently reported as "not specified" when the user did
    specify something.

    Returns the state's own already-computed guardrail facts unchanged
    whenever guardrail_node DID run.
    """
    if "guardrail_request_state" in state:
        return (
            state.get("requested_guardrails", []),
            state.get("guardrail_resolutions", []),
            state["guardrail_request_state"],
        )

    settings = state.get("settings")
    requested = list(getattr(settings, "guardrail_metrics", None) or []) if settings is not None else []
    if not requested:
        return [], [], GuardrailRequestState.NOT_SPECIFIED

    dataset = state.get("dataset")
    available_metrics = dataset.available_metrics if dataset is not None else []
    resolutions = resolve_guardrail_metrics(
        requested,
        available_metrics=available_metrics,
        primary_metric_label=(dataset.metric_label if dataset is not None else None),
    )
    from app.graph.nodes.guardrail_node import derive_guardrail_request_state

    return requested, resolutions, derive_guardrail_request_state(resolutions)


def _try_build_diagnostic_stratification(state: GraphState, column: str) -> DiagnosticStratification | None:
    """
    Best-effort, descriptive-ONLY diagnostic breakdown for `column`,
    computed directly from already-loaded dataset facts (dataset_id +
    experiment_columns, both resolved by classifier_node/validation_node
    before this node ever runs) — never a causal estimate, see
    `build_diagnostic_stratification`'s docstring. Returns None only
    when the dataset/columns aren't available at all (e.g. the
    classifier itself failed), in which case the caller falls back to
    a plain NOT_RUN explanation with no diagnostic block, rather than
    fabricating one.
    """
    dataset_id = state.get("dataset_id")
    columns = state.get("experiment_columns")
    if dataset_id is None or columns is None:
        return None

    try:
        from app.core.dataset_store import get_dataset
        from app.stats.dataset_classifier import deduplicate_by_user, resolve_control_label
        from app.stats.stratification import build_diagnostic_stratification

        df = get_dataset(dataset_id)
        df = deduplicate_by_user(df, columns.user_col)
        control_label = resolve_control_label(df, columns.variant_col)
        return build_diagnostic_stratification(
            df=df,
            variant_col=columns.variant_col,
            control_label=control_label,
            metric_col=columns.metric_col,
            stratification_col=column,
        )
    except Exception:  # noqa: BLE001 — diagnostics are best-effort supporting
        # evidence on top of an already-invalid experiment; a bug here must
        # never crash report assembly. Falling back to a plain NOT_RUN
        # explanation (no diagnostic block) is always safe and honest.
        log.exception("[Decision] Failed to build diagnostic stratification for column=%s", column)
        return None


def _build_not_run_stratification(state: GraphState) -> StratificationResult | None:
    """
    When stratified analysis was explicitly requested but the experiment
    failed the SAME validity gate the ordinary hypothesis test is subject
    to (SRM / conflicting variant assignment / a critical quality
    failure — see route_after_validation), experiment_node never runs AT
    ALL, so state["stratification_result"] is simply absent — leaving
    `report.stratification` at None with zero explanation, indistinguishable
    from "stratified analysis was never requested." This reconstructs an
    honest fact instead, using ONLY already-computed validity facts
    (srm_result, has_conflicting_variant_duplicates,
    conflicting_variant_user_count, quality_checks) — it runs no new
    CAUSAL statistics and never overrides or re-evaluates the validity
    gate itself.

    SRM-FAILURE CASE (status=DIAGNOSTIC): when the block is specifically
    an SRM failure, causal stratified inference is BLOCKED — exactly
    like every other invalid-experiment case — but a non-causal,
    purely descriptive diagnostic breakdown (allocation by variant,
    stratum sizes/composition, descriptive rates, missingness,
    per-stratum SRM concentration) is still computed via
    `_try_build_diagnostic_stratification` and attached under
    `diagnostic`. This NEVER produces a causal treatment effect,
    p-value, confidence interval, or ship/no-ship recommendation —
    see `DiagnosticStratification`'s docstring. Every OTHER blocking
    reason (conflicting variant assignment, critical quality failure,
    experiment step not run) keeps the original NOT_RUN behavior
    unchanged, with no diagnostic block.

    Returns None whenever stratification wasn't requested at all
    (so every non-stratified report is completely unaffected), or
    whenever experiment_node actually DID run (its own
    stratification_result — status=RAN — is used instead; this
    function is a fallback for the "never ran" case only).
    """
    settings = state.get("settings")
    column = getattr(settings, "stratification_column", None) if settings is not None else None
    if settings is None or getattr(settings, "analysis_mode", None) != "stratified" or not column:
        return None

    srm_result = state.get("srm_result")
    if srm_result is not None and not srm_result.passed:
        reason = (
            f"Causal stratified inference is BLOCKED because the Sample Ratio Mismatch check "
            f"failed (p={srm_result.p_value:.4g}). No causal treatment effect, p-value, "
            "confidence interval, or ship/no-ship recommendation can be produced from this "
            "invalid experiment. A descriptive/diagnostic breakdown is available below to help "
            "investigate the source of the allocation problem."
        )
        diagnostic = _try_build_diagnostic_stratification(state, column)
        if diagnostic is not None:
            return StratificationResult(
                status=StratificationStatus.DIAGNOSTIC,
                stratification_column=column,
                eligibility=None,
                estimate=None,
                diagnostic=diagnostic,
                not_run_reason=reason,
            )
        return StratificationResult(
            status=StratificationStatus.NOT_RUN,
            stratification_column=column,
            eligibility=None,
            estimate=None,
            not_run_reason=reason,
        )
    elif state.get("has_conflicting_variant_duplicates"):
        count = state.get("conflicting_variant_user_count")
        count_text = f"{count:,}" if count is not None else "one or more"
        reason = (
            f"Experiment validity failed because {count_text} users have conflicting "
            "variant assignments. Statistical inference was not run on this invalid dataset."
        )
    else:
        critical_failures = [qc for qc in state.get("quality_checks", []) if not qc.passed and qc.critical]
        if critical_failures:
            labels = ", ".join(qc.label for qc in critical_failures)
            reason = (
                f"Experiment validity failed due to critical data-quality issue(s): {labels}. "
                "Statistical inference was not run on this invalid dataset."
            )
        else:
            # Validity passed but experiment_node still never ran (e.g. the
            # planner excluded "experiment" from this run) — a different,
            # honest reason from an actual validity failure.
            reason = (
                "The experiment step did not run for this request, so stratified analysis "
                "could not be executed."
            )

    return StratificationResult(
        status=StratificationStatus.NOT_RUN,
        stratification_column=column,
        eligibility=None,
        estimate=None,
        not_run_reason=reason,
    )


def decision_node(state: GraphState) -> GraphState:
    experiment_ran = "stat_results" in state
    validation_ran = "srm_result" in state
    test_selection = state.get("test_selection")
    srm_result = state.get("srm_result")
    kb_results = state.get("kb_results", [])
    funnel_result = state.get("funnel_result")
    hypothesis = state.get("hypothesis")
    stat_results = state.get("stat_results", [])

    # Phase 2 — deterministic expected-vs-observed comparison, computed
    # once here from already-computed facts (stat_results) and carried
    # through ReportFacts purely as a fact to expose. See
    # app/stats/hypothesis_evaluator.py for the full rule set; no LLM
    # is involved in this call.
    hypothesis_evaluation = evaluate_hypothesis(hypothesis, stat_results)

    # Phase 3 — deterministic Decision Support, computed once here from
    # already-computed facts (hypothesis, hypothesis_evaluation,
    # stat_results, guardrail_results, dataset). See
    # app/stats/decision_support.py for the full rule set; no LLM is
    # involved in this call, and neither evaluate_hypothesis() above
    # nor determine_decision() (inside the report generator, below)
    # are re-run or overridden by it.
    decision_support = build_decision_support(
        hypothesis=hypothesis,
        hypothesis_evaluation=hypothesis_evaluation,
        stat_results=stat_results,
        guardrail_results=state.get("guardrail_results", []),
        dataset=state["dataset"],
    )

    requested_guardrails, guardrail_resolutions, guardrail_request_state = _resolve_guardrail_facts(state)

    facts = ReportFacts(
        user_prompt=state["user_prompt"],
        dataset=state["dataset"],
        quality_checks=state.get("quality_checks", []),
        srm_passed=srm_result.passed if srm_result is not None else True,
        stat_results=stat_results,
        test_selections=[test_selection] if test_selection is not None else [],
        power_analysis=state.get("power_analysis"),
        mde_display=state.get("_mde_display", "N/A — no hypothesis test was run"),
        sample_size_note=state.get(
            "_sample_size_note",
            "N/A — no hypothesis test was run (see confidence reason)",
        ),
        variance_reduction=state.get("variance_reduction"),
        validation_ran=validation_ran,
        kb_results=kb_results,
        kb_error=state.get("kb_error"),
        bootstrap_ci_check=state.get("bootstrap_ci_check"),
        bootstrap_iterations=state.get("bootstrap_iterations"),
        funnel_result=funnel_result,
        funnel_by_group=state.get("funnel_by_group"),
        funnel_skip_reason=state.get("funnel_skip_reason"),
        has_conflicting_variant_duplicates=state.get("has_conflicting_variant_duplicates", False),
        guardrail_results=state.get("guardrail_results", []),
        requested_guardrails=requested_guardrails,
        guardrail_resolutions=guardrail_resolutions,
        guardrail_request_state=guardrail_request_state,
        hypothesis=hypothesis,
        hypothesis_evaluation=hypothesis_evaluation,
        segmentation_result=state.get("segmentation_result"),
        model=(state.get("settings").model if state.get("settings") is not None else None),
    )

    generator = get_report_generator()
    _report_start = time.monotonic()
    log.info("[Decision] starting (generator=%s)", type(generator).__name__)
    try:
        report = generator.generate(facts)
    except Exception as exc:  # noqa: BLE001 — a report-generation bug must never 500 the whole
        # /analyze request when the statistical analysis upstream already succeeded. This is
        # deliberately outside both TemplateReportGenerator and LLMReportGenerator (which already
        # have their own internal fallback for LLM-specific failures) — this is the final backstop
        # for a bug in report assembly itself, e.g. an attribute that doesn't exist on `facts`.
        report = build_emergency_fallback_report(facts, exc)
    log.info("[Decision] completed in %.2fs", time.monotonic() - _report_start)

    # Phase 7 — deterministic Decision Audit Trail, built once here from
    # the already-finalized `report` (covers both the normal generator
    # path and the emergency-fallback path above — both always populate
    # `decision`/`decisionReason`/etc.) plus `facts`. See
    # app/stats/decision_audit.py for the full rule set; no LLM is
    # involved, and `report.decision` itself is never recomputed here.
    decision_audit = build_decision_audit_trail(facts, report)

    # Product improvement — deterministic Decision Narrative (Why this
    # decision / What prevents a full GO / What to monitor / Recommended
    # next step). Built from exactly the same fields `report` already
    # carries post-generation (decision, decision_reason,
    # experiment_validity, guardrail_status, practical_significance) plus
    # state facts already used elsewhere in this node (stat_results,
    # guardrail_results, power_analysis, dataset.available_metrics) — see
    # app/stats/decision_narrative.py. No new statistics, no LLM call.
    decision_narrative = build_decision_narrative(
        decision=report.decision,
        decision_reason=report.decision_reason,
        experiment_validity=report.experiment_validity,
        guardrail_status=report.guardrail_status,
        practical_significance=report.practical_significance,
        stat_results=stat_results,
        guardrail_results=state.get("guardrail_results", []),
        available_metrics=state["dataset"].available_metrics,
        power_analysis=state.get("power_analysis"),
        guardrail_request_state=guardrail_request_state,
    )

    # Phase 2 — stamped on here, after generation, so every path
    # (TemplateReportGenerator's several branches, LLMReportGenerator,
    # AND the emergency fallback above) gets these two fields without
    # needing to touch every one of those construction sites
    # individually — see this module's docstring.
    report = report.model_copy(
        update={
            "hypothesis": hypothesis,
            "hypothesis_evaluation": hypothesis_evaluation,
            "decision_support": decision_support,
            "segmentation": state.get("segmentation_result"),
            "stratification": state.get("stratification_result") or _build_not_run_stratification(state),
            "decision_audit": decision_audit,
            "decision_narrative": decision_narrative,
            # Stamped here unconditionally (same pattern as hypothesis/
            # hypothesis_evaluation above) so EVERY report shape —
            # including the funnel-only/funnel-skipped/conceptual early
            # returns in TemplateReportGenerator, which never see these
            # facts otherwise — reflects the real guardrail request
            # state instead of silently defaulting to NOT_SPECIFIED.
            "guardrail_request_state": guardrail_request_state,
            "requested_guardrails": requested_guardrails,
            "guardrail_resolutions": guardrail_resolutions,
        }
    )

    log.info(
        "[Decision] Report generated — confidence=%s, srm_warning=%s, experiment_ran=%s, validation_ran=%s, kb_chunks=%d, funnel_computed=%s, hypothesis_verdict=%s",
        report.confidence.value,
        report.srm_warning,
        experiment_ran,
        validation_ran,
        len(kb_results),
        funnel_result is not None,
        hypothesis_evaluation.verdict.value if hypothesis_evaluation is not None and hypothesis_evaluation.verdict is not None else None,
    )

    return {**state, "report": report}
