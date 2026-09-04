"""
Regression tests for the `RunMetadata.execution_status` rollup in
`app/api/routes_experiments.py::_build_run_metadata`.

Bug: `execution_status` is the WORST status across every
`ExecutionStep` in the trace (SUCCESS < SKIPPED < WARNING < FAILED).
Segmentation and Funnel Analysis are optional, exploratory capability
stages — the report itself says as much ("Segmentation is exploratory
and does not override the primary experiment decision") — and their
`SKIPPED` status (e.g. "no usable segmentation dimensions") is a
normal, healthy outcome, not an execution problem. Before this fix,
a fully successful run (LLM report generated, every other step
SUCCESS, `report.report_fallback_reason` empty) still surfaced as
top-level `Run Status: SKIPPED` in the UI whenever segmentation
happened to be skipped — read by a user as "the whole analysis was
skipped", which it was not.

These tests exercise `_build_run_metadata` directly (unit-level) so
the rollup logic is pinned precisely, independent of which real
dataset/prompt combination happens to produce a given step mix.
"""

from types import SimpleNamespace

from app.api.routes_experiments import _build_run_metadata
from app.core.config import app_settings
from app.schemas.dataset import DatasetInfo, DatasetType
from app.schemas.execution import ExecutionStep, ExecutionStepGroup, ExecutionStepStatus


def _step(id_: str, status: ExecutionStepStatus, label: str = "") -> ExecutionStep:
    return ExecutionStep(
        id=id_,
        label=label or id_,
        group=ExecutionStepGroup.CAPABILITY,
        detail="detail",
        status=status,
    )


def _final_state() -> dict:
    return {
        "dataset": DatasetInfo(
            type=DatasetType.RAW_USER_LEVEL,
            variants=2,
            users=12400,
            metric_label="Conversion Rate",
            metric_selection_reason="x",
        ),
        "plan": {"intent_label": "Full Experiment Review", "llm_status": "not_used"},
    }


def _request() -> "SimpleNamespace":
    # Only the attributes _build_run_metadata actually reads.
    return SimpleNamespace(
        dataset_name="demo_ab_checkout.csv",
        dataset_id="ds_1",
        settings=SimpleNamespace(model=None),
    )


# ---------------------------------------------------------------------------
# 1. The exact reported scenario: everything succeeded, only the optional
#    Segmentation stage was skipped for a legitimate reason -> run-level
#    status must be SUCCESS, not SKIPPED.
# ---------------------------------------------------------------------------


def test_successful_run_with_segmentation_skipped_is_success_not_skipped():
    steps = [
        _step("classifier", ExecutionStepStatus.SUCCESS),
        _step("planner", ExecutionStepStatus.SUCCESS),
        _step("knowledge_base", ExecutionStepStatus.SUCCESS),
        _step("validation", ExecutionStepStatus.SUCCESS),
        _step("experiment", ExecutionStepStatus.SUCCESS),
        _step("segmentation", ExecutionStepStatus.SKIPPED, "Segmentation — Skipped"),
        _step("decision", ExecutionStepStatus.SUCCESS),  # report generated, no fallback
    ]
    metadata = _build_run_metadata(_request(), _final_state(), steps)
    assert metadata.execution_status == ExecutionStepStatus.SUCCESS


def test_successful_run_with_funnel_skipped_is_success_not_skipped():
    """Same principle, same fix, for the other optional capability stage."""
    steps = [
        _step("classifier", ExecutionStepStatus.SUCCESS),
        _step("planner", ExecutionStepStatus.SUCCESS),
        _step("funnel", ExecutionStepStatus.SKIPPED, "Funnel Analysis — Skipped"),
        _step("decision", ExecutionStepStatus.SUCCESS),
    ]
    metadata = _build_run_metadata(_request(), _final_state(), steps)
    assert metadata.execution_status == ExecutionStepStatus.SUCCESS


def test_segmentation_and_funnel_both_skipped_still_success():
    steps = [
        _step("classifier", ExecutionStepStatus.SUCCESS),
        _step("planner", ExecutionStepStatus.SUCCESS),
        _step("segmentation", ExecutionStepStatus.SKIPPED),
        _step("funnel", ExecutionStepStatus.SKIPPED),
        _step("decision", ExecutionStepStatus.SUCCESS),
    ]
    metadata = _build_run_metadata(_request(), _final_state(), steps)
    assert metadata.execution_status == ExecutionStepStatus.SUCCESS


# ---------------------------------------------------------------------------
# 2. Previously-valid states must remain unchanged.
# ---------------------------------------------------------------------------


def test_all_success_is_still_success():
    steps = [
        _step("classifier", ExecutionStepStatus.SUCCESS),
        _step("planner", ExecutionStepStatus.SUCCESS),
        _step("validation", ExecutionStepStatus.SUCCESS),
        _step("experiment", ExecutionStepStatus.SUCCESS),
        _step("decision", ExecutionStepStatus.SUCCESS),
    ]
    metadata = _build_run_metadata(_request(), _final_state(), steps)
    assert metadata.execution_status == ExecutionStepStatus.SUCCESS


def test_llm_report_fallback_warning_still_wins_even_with_segmentation_skipped():
    """A real LLM/report fallback (WARNING) must still dominate the
    run-level status, exactly as before — the fix only touches how a
    benign, optional-capability SKIP is weighed, never WARNING/FAILED."""
    steps = [
        _step("classifier", ExecutionStepStatus.SUCCESS),
        _step("planner", ExecutionStepStatus.SUCCESS),
        _step("validation", ExecutionStepStatus.SUCCESS),
        _step("experiment", ExecutionStepStatus.SUCCESS),
        _step("segmentation", ExecutionStepStatus.SKIPPED),
        _step("decision", ExecutionStepStatus.WARNING, "Report Generation"),
    ]
    metadata = _build_run_metadata(_request(), _final_state(), steps)
    assert metadata.execution_status == ExecutionStepStatus.WARNING


def test_srm_failure_still_wins_as_failed():
    """A genuine data-quality FAILED step must still dominate, unchanged."""
    steps = [
        _step("classifier", ExecutionStepStatus.SUCCESS),
        _step("planner", ExecutionStepStatus.SUCCESS),
        _step("validation", ExecutionStepStatus.FAILED, "Data Quality Validation"),
        _step("decision", ExecutionStepStatus.WARNING),
    ]
    metadata = _build_run_metadata(_request(), _final_state(), steps)
    assert metadata.execution_status == ExecutionStepStatus.FAILED


def test_genuinely_skipped_non_capability_stage_still_elevates_status():
    """A SKIPPED step that is NOT segmentation/funnel (e.g. the
    hypothesis test itself never ran — the primary result is simply
    absent, not merely a supplementary one) must still surface as
    SKIPPED at the run level, exactly as before this fix — the
    exemption is narrowly scoped to segmentation/funnel only."""
    steps = [
        _step("classifier", ExecutionStepStatus.SUCCESS),
        _step("planner", ExecutionStepStatus.SUCCESS),
        _step("experiment", ExecutionStepStatus.SKIPPED, "Hypothesis Test — Skipped"),
        _step("decision", ExecutionStepStatus.SUCCESS),
    ]
    metadata = _build_run_metadata(_request(), _final_state(), steps)
    assert metadata.execution_status == ExecutionStepStatus.SKIPPED


def test_warning_still_wins_over_a_genuinely_skipped_stage():
    steps = [
        _step("classifier", ExecutionStepStatus.SUCCESS),
        _step("experiment", ExecutionStepStatus.SKIPPED, "Hypothesis Test — Skipped"),
        _step("decision", ExecutionStepStatus.WARNING),
    ]
    metadata = _build_run_metadata(_request(), _final_state(), steps)
    assert metadata.execution_status == ExecutionStepStatus.WARNING
