"""
Tests for evaluation/evaluators/agent_evaluator.py.

Synthetic `ExecutionStep` traces (constructed directly from the real
`ExecutionStep`/`ExecutionStepGroup` schema, not a full live graph
run) exercising all route shapes documented in
`app/graph/graph_builder.py`.
"""

from __future__ import annotations

from app.schemas.execution import ExecutionStep, ExecutionStepGroup
from evaluation.evaluators.agent_evaluator import (
    EXPECTED_TRAJECTORIES,
    evaluate_agent,
    evaluate_end_to_end,
    evaluate_trajectory,
)


def _step(step_id: str, label: str = "label", detail: str = "detail") -> ExecutionStep:
    group = {
        "classifier": ExecutionStepGroup.CLASSIFIER,
        "planner": ExecutionStepGroup.PLANNER,
        "decision": ExecutionStepGroup.DECISION_ENGINE,
    }.get(step_id, ExecutionStepGroup.CAPABILITY)
    return ExecutionStep(id=step_id, label=label, group=group, detail=detail)


def test_normal_review_trajectory_correct():
    steps = [_step("classifier"), _step("planner"), _step("validation"), _step("experiment"), _step("decision")]
    check = evaluate_trajectory("normal_review", steps)
    assert check.correct_sequence
    assert check.unnecessary_steps == []
    assert check.missing_steps == []
    assert check.component_schema_ok


def test_knowledge_base_only_trajectory_correct():
    steps = [_step("classifier"), _step("planner"), _step("knowledge_base"), _step("decision")]
    check = evaluate_trajectory("knowledge_base_only", steps)
    assert check.correct_sequence


def test_trajectory_with_unnecessary_step_detected():
    # experiment ran even though the scenario only needed knowledge_base.
    steps = [_step("classifier"), _step("planner"), _step("knowledge_base"), _step("experiment"), _step("decision")]
    check = evaluate_trajectory("knowledge_base_only", steps)
    assert not check.correct_sequence
    assert "experiment" in check.unnecessary_steps


def test_trajectory_with_missing_step_detected():
    steps = [_step("classifier"), _step("planner"), _step("decision")]  # skipped validation/experiment
    check = evaluate_trajectory("normal_review", steps)
    assert not check.correct_sequence
    assert "validation" in check.missing_steps
    assert "experiment" in check.missing_steps


def test_component_schema_violation_detected():
    steps = [_step("classifier"), _step("planner", label="", detail=""), _step("decision")]
    check = evaluate_trajectory("funnel_only" if False else "knowledge_base_only", steps)
    assert not check.component_schema_ok


def test_end_to_end_reaches_decision_terminal_node():
    steps = [_step("classifier"), _step("planner"), _step("validation"), _step("experiment"), _step("decision")]
    result = evaluate_end_to_end(steps)
    assert result.reached_terminal_node
    assert result.terminal_node_id == "decision"


def test_end_to_end_fails_if_trace_never_reaches_decision():
    steps = [_step("classifier"), _step("planner")]
    result = evaluate_end_to_end(steps)
    assert not result.reached_terminal_node


def test_evaluate_agent_over_all_route_shapes():
    scenarios = {
        "knowledge_base_only": [_step("classifier"), _step("planner"), _step("knowledge_base"), _step("decision")],
        "funnel_only": [_step("classifier"), _step("planner"), _step("funnel"), _step("decision")],
        "funnel_and_validation": [
            _step("classifier"), _step("planner"), _step("funnel"), _step("validation"), _step("experiment"), _step("decision"),
        ],
        "normal_review": [_step("classifier"), _step("planner"), _step("validation"), _step("experiment"), _step("decision")],
        "srm_fail_short_circuit": [
            _step("classifier"), _step("planner"), _step("validation"), _step("experiment"), _step("decision"),
        ],
    }
    assert set(scenarios.keys()) == set(EXPECTED_TRAJECTORIES.keys())
    report = evaluate_agent(scenarios)
    assert report.n_scenarios == 5
    assert report.trajectory_correct_rate == 1.0
    assert report.end_to_end_completion_rate == 1.0
    assert report.component_schema_ok_rate == 1.0
