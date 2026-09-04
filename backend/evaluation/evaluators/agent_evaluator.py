"""
Agent / LangGraph evaluation — Layer 6 of the evaluation strategy.

Per task spec section 8: evaluate the agent at three levels, using
ONLY observable states/tool-calls/node-transitions/outputs — never
hidden chain-of-thought (this codebase doesn't expose any LLM
chain-of-thought to begin with; `ExecutionStep.detail` is
already-computed factual text, not a reasoning trace).

The observable trajectory this codebase exposes is
`list[ExecutionStep]` (see `app/schemas/execution.py` and
`app/api/routes_experiments.py::_build_execution_steps`), built from
the final `GraphState` after a run. Each `ExecutionStep.id` names
which capability node ran (`classifier`, `planner`, `funnel`,
`knowledge_base`, `validation`, `experiment`, `decision`) — this is
the "tool/node call" record for this system; there is no separate
tool-call log to parse.

Per `app/graph/graph_builder.py`'s own docstring, the graph has four
reachable route "shapes":
  1. knowledge_base-only  (conceptual question, no dataset work)
  2. funnel-only          (drop-off question, no validation/experiment)
  3. funnel + validation  (combined drop-off + ship/no-ship question)
  4. validation [+ experiment [+ knowledge_base]]  (the normal
     "should we ship?" review, with SRM failure short-circuiting
     before `experiment` conceptually — though `_build_execution_steps`
     still emits an `experiment` step in that case, labeled
     "Hypothesis Test — Skipped", so the trace stays fully observable
     rather than silently omitting the node)

This module checks a produced trace against the EXPECTED shape for
its scenario — not against the live graph (which would require a full
dataset/LLM run, out of scope for a fast, deterministic CI-safe eval
matching the project's existing `scripts/evaluate_*.py` pattern).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.execution import ExecutionStep

# Canonical expected node-id sequences per route shape (see module docstring).
EXPECTED_TRAJECTORIES: dict[str, list[str]] = {
    "knowledge_base_only": ["classifier", "planner", "knowledge_base", "decision"],
    "funnel_only": ["classifier", "planner", "funnel", "decision"],
    "funnel_and_validation": ["classifier", "planner", "funnel", "validation", "experiment", "decision"],
    "normal_review": ["classifier", "planner", "validation", "experiment", "decision"],
    "srm_fail_short_circuit": ["classifier", "planner", "validation", "experiment", "decision"],
}


@dataclass
class TrajectoryCheck:
    scenario: str
    expected_sequence: list[str]
    actual_sequence: list[str]
    correct_sequence: bool  # exact id-order match
    unnecessary_steps: list[str]  # ids present in actual but not expected
    missing_steps: list[str]  # ids present in expected but not actual
    # component-level: every step in the trace has non-empty label/detail
    # (a step with blank detail is a schema/output-correctness failure,
    # not a sequencing failure)
    component_schema_ok: bool


def evaluate_trajectory(scenario: str, actual_steps: list[ExecutionStep]) -> TrajectoryCheck:
    if scenario not in EXPECTED_TRAJECTORIES:
        raise KeyError(f"Unknown trajectory scenario: {scenario!r}")
    expected = EXPECTED_TRAJECTORIES[scenario]
    actual_ids = [s.id for s in actual_steps]

    unnecessary = [sid for sid in actual_ids if sid not in expected]
    missing = [sid for sid in expected if sid not in actual_ids]
    correct_sequence = actual_ids == expected

    component_ok = all(bool(s.label.strip()) and bool(s.detail.strip()) for s in actual_steps)

    return TrajectoryCheck(
        scenario=scenario,
        expected_sequence=expected,
        actual_sequence=actual_ids,
        correct_sequence=correct_sequence,
        unnecessary_steps=unnecessary,
        missing_steps=missing,
        component_schema_ok=component_ok,
    )


@dataclass
class EndToEndCheck:
    """Task completion: did the trace terminate at `decision` (the only
    valid terminal node — every route shape converges there) with a
    non-empty final report signal?"""

    reached_terminal_node: bool
    terminal_node_id: str | None


def evaluate_end_to_end(actual_steps: list[ExecutionStep]) -> EndToEndCheck:
    if not actual_steps:
        return EndToEndCheck(reached_terminal_node=False, terminal_node_id=None)
    last = actual_steps[-1]
    return EndToEndCheck(reached_terminal_node=last.id == "decision", terminal_node_id=last.id)


@dataclass
class AgentEvalReport:
    n_scenarios: int
    trajectory_correct_rate: float
    unnecessary_step_rate: float  # fraction of scenarios with >=1 unnecessary step
    end_to_end_completion_rate: float
    component_schema_ok_rate: float
    checks: list[TrajectoryCheck]


def evaluate_agent(scenarios: dict[str, list[ExecutionStep]]) -> AgentEvalReport:
    """`scenarios` maps a scenario name (a key in EXPECTED_TRAJECTORIES)
    to an observed `list[ExecutionStep]` trace for that scenario."""
    checks = [evaluate_trajectory(name, steps) for name, steps in scenarios.items()]
    n = len(checks)
    e2e_results = [evaluate_end_to_end(steps) for steps in scenarios.values()]

    return AgentEvalReport(
        n_scenarios=n,
        trajectory_correct_rate=sum(1 for c in checks if c.correct_sequence) / n if n else 0.0,
        unnecessary_step_rate=sum(1 for c in checks if c.unnecessary_steps) / n if n else 0.0,
        end_to_end_completion_rate=sum(1 for r in e2e_results if r.reached_terminal_node) / n if n else 0.0,
        component_schema_ok_rate=sum(1 for c in checks if c.component_schema_ok) / n if n else 0.0,
        checks=checks,
    )
