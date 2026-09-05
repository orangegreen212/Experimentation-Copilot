import pytest
from pydantic import ValidationError

from app.schemas.experiment_definition import (
    ExperimentDefinitionCreateRequest,
    ExperimentMetric,
    HypothesisRole,
    MetricRole,
    RandomizationUnit,
    RoledHypothesis,
    Variant,
)
from app.schemas.hypothesis import ExpectedDirection, Hypothesis


def _hypothesis(statement="Redesign increases signup conversion.", metric="Signup Conversion"):
    return Hypothesis(
        statement=statement,
        primary_metric=metric,
        expected_direction=ExpectedDirection.INCREASE,
    )


def test_minimal_definition_defaults_to_draft():
    definition = ExperimentDefinitionCreateRequest(name="Landing Page Redesign")
    assert definition.status.value == "draft"
    assert definition.hypotheses == []
    assert definition.variants == []


def test_randomization_unit_defaults_to_user():
    definition = ExperimentDefinitionCreateRequest(name="Landing Page Redesign")
    assert definition.randomization_unit == RandomizationUnit.USER


def test_randomization_unit_can_be_set_explicitly():
    definition = ExperimentDefinitionCreateRequest(
        name="Landing Page Redesign", randomization_unit=RandomizationUnit.SESSION
    )
    assert definition.randomization_unit == RandomizationUnit.SESSION


def test_blank_name_rejected():
    with pytest.raises(ValidationError):
        ExperimentDefinitionCreateRequest(name="   ")


def test_requires_exactly_one_primary_hypothesis():
    with pytest.raises(ValidationError):
        ExperimentDefinitionCreateRequest(
            name="X",
            hypotheses=[
                RoledHypothesis(role=HypothesisRole.SECONDARY, hypothesis=_hypothesis()),
            ],
        )


def test_rejects_two_primary_hypotheses():
    with pytest.raises(ValidationError):
        ExperimentDefinitionCreateRequest(
            name="X",
            hypotheses=[
                RoledHypothesis(role=HypothesisRole.PRIMARY, hypothesis=_hypothesis()),
                RoledHypothesis(role=HypothesisRole.PRIMARY, hypothesis=_hypothesis("Other")),
            ],
        )


def test_accepts_one_primary_and_multiple_secondary_hypotheses():
    definition = ExperimentDefinitionCreateRequest(
        name="X",
        hypotheses=[
            RoledHypothesis(role=HypothesisRole.PRIMARY, hypothesis=_hypothesis()),
            RoledHypothesis(role=HypothesisRole.SECONDARY, hypothesis=_hypothesis("Activation")),
            RoledHypothesis(role=HypothesisRole.SECONDARY, hypothesis=_hypothesis("Revenue")),
        ],
    )
    assert len(definition.hypotheses) == 3


def test_variants_require_exactly_one_control():
    with pytest.raises(ValidationError):
        ExperimentDefinitionCreateRequest(
            name="X",
            variants=[
                Variant(name="A", is_control=False, allocation_pct=50),
                Variant(name="B", is_control=False, allocation_pct=50),
            ],
        )


def test_variants_require_allocation_sum_to_100():
    with pytest.raises(ValidationError):
        ExperimentDefinitionCreateRequest(
            name="X",
            variants=[
                Variant(name="Control", is_control=True, allocation_pct=50),
                Variant(name="Treatment", is_control=False, allocation_pct=40),
            ],
        )


def test_valid_variants_pass():
    definition = ExperimentDefinitionCreateRequest(
        name="X",
        variants=[
            Variant(name="Control", is_control=True, allocation_pct=50),
            Variant(name="Treatment A", is_control=False, allocation_pct=25),
            Variant(name="Treatment B", is_control=False, allocation_pct=25),
        ],
    )
    assert len(definition.variants) == 3


def test_at_most_one_primary_metric():
    with pytest.raises(ValidationError):
        ExperimentDefinitionCreateRequest(
            name="X",
            metrics=[
                ExperimentMetric(name="Signup Conversion", role=MetricRole.PRIMARY, type="binary"),
                ExperimentMetric(name="Purchase Conversion", role=MetricRole.PRIMARY, type="binary"),
            ],
        )
