"""
Phase 1 — Experiment Hypothesis schema tests.

Covers: a valid hypothesis, each required-field/invalid-value failure
mode, and that all optional fields can be omitted. See
app/schemas/hypothesis.py for the full contract these enforce.
"""

import math

import pytest
from pydantic import ValidationError

from app.schemas.hypothesis import ExpectedDirection, Hypothesis


def _valid_kwargs(**overrides):
    kwargs = dict(
        statement="Increasing the checkout CTA visibility will increase checkout conversion.",
        primary_metric="Conversion Rate",
        expected_direction=ExpectedDirection.INCREASE,
        expected_effect_relative=0.05,
        rationale="A similar CTA change on the homepage produced a comparable lift last quarter.",
    )
    kwargs.update(overrides)
    return kwargs


def test_valid_hypothesis():
    h = Hypothesis(**_valid_kwargs())
    assert h.statement.startswith("Increasing the checkout CTA")
    assert h.primary_metric == "Conversion Rate"
    assert h.expected_direction == ExpectedDirection.INCREASE
    assert h.expected_effect_relative == 0.05
    assert h.rationale is not None


def test_optional_fields_omitted():
    """expected_effect_relative and rationale are optional and default to None."""
    h = Hypothesis(
        statement="Reducing shipping cost display will increase conversion.",
        primary_metric="Conversion Rate",
        expected_direction=ExpectedDirection.INCREASE,
    )
    assert h.expected_effect_relative is None
    assert h.rationale is None


def test_missing_statement_is_rejected():
    with pytest.raises(ValidationError):
        Hypothesis(**{**_valid_kwargs(), "statement": None})


def test_empty_statement_is_rejected():
    with pytest.raises(ValidationError, match="statement"):
        Hypothesis(**{**_valid_kwargs(), "statement": "   "})


def test_statement_over_max_length_is_rejected():
    with pytest.raises(ValidationError, match="statement"):
        Hypothesis(**{**_valid_kwargs(), "statement": "x" * 501})


def test_missing_primary_metric_is_rejected():
    with pytest.raises(ValidationError):
        Hypothesis(**{**_valid_kwargs(), "primary_metric": None})


def test_empty_primary_metric_is_rejected():
    with pytest.raises(ValidationError, match="primary_metric"):
        Hypothesis(**{**_valid_kwargs(), "primary_metric": "  "})


def test_invalid_direction_is_rejected():
    with pytest.raises(ValidationError):
        Hypothesis(**{**_valid_kwargs(), "expected_direction": "up"})


def test_no_change_direction_is_accepted():
    h = Hypothesis(**{**_valid_kwargs(), "expected_direction": ExpectedDirection.NO_CHANGE, "expected_effect_relative": None})
    assert h.expected_direction == ExpectedDirection.NO_CHANGE


@pytest.mark.parametrize("bad_effect", [math.nan, math.inf, -math.inf])
def test_non_finite_expected_effect_is_rejected(bad_effect):
    with pytest.raises(ValidationError, match="finite"):
        Hypothesis(**{**_valid_kwargs(), "expected_effect_relative": bad_effect})


def test_non_numeric_expected_effect_is_rejected():
    with pytest.raises(ValidationError):
        Hypothesis(**{**_valid_kwargs(), "expected_effect_relative": "five percent"})


@pytest.mark.parametrize("direction", [ExpectedDirection.INCREASE, ExpectedDirection.DECREASE])
def test_negative_expected_effect_is_rejected_for_increase_or_decrease(direction):
    with pytest.raises(ValidationError, match="non-negative"):
        Hypothesis(**{**_valid_kwargs(), "expected_direction": direction, "expected_effect_relative": -0.05})


def test_positive_expected_effect_is_accepted_for_decrease():
    """direction carries the sign — a positive magnitude with 'decrease' means '5% relative decrease', not a rejection."""
    h = Hypothesis(**{**_valid_kwargs(), "expected_direction": ExpectedDirection.DECREASE, "expected_effect_relative": 0.05})
    assert h.expected_effect_relative == 0.05


def test_expected_effect_relative_is_not_percentage_points():
    """
    Documents the exact distinction the ticket required: 0.05 means
    '+5% relative' (10% baseline -> 10.5%), not '+5 percentage points'
    (10% baseline -> 15%). This is purely a contract/documentation
    test — the schema performs NO unit conversion at all, so the
    value must round-trip completely unchanged.
    """
    h = Hypothesis(**{**_valid_kwargs(), "expected_effect_relative": 0.05})
    baseline = 0.10
    expected_under_relative_interpretation = baseline * (1 + h.expected_effect_relative)
    expected_under_percentage_points_interpretation = baseline + h.expected_effect_relative
    assert expected_under_relative_interpretation == pytest.approx(0.105)
    assert expected_under_percentage_points_interpretation == pytest.approx(0.15)
    assert expected_under_relative_interpretation != expected_under_percentage_points_interpretation


def test_blank_rationale_is_normalized_to_none():
    h = Hypothesis(**{**_valid_kwargs(), "rationale": "   "})
    assert h.rationale is None


def test_rationale_over_max_length_is_rejected():
    with pytest.raises(ValidationError, match="rationale"):
        Hypothesis(**{**_valid_kwargs(), "rationale": "x" * 2001})


def test_camel_case_wire_format():
    """API boundary uses camelCase (CamelModel convention) — expectedEffectRelative, not expected_effect_relative."""
    h = Hypothesis(**_valid_kwargs())
    payload = h.model_dump(by_alias=True)
    assert "expectedEffectRelative" in payload
    assert "primaryMetric" in payload
    assert "expectedDirection" in payload
