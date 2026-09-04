import numpy as np
import pandas as pd
import pytest

from app.schemas.statistics import MetricType
from app.stats.crm_funnel_summary import compute_crm_funnel_summary


def _make_df(n_per_arm=2000, seed=0):
    """
    3-arm dataset shaped like Hillstrom: visit -> conversion -> spend,
    each row is one customer (no event log). 'Mens' gets a real lift
    at every stage; 'Womens' gets a real visit lift that does NOT
    carry through to conversion — exercises the "didn't translate"
    narrative branch on purpose (unlike real Hillstrom, where every
    stage happened to be significant for both treatment arms).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for arm, visit_p, conv_p, spend_mean in [
        ("No E-Mail", 0.10, 0.05, 5.0),
        ("Mens E-Mail", 0.20, 0.15, 9.0),
        ("Womens E-Mail", 0.20, 0.051, 5.05),  # visit lift real; conversion/spend lift negligible
    ]:
        visit = rng.binomial(1, visit_p, n_per_arm)
        conversion = rng.binomial(1, conv_p, n_per_arm)
        spend = rng.normal(spend_mean, 2.0, n_per_arm)
        rows.append(pd.DataFrame({"segment": arm, "visit": visit, "conversion": conversion, "spend": spend}))
    return pd.concat(rows, ignore_index=True)


def test_crm_funnel_summary_rows_cover_every_metric_in_order():
    df = _make_df()
    summary = compute_crm_funnel_summary(
        df, "segment", "No E-Mail",
        funnel_metrics=[
            ("visit", MetricType.BINARY),
            ("conversion", MetricType.BINARY),
            ("spend", MetricType.CONTINUOUS_MONETARY),
        ],
    )
    assert [r.metric_column for r in summary.rows] == ["visit", "conversion", "spend"]
    for row in summary.rows:
        arms = {av.arm for av in row.arm_values}
        assert arms == {"No E-Mail", "Mens E-Mail", "Womens E-Mail"}
        assert {r.arm for r in row.treatment_results} == {"Mens E-Mail", "Womens E-Mail"}


def test_crm_funnel_summary_reuses_existing_stat_result_not_a_new_calculation():
    """Every StatResult in a row must come from the same compute_stat_result the two-arm engine already uses — same fields, same test-type enum, nothing new."""
    from app.stats.hypothesis_tests import compute_stat_result

    df = _make_df()
    summary = compute_crm_funnel_summary(
        df, "segment", "No E-Mail", funnel_metrics=[("visit", MetricType.BINARY)],
    )
    row = summary.rows[0]
    mens_result = next(r for r in row.treatment_results if r.arm == "Mens E-Mail")

    control = df.loc[df["segment"] == "No E-Mail", "visit"]
    variant = df.loc[df["segment"] == "Mens E-Mail", "visit"]
    expected = compute_stat_result(control, variant, MetricType.BINARY, "Visit")

    assert mens_result.test_type == expected.test_type
    assert mens_result.p_value == expected.p_value
    assert mens_result.delta == expected.delta


def test_crm_funnel_narrative_flags_lift_that_does_not_translate():
    """The exact scenario this module exists for: a stage-1 lift that a later stage does NOT carry through."""
    df = _make_df()
    summary = compute_crm_funnel_summary(
        df, "segment", "No E-Mail",
        funnel_metrics=[("visit", MetricType.BINARY), ("conversion", MetricType.BINARY)],
    )
    narrative_text = " ".join(summary.narrative)
    assert "Womens E-Mail" in narrative_text
    assert "did not translate" in narrative_text
    # Mens E-Mail's lift DOES carry through — must get the opposite phrasing, not the same one.
    assert any(
        line.startswith("Mens E-Mail") and "did not translate" not in line
        for line in summary.narrative
    )


def test_crm_funnel_narrative_empty_for_single_metric():
    """No consecutive pair exists with only one funnel stage -> no narrative, but rows still populate."""
    df = _make_df()
    summary = compute_crm_funnel_summary(df, "segment", "No E-Mail", funnel_metrics=[("visit", MetricType.BINARY)])
    assert summary.narrative == []
    assert len(summary.rows) == 1


def test_crm_funnel_summary_control_row_uses_same_display_convention_as_treatment_rows():
    """Control arm's own display value (computed locally) must match the SAME formatting compute_stat_result uses for its 'control' field — never a second, silently-diverging format."""
    df = _make_df()
    summary = compute_crm_funnel_summary(
        df, "segment", "No E-Mail", funnel_metrics=[("spend", MetricType.CONTINUOUS_MONETARY)],
    )
    row = summary.rows[0]
    control_display = next(av.display_value for av in row.arm_values if av.arm == "No E-Mail")
    assert control_display == row.treatment_results[0].control
    assert control_display.startswith("$")
