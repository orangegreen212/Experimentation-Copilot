import numpy as np
import pandas as pd
import pytest

from app.schemas.statistics import MetricType
from app.schemas.stratification import StratificationIneligibilityReason
from app.stats.stratification import (
    check_stratification_eligibility,
    run_stratified_analysis,
)


def _perfectly_associated_df(n_per_arm=200):
    """The exact bug-report scenario: landing_page is fully determined by group."""
    rng = np.random.default_rng(42)
    control_conv = rng.binomial(1, 0.10, n_per_arm)
    treatment_conv = rng.binomial(1, 0.12, n_per_arm)
    return pd.DataFrame({
        "user_id": list(range(2 * n_per_arm)),
        "group": ["control"] * n_per_arm + ["treatment"] * n_per_arm,
        "landing_page": ["old_page"] * n_per_arm + ["new_page"] * n_per_arm,
        "converted": list(control_conv) + list(treatment_conv),
    })


def _valid_stratifiable_df(n_per_stratum_arm=50):
    """A device_type variable that is NOT determined by group — both arms appear in every stratum."""
    rng = np.random.default_rng(7)
    rows = []
    uid = 0
    for device in ["mobile", "desktop"]:
        base_rate = 0.10 if device == "mobile" else 0.20
        for arm, lift in [("control", 0.0), ("treatment", 0.03)]:
            conv = rng.binomial(1, base_rate + lift, n_per_stratum_arm)
            for c in conv:
                rows.append({"user_id": uid, "group": arm, "device_type": device, "converted": c})
                uid += 1
    return pd.DataFrame(rows)


class TestEligibility:
    def test_perfectly_associated_variable_is_ineligible(self):
        df = _perfectly_associated_df()
        result = check_stratification_eligibility(df, "group", "control", "landing_page")
        assert result.eligible is False
        assert result.ineligibility_reason == StratificationIneligibilityReason.PERFECTLY_ASSOCIATED_WITH_ASSIGNMENT
        assert "perfectly associated with experiment assignment" in result.reason
        assert "landing_page" in result.reason

    def test_valid_variable_is_eligible(self):
        df = _valid_stratifiable_df()
        result = check_stratification_eligibility(df, "group", "control", "device_type")
        assert result.eligible is True
        assert len(result.strata) == 2
        assert all(s.has_both_variants for s in result.strata)
        assert all(s.sufficient for s in result.strata)

    def test_variable_is_treatment_assignment_itself_is_ineligible(self):
        df = _valid_stratifiable_df()
        result = check_stratification_eligibility(df, "group", "control", "group")
        assert result.eligible is False
        assert result.ineligibility_reason == StratificationIneligibilityReason.IS_TREATMENT_ASSIGNMENT

    def test_column_not_found(self):
        df = _valid_stratifiable_df()
        result = check_stratification_eligibility(df, "group", "control", "nonexistent_col")
        assert result.eligible is False
        assert result.ineligibility_reason == StratificationIneligibilityReason.COLUMN_NOT_FOUND

    def test_sparse_strata_excluded_but_reported(self):
        df = _valid_stratifiable_df(n_per_stratum_arm=50)
        # Add a third device value with only 2 observations per arm — too sparse.
        sparse_rows = pd.DataFrame({
            "user_id": [10000, 10001, 10002, 10003],
            "group": ["control", "control", "treatment", "treatment"],
            "device_type": ["tablet"] * 4,
            "converted": [0, 1, 0, 1],
        })
        df = pd.concat([df, sparse_rows], ignore_index=True)
        result = check_stratification_eligibility(df, "group", "control", "device_type", min_arm_size=10)
        assert result.eligible is True  # other 2 strata are still fine
        assert "tablet" in result.sparse_stratum_values
        tablet_summary = next(s for s in result.strata if s.stratum_value == "tablet")
        assert tablet_summary.has_both_variants is True
        assert tablet_summary.sufficient is False

    def test_missing_values_reported_explicitly(self):
        df = _valid_stratifiable_df()
        df.loc[0:9, "device_type"] = None
        result = check_stratification_eligibility(df, "group", "control", "device_type")
        assert result.missing_count == 10
        assert result.missing_proportion > 0
        assert "missing" in result.reason.lower()

    def test_all_strata_too_sparse_is_ineligible(self):
        df = pd.DataFrame({
            "user_id": range(8),
            "group": ["control", "treatment"] * 4,
            "device_type": ["mobile"] * 4 + ["desktop"] * 4,
            "converted": [0, 1, 0, 1, 1, 0, 1, 0],
        })
        result = check_stratification_eligibility(df, "group", "control", "device_type", min_arm_size=10)
        assert result.eligible is False
        assert result.ineligibility_reason == StratificationIneligibilityReason.ALL_STRATA_TOO_SPARSE


class TestStratifiedEstimator:
    def test_ineligible_variable_produces_no_estimate(self):
        df = _perfectly_associated_df()
        eligibility, estimate = run_stratified_analysis(
            df, "group", "control", "converted", MetricType.BINARY, "landing_page", "Conversion Rate"
        )
        assert eligibility.eligible is False
        assert estimate is None

    def test_valid_variable_produces_combined_estimate(self):
        df = _valid_stratifiable_df()
        eligibility, estimate = run_stratified_analysis(
            df, "group", "control", "converted", MetricType.BINARY, "device_type", "Conversion Rate"
        )
        assert eligibility.eligible is True
        assert estimate is not None
        assert estimate.strata_used == 2
        assert estimate.effect_estimate > 0  # treatment lift was positive in both strata
        assert estimate.standard_error > 0
        assert estimate.ci_lower < estimate.effect_estimate < estimate.ci_upper
        assert 0.0 <= estimate.p_value <= 1.0

    def test_estimate_is_not_a_naive_unweighted_average(self):
        """
        Build two strata with very different sizes/precision. A naive
        unweighted average of stratum effects would land exactly
        halfway between them; the inverse-variance-weighted combination
        must NOT do that — it should be pulled toward the larger,
        lower-variance stratum's effect.
        """
        rng = np.random.default_rng(3)
        rows = []
        uid = 0
        # Large stratum: big lift, huge n -> low variance, dominant weight.
        for arm, lift in [("control", 0.0), ("treatment", 0.20)]:
            conv = rng.binomial(1, 0.10 + lift, 2000)
            for c in conv:
                rows.append({"user_id": uid, "group": arm, "seg": "big", "converted": c})
                uid += 1
        # Small stratum: small lift, tiny n -> high variance, small weight.
        for arm, lift in [("control", 0.0), ("treatment", 0.01)]:
            conv = rng.binomial(1, 0.10 + lift, 15)
            for c in conv:
                rows.append({"user_id": uid, "group": arm, "seg": "small", "converted": c})
                uid += 1
        df = pd.DataFrame(rows)

        eligibility, estimate = run_stratified_analysis(
            df, "group", "control", "converted", MetricType.BINARY, "seg", "Conversion Rate", min_arm_size=10
        )
        assert eligibility.eligible is True
        assert estimate is not None
        naive_average = (0.20 + 0.01) / 2
        # The weighted estimate must be much closer to the big/precise
        # stratum's true lift (~0.20) than the naive midpoint (~0.105).
        assert abs(estimate.effect_estimate - 0.20) < abs(estimate.effect_estimate - naive_average)

    def test_continuous_metric_estimate(self):
        rng = np.random.default_rng(11)
        rows = []
        uid = 0
        for device in ["mobile", "desktop"]:
            for arm, lift in [("control", 0.0), ("treatment", 5.0)]:
                values = rng.normal(50 + lift, 10, 60)
                for v in values:
                    rows.append({"user_id": uid, "group": arm, "device_type": device, "order_value": v})
                    uid += 1
        df = pd.DataFrame(rows)
        eligibility, estimate = run_stratified_analysis(
            df, "group", "control", "order_value", MetricType.CONTINUOUS_MONETARY,
            "device_type", "Order Value", min_arm_size=10,
        )
        assert eligibility.eligible is True
        assert estimate is not None
        assert 2.0 < estimate.effect_estimate < 8.0  # true lift is 5.0
