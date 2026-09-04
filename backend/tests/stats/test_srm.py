import pytest

from app.stats.srm import check_srm, srm_result_to_quality_check


def test_balanced_split_passes():
    result = check_srm(observed_control=6200, observed_variant=6200, expected_ratio=0.5)
    assert result.passed is True
    assert result.p_value > 0.05


def test_severe_imbalance_fails():
    # ~58/42 split on n=980, matching mock-data.ts LOW_CONFIDENCE_REPORT
    result = check_srm(observed_control=568, observed_variant=412, expected_ratio=0.5)
    assert result.passed is False
    assert result.p_value < 0.001


def test_slight_imbalance_within_noise_passes():
    # 50.1 / 49.9 on n=12400, matching mock-data.ts HIGH_CONFIDENCE_REPORT
    result = check_srm(observed_control=6212, observed_variant=6188, expected_ratio=0.5)
    assert result.passed is True


def test_zero_total_raises():
    with pytest.raises(ValueError):
        check_srm(observed_control=0, observed_variant=0)


def test_negative_counts_raise():
    with pytest.raises(ValueError):
        check_srm(observed_control=-5, observed_variant=100)


def test_invalid_expected_ratio_raises():
    with pytest.raises(ValueError):
        check_srm(observed_control=100, observed_variant=100, expected_ratio=1.5)


def test_quality_check_conversion_passed():
    result = check_srm(observed_control=6212, observed_variant=6188, expected_ratio=0.5)
    qc = srm_result_to_quality_check(result)
    assert qc.label == "Sample Ratio Mismatch (SRM)"
    assert qc.passed is True
    assert "FAILED" not in qc.detail


def test_quality_check_conversion_failed():
    result = check_srm(observed_control=568, observed_variant=412, expected_ratio=0.5)
    qc = srm_result_to_quality_check(result)
    assert qc.passed is False
    assert "FAILED" in qc.detail
    assert "p < 0.001" in qc.detail


def test_multi_arm_srm_uses_all_arms():
    from app.stats.srm import check_srm_multi_arm, srm_result_to_quality_check

    result = check_srm_multi_arm([33340, 33330, 33330, 33330, 33330, 33340])
    assert result.passed is True
    assert result.arm_counts == [33340, 33330, 33330, 33330, 33330, 33340]
    qc = srm_result_to_quality_check(result)
    assert "6-arm" in qc.detail
    assert "16.7%" in qc.detail


def test_six_arm_balanced_allocation_uses_one_sixth_expected_ratio():
    from app.stats.srm import check_srm_multi_arm, srm_result_to_quality_check

    result = check_srm_multi_arm([33340, 33330, 33330, 33330, 33330, 33340])
    assert result.passed is True
    assert result.expected_ratio == pytest.approx(1 / 6)
    qc = srm_result_to_quality_check(result)
    assert "6-arm" in qc.detail
    assert "16.7% per arm" in qc.detail
    assert "50/50" not in qc.detail


def test_six_arm_mismatch_is_not_collapsed_to_control_vs_pooled_treatment():
    from app.stats.srm import check_srm_multi_arm

    # One arm is badly under-allocated, while the other five are roughly equal.
    result = check_srm_multi_arm([1000, 39800, 39800, 39800, 39800, 39800])
    assert result.passed is False
    assert result.expected_ratio == pytest.approx(1 / 6)
    assert result.arm_counts == [1000, 39800, 39800, 39800, 39800, 39800]
