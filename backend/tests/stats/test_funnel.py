import pandas as pd
import pytest

from app.stats.funnel import FunnelComputationError, compute_funnel, compute_funnel_by_group


def _simple_funnel_df() -> pd.DataFrame:
    """
    10 users visit; 7 of those signup; 4 of those trial; 1 of those
    purchases. Deliberately NOT strict-order-clean: includes a user
    who did Purchase WITHOUT Signup, to verify strict funnel semantics
    exclude them from the Purchase count.
    """
    rows = []
    for i in range(10):
        rows.append({"user_id": f"u{i}", "event": "Visit"})
    for i in range(7):
        rows.append({"user_id": f"u{i}", "event": "Signup"})
    for i in range(4):
        rows.append({"user_id": f"u{i}", "event": "Trial"})
    for i in range(1):
        rows.append({"user_id": f"u{i}", "event": "Purchase"})
    # A user who purchased WITHOUT going through Signup/Trial — must NOT count.
    rows.append({"user_id": "u_skip", "event": "Purchase"})
    return pd.DataFrame(rows)


class TestComputeFunnel:
    def test_step_counts_match_expected(self):
        df = _simple_funnel_df()
        result = compute_funnel(df, "user_id", "event", ["Visit", "Signup", "Trial", "Purchase"])

        counts = {s.name: s.users for s in result.steps}
        assert counts == {"Visit": 10, "Signup": 7, "Trial": 4, "Purchase": 1}

    def test_strict_funnel_excludes_users_who_skipped_a_step(self):
        """u_skip did Purchase without Signup/Trial — must not inflate the Purchase count."""
        df = _simple_funnel_df()
        result = compute_funnel(df, "user_id", "event", ["Visit", "Signup", "Trial", "Purchase"])
        purchase_step = next(s for s in result.steps if s.name == "Purchase")
        assert purchase_step.users == 1  # not 2

    def test_conversion_from_start_and_previous(self):
        df = _simple_funnel_df()
        result = compute_funnel(df, "user_id", "event", ["Visit", "Signup", "Trial", "Purchase"])
        by_name = {s.name: s for s in result.steps}

        assert by_name["Visit"].conversion_from_start == 1.0
        assert by_name["Signup"].conversion_from_start == pytest.approx(0.7)
        assert by_name["Signup"].conversion_from_previous == pytest.approx(0.7)
        assert by_name["Trial"].conversion_from_previous == pytest.approx(4 / 7)
        assert by_name["Purchase"].conversion_from_previous == pytest.approx(1 / 4)

    def test_overall_conversion(self):
        df = _simple_funnel_df()
        result = compute_funnel(df, "user_id", "event", ["Visit", "Signup", "Trial", "Purchase"])
        assert result.overall_conversion == pytest.approx(0.1)

    def test_largest_dropoff_identified_correctly(self):
        df = _simple_funnel_df()
        result = compute_funnel(df, "user_id", "event", ["Visit", "Signup", "Trial", "Purchase"])
        # Trial(4) -> Purchase(1) is a 75% drop, the largest of the three transitions
        # (Visit->Signup: 30% drop, Signup->Trial: ~43% drop, Trial->Purchase: 75% drop)
        assert result.largest_dropoff_from == "Trial"
        assert result.largest_dropoff_to == "Purchase"
        assert result.largest_dropoff_rate == pytest.approx(0.75)

    def test_fewer_than_two_steps_raises(self):
        df = _simple_funnel_df()
        with pytest.raises(FunnelComputationError):
            compute_funnel(df, "user_id", "event", ["Visit"])

    def test_no_users_at_first_step_raises(self):
        df = pd.DataFrame({"user_id": ["u1"], "event": ["Signup"]})
        with pytest.raises(FunnelComputationError):
            compute_funnel(df, "user_id", "event", ["Visit", "Signup"])

    def test_zero_users_reaching_a_later_step_does_not_crash(self):
        """Nobody reached Purchase at all — must return 0, not divide-by-zero crash."""
        df = pd.DataFrame({
            "user_id": ["u1", "u2"],
            "event": ["Visit", "Visit"],
        })
        result = compute_funnel(df, "user_id", "event", ["Visit", "Signup", "Purchase"])
        by_name = {s.name: s for s in result.steps}
        assert by_name["Signup"].users == 0
        assert by_name["Purchase"].users == 0
        assert by_name["Purchase"].conversion_from_previous == 0.0


class TestComputeFunnelByGroup:
    def test_splits_by_group_correctly(self):
        rows = []
        # control: 5 visit, 5 signup (100% conversion)
        for i in range(5):
            rows.append({"user_id": f"c{i}", "event": "Visit", "grp": "control"})
            rows.append({"user_id": f"c{i}", "event": "Signup", "grp": "control"})
        # variant: 5 visit, 1 signup (20% conversion)
        for i in range(5):
            rows.append({"user_id": f"v{i}", "event": "Visit", "grp": "variant"})
        rows.append({"user_id": "v0", "event": "Signup", "grp": "variant"})
        df = pd.DataFrame(rows)

        results = compute_funnel_by_group(df, "user_id", "event", ["Visit", "Signup"], "grp")

        assert set(results.keys()) == {"control", "variant"}
        assert results["control"].overall_conversion == pytest.approx(1.0)
        assert results["variant"].overall_conversion == pytest.approx(0.2)

    def test_empty_group_col_raises(self):
        df = _simple_funnel_df()
        with pytest.raises(FunnelComputationError):
            compute_funnel_by_group(df, "user_id", "event", ["Visit", "Signup"], "")

        with pytest.raises(FunnelComputationError):
            compute_funnel_by_group(df, "user_id", "event", ["Visit", "Signup"], None)
