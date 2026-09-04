"""
End-to-end test for the assignment-dataset flow, exercised through the
real FastAPI app exactly the way the frontend does it:

    POST /datasets/classify (primary file)
        -> POST /datasets/classify (assignment file)
        -> POST /experiments/analyze (assignmentDatasetId set)

This is the test that proves the full path — frontend upload pattern
-> API -> AnalyzeExperimentRequest -> GraphState -> classifier_node ->
classify_dataset(..., assignment_df=...) / detect_experiment_columns(...,
assignment_df=...) — is actually reachable, not just unit-tested in
isolation at the classifier level (see test_enrich_with_assignment.py).
"""

import io

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

_PRIMARY_CSV = b"user_id,order_value\n1,120\n2,95\n3,140\n"
_ASSIGNMENT_CSV = b"user_id,variant\n1,control\n2,treatment\n3,treatment\n"
_ASSIGNMENT_BUSINESS_FIELD_ONLY_CSV = b"user_id,plan\n1,pro\n2,pro\n3,free\n"
_ASSIGNMENT_DUPLICATE_CSV = b"user_id,variant\n1,control\n1,treatment\n2,treatment\n"


def _larger_scenario_csvs() -> tuple[bytes, bytes]:
    """
    Same shape as the reported 3-row scenario (user_id | order_value,
    user_id | variant), scaled up to enough users per arm for the real
    statistical pipeline (validation_node's Shapiro-Wilk normality
    check requires >=3 non-null observations per arm) — the literal
    3-row example is checked directly against classify_dataset() in
    `test_classify_alone_reports_two_variants_three_users` below, where
    exact counts matter more than statistical validity.
    """
    primary_lines = ["user_id,order_value"]
    assignment_lines = ["user_id,variant"]
    for uid in range(1, 41):
        variant = "control" if uid % 2 == 0 else "treatment"
        order_value = 100 + (uid % 7) * 5
        primary_lines.append(f"{uid},{order_value}")
        assignment_lines.append(f"{uid},{variant}")
    return (
        ("\n".join(primary_lines) + "\n").encode(),
        ("\n".join(assignment_lines) + "\n").encode(),
    )


def _upload(csv_bytes: bytes, filename: str) -> str:
    resp = client.post(
        "/datasets/classify",
        files={"file": (filename, io.BytesIO(csv_bytes), "text/csv")},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["datasetId"]


class TestAssignmentDatasetEndToEnd:
    def test_reported_scenario_produces_expected_columns_and_counts(self):
        """(11) Same shape as the bug report, run through the real API
        end to end (scaled up to a statistically valid sample size —
        see _larger_scenario_csvs docstring): 0 Variants /
        "missing a recognizable variant/group column" must NOT occur."""
        primary_csv, assignment_csv = _larger_scenario_csvs()
        primary_id = _upload(primary_csv, "primary.csv")
        assignment_id = _upload(assignment_csv, "assignment.csv")

        analyze_resp = client.post(
            "/experiments/analyze",
            json={
                "datasetId": primary_id,
                "assignmentDatasetId": assignment_id,
                "prompt": "Should we ship variant B?",
                "settings": {"cuped": False, "bootstrap": False},
            },
        )
        assert analyze_resp.status_code == 200, analyze_resp.text
        body = analyze_resp.json()

        # The report reaching this point at all means variant_col and
        # metric_col both resolved (detect_experiment_columns didn't
        # raise) — the exact previously-reported failure.
        assert body["report"]["stats"] != []
        run_metadata = body["report"].get("runMetadata") or {}
        # metric surfaces via the primary-metric label somewhere in the
        # report; assert the specific failure strings never appear
        # anywhere in the serialized response.
        import json as _json

        full_text = _json.dumps(body)
        assert "0 Variants" not in full_text
        assert "missing a recognizable variant/group column" not in full_text

    def test_classify_alone_reports_two_variants_three_users(self):
        """Directly checks the DatasetInfo the classify-only step would
        show — i.e. the frontend's Classifier Banner numbers — using the
        same classify_dataset() call path as the primary-only route.
        This mirrors the assignment merge indirectly: the analyze
        response above is the authoritative full-pipeline check; this
        confirms the reported dataset shape at the classify layer using
        the direct classifier import (fast, no second upload needed).
        """
        import pandas as pd

        from app.stats.dataset_classifier import classify_dataset

        primary_df = pd.read_csv(io.BytesIO(_PRIMARY_CSV))
        assignment_df = pd.read_csv(io.BytesIO(_ASSIGNMENT_CSV))
        info = classify_dataset(primary_df, assignment_df=assignment_df)
        assert info.variants == 2
        assert info.users == 3
        assert info.metric_label == "Order Value"

    def test_without_assignment_file_behavior_is_unchanged(self):
        """(9) Byte-for-byte equivalent behavior when no assignment
        dataset is supplied — the exact previously-reported failure mode
        must still occur (correctly) with only the primary file."""
        primary_id = _upload(_PRIMARY_CSV, "primary.csv")

        analyze_resp = client.post(
            "/experiments/analyze",
            json={
                "datasetId": primary_id,
                "prompt": "Should we ship variant B?",
                "settings": {"cuped": False, "bootstrap": False},
            },
        )
        assert analyze_resp.status_code == 422
        assert "missing a recognizable variant/group column" in analyze_resp.json()["detail"]

    def test_business_field_only_assignment_never_becomes_variant(self):
        """(12) Assignment file with only `user_id | plan` — `plan` must
        never be treated as the experiment variant."""
        primary_id = _upload(_PRIMARY_CSV, "primary.csv")
        assignment_id = _upload(_ASSIGNMENT_BUSINESS_FIELD_ONLY_CSV, "assignment_business_only.csv")

        analyze_resp = client.post(
            "/experiments/analyze",
            json={
                "datasetId": primary_id,
                "assignmentDatasetId": assignment_id,
                "prompt": "Should we ship variant B?",
                "settings": {"cuped": False, "bootstrap": False},
            },
        )
        # No recognizable variant column anywhere (primary has none,
        # assignment has only a business field) -> the same honest
        # "missing a recognizable variant/group column" failure as the
        # no-assignment-file case, NOT a silent "plan" promotion.
        assert analyze_resp.status_code == 422
        assert "variant/group column" in analyze_resp.json()["detail"]
        assert "plan" not in analyze_resp.json()["detail"].lower()

    def test_duplicate_assignment_rows_fail_deterministically(self):
        """(13) Assignment file with duplicate user assignments must
        fail with the deterministic assignment-validation error, not a
        silent pick-one-row resolution or an unhandled 500."""
        primary_id = _upload(_PRIMARY_CSV, "primary.csv")
        assignment_id = _upload(_ASSIGNMENT_DUPLICATE_CSV, "assignment_duplicate.csv")

        analyze_resp = client.post(
            "/experiments/analyze",
            json={
                "datasetId": primary_id,
                "assignmentDatasetId": assignment_id,
                "prompt": "Should we ship variant B?",
                "settings": {"cuped": False, "bootstrap": False},
            },
        )
        assert analyze_resp.status_code == 422
        assert "duplicate" in analyze_resp.json()["detail"].lower()

    def test_unknown_assignment_dataset_id_returns_404(self):
        """A malformed/expired assignmentDatasetId fails cleanly, same
        pattern as an unknown primary dataset_id, never a 500."""
        primary_id = _upload(_PRIMARY_CSV, "primary.csv")

        analyze_resp = client.post(
            "/experiments/analyze",
            json={
                "datasetId": primary_id,
                "assignmentDatasetId": "not-a-real-dataset-id",
                "prompt": "Should we ship variant B?",
                "settings": {"cuped": False, "bootstrap": False},
            },
        )
        assert analyze_resp.status_code == 404
