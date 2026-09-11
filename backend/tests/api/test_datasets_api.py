"""
Tests for POST /datasets/classify, focused on the "REFRESH mode"
branch (`dataset_id` provided instead of a file/use_demo/dataset_key)
that backs `lib/api.ts`'s `refreshClassification()` — the Classifier
Banner recompute triggered when the analyst attaches or removes a
separate assignment file.

REGRESSION BEING GUARDED: before this fix, the route had no
`dataset_id` / `assignment_dataset_id` / `file_name` parameters at
all. FastAPI silently ignores unrecognized form fields rather than
rejecting the request, so every `refreshClassification()` call fell
through every `if`/`elif` branch (file is None, use_demo is False,
dataset_key is None) straight into the final `else` and always
returned 400 — the Classifier Banner never actually recomputed after
attaching/removing an assignment file, silently, with no error visible
anywhere except a rejected network request.
"""

import io

from fastapi.testclient import TestClient

from app.main import app
from app.core.dataset_store import _get_session_factory, DatasetModel

client = TestClient(app)

_PRIMARY_CSV = b"user_id,order_value\n1,120\n2,95\n3,140\n4,110\n"
_ASSIGNMENT_CSV = b"user_id,variant\n1,control\n2,treatment\n3,treatment\n4,control\n"


def _upload(csv_bytes: bytes, filename: str) -> str:
    resp = client.post("/datasets/classify", files={"file": (filename, io.BytesIO(csv_bytes), "text/csv")})
    assert resp.status_code == 200, resp.text
    return resp.json()["datasetId"]


def _dataset_row_count() -> int:
    session_factory = _get_session_factory()
    with session_factory() as session:
        return session.query(DatasetModel).count()


class TestRefreshClassificationContract:
    """Pins down exactly what lib/api.ts's refreshClassification()
    sends, so this test fails immediately if either side of the
    contract drifts again."""

    def test_refresh_with_only_dataset_id_matches_a_fresh_classify_of_the_same_data(self):
        dataset_id = _upload(_PRIMARY_CSV, "primary.csv")
        rows_after_upload = _dataset_row_count()

        resp = client.post("/datasets/classify", data={"dataset_id": dataset_id, "file_name": "primary.csv"})

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "dataset" in body
        assert body["dataset"]["users"] == 4
        # REFRESH must not write a new `datasets` row — it's a
        # read-only preview of the already-stored dataset.
        assert _dataset_row_count() == rows_after_upload

    def test_refresh_with_assignment_dataset_id_merges_like_a_real_analyze_would(self):
        """The primary CSV alone has no variant column — classify_dataset()
        can't find one. Attaching the assignment dataset in REFRESH mode
        must resolve variants=2, matching what classifier_node would
        resolve during a real /experiments/analyze call with the same
        two ids (see test_assignment_dataset_e2e.py)."""
        primary_id = _upload(_PRIMARY_CSV, "primary.csv")
        assignment_id = _upload(_ASSIGNMENT_CSV, "assignment.csv")
        rows_after_uploads = _dataset_row_count()

        resp = client.post(
            "/datasets/classify",
            data={"dataset_id": primary_id, "assignment_dataset_id": assignment_id, "file_name": "primary.csv"},
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["dataset"]["variants"] == 2
        # Still a read-only preview — merging for the banner must not
        # persist the merged frame (analyze() does that itself, once,
        # for real, when actually run).
        assert _dataset_row_count() == rows_after_uploads

    def test_refresh_with_unknown_dataset_id_is_a_clean_404_not_a_500(self):
        resp = client.post("/datasets/classify", data={"dataset_id": "does-not-exist"})
        assert resp.status_code == 404

    def test_no_recognized_field_still_produces_the_original_400(self):
        """The pre-existing behavior for a genuinely malformed request
        (no file, no use_demo, no dataset_key, no dataset_id) must be
        unchanged by adding the new branch."""
        resp = client.post("/datasets/classify", data={})
        assert resp.status_code == 400
