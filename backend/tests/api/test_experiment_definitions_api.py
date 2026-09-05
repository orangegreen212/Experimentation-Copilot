"""
API-level tests for the new Phase 1 CRUD routes
(app/api/routes_experiment_definitions.py). Also asserts the existing
/experiments/analyze flow is completely unaffected by this addition —
this is the regression check the Stage 0 plan calls for before moving
to Phase 2.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_create_minimal_definition_defaults_to_draft():
    resp = client.post("/experiment-definitions", json={"name": "Landing Page Redesign"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "draft"
    assert body["id"]
    assert body["hypotheses"] == []
    assert body["variants"] == []


def test_create_rejects_invalid_variant_allocation():
    resp = client.post(
        "/experiment-definitions",
        json={
            "name": "Bad Allocation",
            "variants": [
                {"name": "Control", "isControl": True, "allocationPct": 50},
                {"name": "Treatment", "isControl": False, "allocationPct": 40},
            ],
        },
    )
    assert resp.status_code == 422


def test_full_crud_lifecycle():
    create_resp = client.post(
        "/experiment-definitions",
        json={
            "name": "Checkout Flow Experiment",
            "productArea": "Checkout",
            "owner": "jane@example.com",
            "hypotheses": [
                {
                    "role": "primary",
                    "hypothesis": {
                        "statement": "New CTA increases checkout conversion.",
                        "primaryMetric": "Checkout Conversion",
                        "expectedDirection": "increase",
                    },
                }
            ],
            "variants": [
                {"name": "Control", "isControl": True, "allocationPct": 50},
                {"name": "New CTA", "isControl": False, "allocationPct": 50},
            ],
            "metrics": [
                {"name": "Checkout Conversion", "role": "primary", "type": "binary"},
                {"name": "Refund Rate", "role": "guardrail", "type": "binary"},
            ],
        },
    )
    assert create_resp.status_code == 200
    definition_id = create_resp.json()["id"]

    # Appears in the Library list
    list_resp = client.get("/experiment-definitions")
    assert list_resp.status_code == 200
    ids = [row["id"] for row in list_resp.json()]
    assert definition_id in ids
    matching = next(row for row in list_resp.json() if row["id"] == definition_id)
    assert matching["primaryMetric"] == "Checkout Conversion"
    assert matching["status"] == "draft"

    # Full record round-trips correctly
    get_resp = client.get(f"/experiment-definitions/{definition_id}")
    assert get_resp.status_code == 200
    detail = get_resp.json()
    assert detail["name"] == "Checkout Flow Experiment"
    assert len(detail["variants"]) == 2
    assert detail["hypotheses"][0]["role"] == "primary"

    # Partial update — only status changes, everything else untouched
    patch_resp = client.patch(f"/experiment-definitions/{definition_id}", json={"status": "ready"})
    assert patch_resp.status_code == 200
    patched = patch_resp.json()
    assert patched["status"] == "ready"
    assert patched["name"] == "Checkout Flow Experiment"
    assert len(patched["variants"]) == 2

    # Delete
    delete_resp = client.delete(f"/experiment-definitions/{definition_id}")
    assert delete_resp.status_code == 200
    assert delete_resp.json() == {"deleted": True}

    assert client.get(f"/experiment-definitions/{definition_id}").status_code == 404


def test_get_unknown_definition_returns_404():
    resp = client.get("/experiment-definitions/does-not-exist")
    assert resp.status_code == 404


def test_patch_unknown_definition_returns_404():
    resp = client.patch("/experiment-definitions/does-not-exist", json={"status": "ready"})
    assert resp.status_code == 404


def test_delete_unknown_definition_returns_404():
    resp = client.delete("/experiment-definitions/does-not-exist")
    assert resp.status_code == 404


def test_existing_analyze_flow_unaffected_by_new_router():
    """
    Regression guard: adding the experiment-definitions router and the
    nullable definition_id column must not change the existing
    classify -> analyze -> chat flow's behavior or response shape.
    """
    classify_resp = client.post("/datasets/classify", data={"use_demo": "true"})
    assert classify_resp.status_code == 200
    dataset_id = classify_resp.json()["datasetId"]

    analyze_resp = client.post(
        "/experiments/analyze",
        json={
            "datasetId": dataset_id,
            "prompt": "Should we ship variant B?",
            "settings": {"cuped": False, "bootstrap": False, "model": "claude-sonnet", "costUsd": 0},
        },
    )
    assert analyze_resp.status_code == 200
    body = analyze_resp.json()
    assert body["experimentId"]
    assert body["report"]["stats"] != []
