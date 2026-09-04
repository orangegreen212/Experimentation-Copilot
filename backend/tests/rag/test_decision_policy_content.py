"""
Decision-policy content tests (task: "Improve the Experimentation RAG
Knowledge Base", item 7).

IMPORTANT: these tests check the RETRIEVED KNOWLEDGE BASE TEXT, not the
app's actual ship/no-ship decision — that decision is computed entirely
deterministically in app/graph/report_generator.py's `determine_decision`
and never touches the LLM or the RAG (see that module and
app/rag/blocking_topics.py). What's under test here is narrower and
just as important: that the methodology/decision-policy text this
system would show as supporting EVIDENCE says the right thing, and
does not contain language that could mislead a reader into the wrong
conclusion (e.g. treating "underpowered" as "no effect").
"""

from app.rag.retriever import KnowledgeBaseRetriever, load_documents

_RETRIEVER = KnowledgeBaseRetriever(load_documents())


def _top_chunk_text(query: str, top_k: int = 3) -> str:
    results = _RETRIEVER.retrieve(query, top_k=top_k)
    return " ".join(f"{r.chunk.heading} {r.chunk.content}" for r in results).lower()


def test_underpowered_policy_supports_insufficient_evidence():
    text = _top_chunk_text("underpowered experiment insufficient evidence")
    assert "insufficient evidence" in text or "absence of evidence" in text


def test_underpowered_policy_does_not_claim_no_effect():
    """
    The retrieved underpowered/insufficient-evidence policy chunk must
    not itself assert "the treatment has no effect" — it should say
    the opposite: that a null result under low power does NOT establish
    that.
    """
    results = _RETRIEVER.retrieve("underpowered experiment insufficient evidence", top_k=3)
    policy_chunk = next(
        (r for r in results if r.chunk.metadata.get("concept") == "underpowered"), None
    )
    assert policy_chunk is not None
    content = policy_chunk.chunk.content.lower()
    # The doc explicitly warns against this exact conflation — it must
    # contain the corrective framing, not assert the conflated claim as fact.
    assert "not the same claim" in content or "not proof" in content or "not evidence of absence" in content


def test_srm_failure_policy_supports_invalid_stop_analysis():
    text = _top_chunk_text("Sample Ratio Mismatch SRM failure what to do")
    assert "invalid" in text or "discard" in text or "stop the analysis" in text


def test_significant_and_practically_meaningful_supports_ship():
    text = _top_chunk_text("statistically significant effect practically significant guardrails fine ship")
    assert "ship" in text


def test_significant_but_practically_insignificant_does_not_auto_ship():
    """
    A significant-but-practically-trivial result should retrieve policy
    text that gates SHIP on practical significance, not text that
    endorses shipping on significance alone.
    """
    results = _RETRIEVER.retrieve(
        "statistically significant but effect size below practical significance threshold", top_k=3
    )
    assert results
    combined = " ".join(r.chunk.content.lower() for r in results)
    assert "practical significance" in combined
    # Should not be dominated by unconditional "SHIP" framing without the gate.
    assert "clears practical significance" in combined or "does not clear practical significance" in combined \
        or "not practically significant" in combined or "no-go" in combined


def test_guardrail_regression_supports_caution():
    text = _top_chunk_text("primary metric improves but guardrail regresses severely")
    assert "caution" in text or "guardrail" in text


def test_ship_policy_requires_validity_and_direction_and_guardrails():
    results = _RETRIEVER.retrieve("SHIP policy requirements", top_k=3)
    ship_chunk = next((r for r in results if r.chunk.metadata.get("concept") == "ship"), None)
    assert ship_chunk is not None
    content = ship_chunk.chunk.content.lower()
    assert "valid" in content
    assert "guardrail" in content
    assert "practical significance" in content
