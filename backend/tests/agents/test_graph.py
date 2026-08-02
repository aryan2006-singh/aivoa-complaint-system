from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import delete, select

from app.agents.graph import build_graph
from app.db.models import AIAssessment, Complaint
from app.db.session import SessionLocal

FAKE_EXTRACT = {
    "product_name": {"value": "Amoxicillin 500mg", "confidence": 0.9},
    "batch_lot_number": {"value": "B99999", "confidence": 0.9},
    "customer_name": {"value": "Jane Doe", "confidence": 0.8},
    "customer_contact": {"value": "jane@example.com", "confidence": 0.8},
    "date_received": {"value": "2026-08-02", "confidence": 0.8},
    "description": {"value": "Tablets are cracked in several blister cells.", "confidence": 0.9},
    "category": {"value": "Quality", "confidence": 0.8},
    "source": {"value": "email", "confidence": 0.7},
}
FAKE_RISK = {"classification": "Major", "rationale": "Cracked tablets may affect dosing."}
FAKE_REGULATORY = {"reportable": False, "rationale": "No patient harm reported."}
FAKE_ROOT_CAUSE = {"category": "Machine", "explanation": "Likely tableting press fault."}
FAKE_CAPA = {"corrective": "Inspect press tooling.", "preventive": "Add in-process cracked-tablet detection."}


async def _fake_stream(*_args, **_kwargs):
    for token in ["Summary ", "of ", "the ", "complaint."]:
        yield token


@pytest.mark.asyncio
async def test_full_graph_normal_path_persists_complaint() -> None:
    async with SessionLocal() as db:
        await db.execute(delete(AIAssessment))
        await db.execute(delete(Complaint))
        await db.commit()

    try:
        with (
            patch("app.agents.nodes.extract_fields.call_groq_json", new=AsyncMock(return_value=FAKE_EXTRACT)),
            patch("app.agents.nodes.classify_risk.call_groq_json", new=AsyncMock(return_value=FAKE_RISK)),
            patch(
                "app.agents.nodes.regulatory_reportability.call_groq_json",
                new=AsyncMock(return_value=FAKE_REGULATORY),
            ),
            patch("app.agents.nodes.suggest_root_cause.call_groq_json", new=AsyncMock(return_value=FAKE_ROOT_CAUSE)),
            patch("app.agents.nodes.recommend_capa.call_groq_json", new=AsyncMock(return_value=FAKE_CAPA)),
            patch("app.agents.nodes.summarize.call_groq_stream", new=_fake_stream),
        ):
            graph = build_graph()
            events = []
            async for event in graph.astream(
                {"raw_text": "Tablets cracked in blister.", "trace": [], "errors": []},
                stream_mode="custom",
            ):
                events.append(event)

        steps_seen = {e["step"] for e in events}
        assert "extract_fields" in steps_seen
        assert "summarize" in steps_seen
        assert any(e["step"] == "summarize" and e["status"] == "token" for e in events)

        async with SessionLocal() as db:
            complaint = (await db.execute(select(Complaint))).scalar_one()
            assert complaint.batch_lot_number == "B99999"
            assert complaint.product_name == "Amoxicillin 500mg"
            assert complaint.severity == "Major"

            assessment = (await db.execute(select(AIAssessment))).scalar_one()
            assert assessment.complaint_id == complaint.id
            assert assessment.risk_classification == "Major"
            assert assessment.regulatory_reportable is False
            assert assessment.root_cause_suggestion == "Likely tableting press fault."
            assert assessment.capa_recommendation == "Inspect press tooling."
            assert assessment.summary == "Summary of the complaint."
    finally:
        # Defense in depth: this test writes rows to the shared live database,
        # so it must clean up after itself regardless of outcome -- other tests
        # (e.g. test_startup.py) run against the same database in the same
        # session and must not observe leftover rows.
        async with SessionLocal() as db:
            await db.execute(delete(AIAssessment))
            await db.execute(delete(Complaint))
            await db.commit()


@pytest.mark.asyncio
async def test_full_graph_short_circuits_on_duplicate() -> None:
    # `check_duplicates` is wrapped by `traced_node`, whose `wrapper` closes over
    # the original function object directly rather than indirecting through
    # `wrapper.__wrapped__` -- so patching `check_duplicates.__wrapped__` (as
    # originally proposed) has no effect on what actually runs. Patching the name
    # bound in `app.agents.graph` (where `build_graph` looks it up when wiring
    # `add_node`) does work, as long as the patch is active while `build_graph()`
    # runs.
    fake_result = {
        "duplicate": {"complaint_id": "fake-id", "confidence": 0.97},
        "is_duplicate": True,
        "embedding": [0.0] * 384,
        "trace": [{"step": "check_duplicates", "status": "completed"}],
    }

    async def _fake_check_duplicates(_state):
        return fake_result

    with (
        patch("app.agents.nodes.extract_fields.call_groq_json", new=AsyncMock(return_value=FAKE_EXTRACT)),
        patch("app.agents.graph.check_duplicates", new=_fake_check_duplicates),
    ):
        graph = build_graph()
        events = []
        async for event in graph.astream(
            {"raw_text": "Tablets cracked in blister.", "trace": [], "errors": []},
            stream_mode="custom",
        ):
            events.append(event)

    steps_seen = {e["step"] for e in events}
    assert "finalize_duplicate" in steps_seen
    assert "classify_risk" not in steps_seen
    assert "suggest_root_cause" not in steps_seen
    assert "summarize" not in steps_seen

    finalize_events = [e for e in events if e["step"] == "finalize_duplicate"]
    assert finalize_events[-1]["data"]["duplicate_of"] == "fake-id"
