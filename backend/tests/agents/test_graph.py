from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import delete, select

from app.agents.graph import build_graph
from app.db.models import AIAssessment, Complaint
from app.db.session import SessionLocal
from tests.agents.fixtures import (
    FAKE_CAPA,
    FAKE_EXTRACT,
    FAKE_REGULATORY,
    FAKE_RISK,
    FAKE_ROOT_CAUSE,
)
from tests.agents.fixtures import fake_stream as _fake_stream


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


@pytest.mark.asyncio
async def test_full_graph_survives_extract_fields_failure() -> None:
    """Regression test for a real crash: if `extract_fields` fails, `traced_node`
    swallows the exception and never writes `extracted_fields` to state (state is
    `total=False`). Every downstream node except `summarize`/`finalize` is also
    wrapped in `traced_node`, so a `state["extracted_fields"]` KeyError there is
    likewise swallowed -- but `summarize` and `finalize` are deliberately NOT
    wrapped (so their own started/completed events can carry the final
    `complaint_id`), and previously did a bare `state["extracted_fields"]`
    subscript with no fallback. That raised an uncaught KeyError that tore down
    the whole graph run, with no `finalize` event and no `complaint_id` ever
    emitted -- exactly backwards from the brief's stated intent. This proves the
    full pipeline now degrades gracefully end-to-end instead of crashing.
    """
    async with SessionLocal() as db:
        await db.execute(delete(AIAssessment))
        await db.execute(delete(Complaint))
        await db.commit()

    try:
        with (
            patch(
                "app.agents.nodes.extract_fields.call_groq_json",
                new=AsyncMock(side_effect=RuntimeError("groq outage")),
            ),
            patch("app.agents.nodes.summarize.call_groq_stream", new=_fake_stream),
        ):
            graph = build_graph()
            events = []
            async for event in graph.astream(
                {"raw_text": "Tablets cracked in blister, extraction fails entirely.", "trace": [], "errors": []},
                stream_mode="custom",
            ):
                events.append(event)

        steps_seen = {e["step"] for e in events}
        assert "extract_fields" in steps_seen
        assert "finalize" in steps_seen

        finalize_events = [e for e in events if e["step"] == "finalize" and e["status"] == "completed"]
        assert len(finalize_events) == 1
        assert finalize_events[0]["data"]["complaint_id"] is not None

        async with SessionLocal() as db:
            complaint = (await db.execute(select(Complaint))).scalar_one()
            assert complaint.product_name == "Unknown"
            assert complaint.batch_lot_number == "Unknown"
            assert complaint.description == "Tablets cracked in blister, extraction fails entirely."
    finally:
        async with SessionLocal() as db:
            await db.execute(delete(AIAssessment))
            await db.execute(delete(Complaint))
            await db.commit()
