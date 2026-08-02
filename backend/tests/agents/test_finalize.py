import uuid
from unittest.mock import patch

import pytest
from sqlalchemy import delete, select

from app.agents.nodes.finalize import finalize
from app.db.models import AIAssessment, Complaint
from app.db.session import SessionLocal


@pytest.mark.asyncio
async def test_finalize_does_not_crash_when_extracted_fields_missing() -> None:
    """Regression test: `finalize` used to do `state["extracted_fields"]` (a bare
    dict subscript). If an upstream node like `extract_fields` fails, `traced_node`
    swallows the exception and never writes the `extracted_fields` key at all
    (state is `total=False`) -- so `finalize`, which is deliberately NOT wrapped
    in `traced_node`, would raise an uncaught KeyError and crash the whole graph
    run with no `finalize` event and no `complaint_id` ever emitted. `finalize`
    must degrade gracefully instead, same as every other node.
    """
    async with SessionLocal() as db:
        await db.execute(delete(AIAssessment))
        await db.execute(delete(Complaint))
        await db.commit()

    try:
        state = {
            "raw_text": "Tablets cracked in blister, no fields could be extracted.",
            "trace": [],
            "errors": ["extract_fields: groq outage"],
            # note: no "extracted_fields" key at all, simulating extract_fields
            # having failed and traced_node swallowing the exception.
        }
        with patch("app.agents.nodes.finalize.get_stream_writer", return_value=lambda *_: None):
            result = await finalize(state)

        assert result["complaint_id"] is not None

        async with SessionLocal() as db:
            complaint = (
                await db.execute(select(Complaint).where(Complaint.id == uuid.UUID(result["complaint_id"])))
            ).scalar_one()
            assert complaint.product_name == "Unknown"
            assert complaint.batch_lot_number == "Unknown"
            assert complaint.description == state["raw_text"]

            assessment = (
                await db.execute(select(AIAssessment).where(AIAssessment.complaint_id == complaint.id))
            ).scalar_one()
            assert assessment.extracted_fields == {}
    finally:
        async with SessionLocal() as db:
            await db.execute(delete(AIAssessment))
            await db.execute(delete(Complaint))
            await db.commit()


@pytest.mark.asyncio
async def test_finalize_retries_on_complaint_number_collision() -> None:
    """`_next_complaint_number` derives the number from a plain COUNT(*), so two
    concurrent intakes can race and generate the same number, tripping the
    `complaint_number` unique constraint. `finalize` must retry with a freshly
    generated number rather than letting the IntegrityError crash the run.
    """
    async with SessionLocal() as db:
        await db.execute(delete(AIAssessment))
        await db.execute(delete(Complaint))
        existing = Complaint(
            id=uuid.uuid4(),
            complaint_number="CMPL-COLLIDE-000001",
            product_name="Existing Product",
            batch_lot_number="B00000",
            source="manual",
            description="Pre-existing complaint occupying the number finalize will try first.",
            status="New",
        )
        db.add(existing)
        await db.commit()

    try:
        call_count = {"n": 0}

        async def fake_next_complaint_number(_db):
            call_count["n"] += 1
            # First attempt collides with the pre-seeded row above; second attempt
            # is unique and should succeed.
            return "CMPL-COLLIDE-000001" if call_count["n"] == 1 else "CMPL-COLLIDE-000002"

        state = {
            "raw_text": "Tablets cracked in blister.",
            "extracted_fields": {
                "product_name": {"value": "Amoxicillin 500mg", "confidence": 0.9},
                "batch_lot_number": {"value": "B99999", "confidence": 0.9},
            },
            "trace": [],
        }
        with (
            patch("app.agents.nodes.finalize.get_stream_writer", return_value=lambda *_: None),
            patch("app.agents.nodes.finalize._next_complaint_number", new=fake_next_complaint_number),
        ):
            result = await finalize(state)

        assert call_count["n"] == 2
        assert result["complaint_id"] is not None

        async with SessionLocal() as db:
            complaint = (
                await db.execute(select(Complaint).where(Complaint.id == uuid.UUID(result["complaint_id"])))
            ).scalar_one()
            assert complaint.complaint_number == "CMPL-COLLIDE-000002"

            # The pre-seeded row must still be intact and untouched.
            preexisting = (
                await db.execute(select(Complaint).where(Complaint.complaint_number == "CMPL-COLLIDE-000001"))
            ).scalar_one()
            assert preexisting.product_name == "Existing Product"
    finally:
        async with SessionLocal() as db:
            await db.execute(delete(AIAssessment))
            await db.execute(delete(Complaint))
            await db.commit()
