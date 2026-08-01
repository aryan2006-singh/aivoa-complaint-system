import uuid
from unittest.mock import patch

import pytest
from sqlalchemy import delete

from app.agents.nodes.check_duplicates import check_duplicates
from app.db.models import AIAssessment, Complaint
from app.db.session import SessionLocal
from app.services.embeddings import get_embedding


@pytest.mark.asyncio
async def test_detects_near_duplicate_by_semantic_similarity() -> None:
    async with SessionLocal() as db:
        await db.execute(delete(AIAssessment))
        await db.execute(delete(Complaint))
        existing = Complaint(
            id=uuid.uuid4(),
            complaint_number="CMPL-TEST-0001",
            product_name="Amoxicillin 500mg",
            batch_lot_number="B12345",
            source="manual",
            description="Tablets show yellow discoloration in the blister pack.",
            status="New",
        )
        db.add(existing)
        await db.flush()
        db.add(
            AIAssessment(
                complaint_id=existing.id,
                extracted_fields={},
                model_used="test",
                embedding=get_embedding("Tablets show yellow discoloration in the blister pack."),
            )
        )
        await db.commit()

    try:
        state = {
            "raw_text": "Noticed yellow discoloration on the tablets inside the blister.",
            "extracted_fields": {
                "product_name": {"value": "Amoxicillin 500mg", "confidence": 0.9},
                "batch_lot_number": {"value": "B12345", "confidence": 0.9},
                "description": {"value": "Noticed yellow discoloration on the tablets inside the blister.", "confidence": 0.9},
            },
        }
        with patch("app.agents.tracing.get_stream_writer", return_value=lambda *_: None):
            result = await check_duplicates(state)
        assert result["is_duplicate"] is True
        assert result["duplicate"]["confidence"] > 0.85
    finally:
        # Defense in depth: this test writes rows to the shared live database,
        # so it must clean up after itself regardless of outcome — other tests
        # (e.g. test_startup.py) run against the same database in the same
        # session and must not observe leftover rows.
        async with SessionLocal() as db:
            await db.execute(delete(AIAssessment))
            await db.execute(delete(Complaint))
            await db.commit()
