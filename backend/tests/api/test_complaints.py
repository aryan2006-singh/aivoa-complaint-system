import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete

from app.db.models import AIAssessment, Complaint, ComplaintStatusHistory
from app.db.session import SessionLocal
from app.main import app


async def _seed_complaint() -> uuid.UUID:
    async with SessionLocal() as db:
        await db.execute(delete(ComplaintStatusHistory))
        await db.execute(delete(AIAssessment))
        await db.execute(delete(Complaint))
        complaint = Complaint(
            id=uuid.uuid4(),
            complaint_number="CMPL-TEST-0002",
            product_name="Ibuprofen 200mg",
            batch_lot_number="B55555",
            source="manual",
            description="Bottle seal broken on arrival.",
            status="New",
        )
        db.add(complaint)
        await db.commit()
        return complaint.id


@pytest.mark.asyncio
async def test_patch_creates_history_with_correct_values() -> None:
    """Verify that PATCH creates ComplaintStatusHistory rows with correct old_value,
    new_value, and changed_by. This is the core correctness risk of the PATCH endpoint."""
    complaint_id = await _seed_complaint()
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Change status from "New" to "Under Review"
            patched = await client.patch(
                f"/api/complaints/{complaint_id}",
                json={"status": "Under Review"}
            )
            assert patched.status_code == 200
            assert patched.json()["status"] == "Under Review"

            # Verify the history row has correct values
            history = await client.get(f"/api/complaints/{complaint_id}/history")
            assert history.status_code == 200
            rows = history.json()
            assert len(rows) == 1
            assert rows[0]["field"] == "status"
            assert rows[0]["old_value"] == "New"
            assert rows[0]["new_value"] == "Under Review"
            assert rows[0]["changed_by"] == "reviewer"  # Default from ComplaintUpdate schema
    finally:
        async with SessionLocal() as db:
            await db.execute(delete(ComplaintStatusHistory))
            await db.execute(delete(AIAssessment))
            await db.execute(delete(Complaint))
            await db.commit()


@pytest.mark.asyncio
async def test_patch_no_op_update_creates_no_history_row() -> None:
    """Verify that setting a field to its current value creates no history row.
    This tests the "one row per changed field" semantics."""
    complaint_id = await _seed_complaint()
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # First, verify history is empty
            history = await client.get(f"/api/complaints/{complaint_id}/history")
            assert history.status_code == 200
            assert len(history.json()) == 0

            # PATCH with status set to its current value "New"
            patched = await client.patch(
                f"/api/complaints/{complaint_id}",
                json={"status": "New"}  # Same as current value
            )
            assert patched.status_code == 200
            assert patched.json()["status"] == "New"

            # Verify no history row was created for the no-op update
            history = await client.get(f"/api/complaints/{complaint_id}/history")
            assert history.status_code == 200
            assert len(history.json()) == 0  # Still empty
    finally:
        async with SessionLocal() as db:
            await db.execute(delete(ComplaintStatusHistory))
            await db.execute(delete(AIAssessment))
            await db.execute(delete(Complaint))
            await db.commit()


@pytest.mark.asyncio
async def test_patch_multiple_fields_creates_one_row_per_changed_field() -> None:
    """Verify that changing multiple fields in one PATCH creates exactly one
    history row per field that actually changed."""
    complaint_id = await _seed_complaint()
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # PATCH with two fields that will change (status, category)
            patched = await client.patch(
                f"/api/complaints/{complaint_id}",
                json={
                    "status": "Under Review",
                    "category": "Contamination"
                }
            )
            assert patched.status_code == 200
            assert patched.json()["status"] == "Under Review"
            assert patched.json()["category"] == "Contamination"

            # Verify exactly 2 history rows exist, one per changed field
            history = await client.get(f"/api/complaints/{complaint_id}/history")
            assert history.status_code == 200
            rows = history.json()
            assert len(rows) == 2

            # Check both rows exist (order is desc by changed_at, so newest first)
            fields_changed = {row["field"] for row in rows}
            assert fields_changed == {"status", "category"}

            # Verify each row has correct values
            status_row = next(r for r in rows if r["field"] == "status")
            assert status_row["old_value"] == "New"
            assert status_row["new_value"] == "Under Review"
            assert status_row["changed_by"] == "reviewer"

            category_row = next(r for r in rows if r["field"] == "category")
            assert category_row["old_value"] is None  # Started as None
            assert category_row["new_value"] == "Contamination"
            assert category_row["changed_by"] == "reviewer"
    finally:
        async with SessionLocal() as db:
            await db.execute(delete(ComplaintStatusHistory))
            await db.execute(delete(AIAssessment))
            await db.execute(delete(Complaint))
            await db.commit()


@pytest.mark.asyncio
async def test_list_and_get_complaint() -> None:
    complaint_id = await _seed_complaint()
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            listed = await client.get("/api/complaints")
            assert listed.status_code == 200
            assert len(listed.json()) == 1

            detail = await client.get(f"/api/complaints/{complaint_id}")
            assert detail.status_code == 200
            assert detail.json()["product_name"] == "Ibuprofen 200mg"
    finally:
        async with SessionLocal() as db:
            await db.execute(delete(ComplaintStatusHistory))
            await db.execute(delete(AIAssessment))
            await db.execute(delete(Complaint))
            await db.commit()


@pytest.mark.asyncio
async def test_get_missing_complaint_returns_404() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/api/complaints/{uuid.uuid4()}")
    assert response.status_code == 404
