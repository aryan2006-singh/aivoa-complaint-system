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
async def test_list_and_get_and_patch_complaint() -> None:
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

            patched = await client.patch(f"/api/complaints/{complaint_id}", json={"status": "Under Review"})
            assert patched.status_code == 200
            assert patched.json()["status"] == "Under Review"

            history = await client.get(f"/api/complaints/{complaint_id}/history")
            assert history.status_code == 200
            assert history.json()[0]["field"] == "status"
    finally:
        # Defense in depth: this test writes rows to the shared live database,
        # so it must clean up after itself regardless of outcome -- other tests
        # run against the same database in the same session and must not observe
        # leftover rows.
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
