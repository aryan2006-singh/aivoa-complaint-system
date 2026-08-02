from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete

from app.db.models import AIAssessment, Complaint
from app.db.session import SessionLocal
from app.main import app
from tests.agents.fixtures import FAKE_CAPA, FAKE_EXTRACT, FAKE_REGULATORY, FAKE_RISK, FAKE_ROOT_CAUSE, fake_stream


@pytest.mark.asyncio
async def test_intake_streams_sse_events_for_pasted_text() -> None:
    # This drives the real graph through `finalize`, which persists a Complaint
    # to the shared live database (see tests/agents/test_graph.py for the same
    # pattern). Clean up before and after so this test is robust to leftover
    # rows and doesn't leave any behind for other tests in the suite.
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
            patch("app.agents.nodes.summarize.call_groq_stream", new=fake_stream),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                async with client.stream(
                    "POST", "/api/complaints/intake", data={"text": "Tablets cracked in blister, batch B99999."}
                ) as response:
                    body = b""
                    async for chunk in response.aiter_bytes():
                        body += chunk
        assert response.status_code == 200
        assert b'"step": "extract_fields"' in body or b'"step":"extract_fields"' in body
        assert b"summarize" in body
    finally:
        async with SessionLocal() as db:
            await db.execute(delete(AIAssessment))
            await db.execute(delete(Complaint))
            await db.commit()


@pytest.mark.asyncio
async def test_intake_rejects_missing_text_and_file() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/complaints/intake", data={})
    assert response.status_code == 400
