from unittest.mock import AsyncMock, patch

import pytest

from app.agents.nodes.extract_fields import extract_fields


@pytest.mark.asyncio
async def test_extract_fields_returns_structured_data() -> None:
    fake_response = {
        "product_name": {"value": "Amoxicillin 500mg", "confidence": 0.95},
        "batch_lot_number": {"value": "B12345", "confidence": 0.9},
        "customer_name": {"value": None, "confidence": 0.0},
        "customer_contact": {"value": None, "confidence": 0.0},
        "date_received": {"value": "2026-08-01", "confidence": 0.8},
        "description": {"value": "Tablet discoloration reported.", "confidence": 0.9},
        "category": {"value": "Quality", "confidence": 0.85},
        "source": {"value": "email", "confidence": 0.7},
    }
    with (
        patch("app.agents.nodes.extract_fields.call_groq_json", new=AsyncMock(return_value=fake_response)),
        patch("app.agents.tracing.get_stream_writer", return_value=lambda *_: None),
    ):
        result = await extract_fields({"raw_text": "Batch B12345 of Amoxicillin discolored."})
    assert result["extracted_fields"]["product_name"]["value"] == "Amoxicillin 500mg"
    assert result["trace"][0]["status"] == "completed"
