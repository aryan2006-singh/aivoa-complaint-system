from unittest.mock import patch

import pytest

from app.agents.nodes.check_completeness import check_completeness


@pytest.mark.asyncio
async def test_flags_missing_batch_number() -> None:
    state = {
        "extracted_fields": {
            "product_name": {"value": "Amoxicillin", "confidence": 0.9},
            "batch_lot_number": {"value": None, "confidence": 0.0},
            "description": {"value": "Tablet discoloration", "confidence": 0.9},
            "date_received": {"value": "2026-08-01", "confidence": 0.8},
        }
    }
    with patch("app.agents.tracing.get_stream_writer", return_value=lambda *_: None):
        result = await check_completeness(state)
    assert "batch_lot_number" in result["completeness"]["missing_fields"]
    assert result["completeness"]["score"] == 0.75
