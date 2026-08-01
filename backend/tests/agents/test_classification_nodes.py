from unittest.mock import AsyncMock, patch

import pytest

from app.agents.nodes.classify_risk import classify_risk
from app.agents.nodes.recommend_capa import recommend_capa
from app.agents.nodes.regulatory_reportability import regulatory_reportability
from app.agents.nodes.suggest_root_cause import suggest_root_cause


@pytest.mark.asyncio
async def test_classify_risk() -> None:
    fake = {"classification": "Major", "rationale": "Discoloration may indicate degradation."}
    with (
        patch("app.agents.nodes.classify_risk.call_groq_json", new=AsyncMock(return_value=fake)),
        patch("app.agents.tracing.get_stream_writer", return_value=lambda *_: None),
    ):
        result = await classify_risk({"raw_text": "x", "extracted_fields": {}})
    assert result["risk"]["classification"] == "Major"


@pytest.mark.asyncio
async def test_regulatory_reportability_uses_risk_context() -> None:
    fake = {"reportable": True, "rationale": "Potential product degradation affecting efficacy."}
    with (
        patch("app.agents.nodes.regulatory_reportability.call_groq_json", new=AsyncMock(return_value=fake)) as mock_call,
        patch("app.agents.tracing.get_stream_writer", return_value=lambda *_: None),
    ):
        result = await regulatory_reportability(
            {"raw_text": "x", "extracted_fields": {}, "risk": {"classification": "Major", "rationale": "..."}}
        )
    assert result["regulatory"]["reportable"] is True
    assert "Major" in mock_call.call_args.args[1]


@pytest.mark.asyncio
async def test_suggest_root_cause() -> None:
    fake = {"category": "Material", "explanation": "Likely raw material degradation."}
    with (
        patch("app.agents.nodes.suggest_root_cause.call_groq_json", new=AsyncMock(return_value=fake)),
        patch("app.agents.tracing.get_stream_writer", return_value=lambda *_: None),
    ):
        result = await suggest_root_cause({"raw_text": "x", "extracted_fields": {}})
    assert result["root_cause"]["category"] == "Material"


@pytest.mark.asyncio
async def test_recommend_capa_uses_root_cause_context() -> None:
    fake = {"corrective": "Quarantine batch B12345.", "preventive": "Add incoming material stability testing."}
    with (
        patch("app.agents.nodes.recommend_capa.call_groq_json", new=AsyncMock(return_value=fake)) as mock_call,
        patch("app.agents.tracing.get_stream_writer", return_value=lambda *_: None),
    ):
        result = await recommend_capa(
            {"raw_text": "x", "extracted_fields": {}, "root_cause": {"category": "Material", "explanation": "..."}}
        )
    assert "Quarantine" in result["capa"]["corrective"]
    assert "Material" in mock_call.call_args.args[1]
