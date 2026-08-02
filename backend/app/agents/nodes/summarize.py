from langgraph.config import get_stream_writer

from app.agents.state import ComplaintAgentState
from app.services.groq_client import call_groq_stream

SYSTEM_PROMPT = """You are a QMS reviewer assistant. Write a concise 2-3 sentence
summary of this complaint and the AI assessment for a human reviewer to skim."""


async def summarize(state: ComplaintAgentState) -> dict:
    writer = get_stream_writer()
    writer({"step": "summarize", "status": "started"})
    fields = state["extracted_fields"] or {}
    description = (fields.get("description") or {}).get("value") or state["raw_text"]
    context = (
        f"Complaint: {description}\n"
        f"Risk: {(state.get('risk') or {}).get('classification')}\n"
        f"Root cause: {(state.get('root_cause') or {}).get('category')}\n"
        f"CAPA: {(state.get('capa') or {}).get('corrective')}"
    )
    text = ""
    async for token in call_groq_stream(SYSTEM_PROMPT, context):
        text += token
        writer({"step": "summarize", "status": "token", "data": token})
    writer({"step": "summarize", "status": "completed", "data": text})
    return {"summary": text, "trace": [{"step": "summarize", "status": "completed"}]}
