from app.agents.state import ComplaintAgentState
from app.agents.tracing import traced_node
from app.services.groq_client import call_groq_json

SYSTEM_PROMPT = """You are a pharmaceutical regulatory affairs assistant. Given a
complaint description and its assessed risk level, decide whether this complaint
pattern would typically warrant a regulatory report (e.g. an adverse event or field
alert report) under standard pharma QMS practice, and explain why in 1-2 sentences.
Return strict JSON: {"reportable": true|false, "rationale": "..."}"""


@traced_node("regulatory_reportability")
async def regulatory_reportability(state: ComplaintAgentState) -> dict:
    fields = state["extracted_fields"] or {}
    description = (fields.get("description") or {}).get("value") or state["raw_text"]
    risk = state.get("risk") or {}
    prompt = f"Complaint: {description}\nAssessed risk: {risk.get('classification')} - {risk.get('rationale')}"
    result = await call_groq_json(SYSTEM_PROMPT, prompt)
    return {"regulatory": result}
