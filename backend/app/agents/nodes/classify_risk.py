from app.agents.state import ComplaintAgentState
from app.agents.tracing import traced_node
from app.services.groq_client import call_groq_json

SYSTEM_PROMPT = """You are a pharmaceutical QMS risk assessor. Classify the patient
safety risk of this complaint as one of "Critical", "Major", or "Minor" based on
potential patient harm, and explain why in 1-2 sentences. Return strict JSON:
{"classification": "Critical"|"Major"|"Minor", "rationale": "..."}"""


@traced_node("classify_risk")
async def classify_risk(state: ComplaintAgentState) -> dict:
    fields = state["extracted_fields"] or {}
    description = (fields.get("description") or {}).get("value") or state["raw_text"]
    result = await call_groq_json(SYSTEM_PROMPT, description)
    return {"risk": result}
