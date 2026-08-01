from app.agents.state import ComplaintAgentState
from app.agents.tracing import traced_node
from app.services.groq_client import call_groq_json

SYSTEM_PROMPT = """You are a pharmaceutical manufacturing quality engineer. Using the
Man/Machine/Material/Method/Environment framework, suggest the single most likely
root cause category for this complaint and explain your reasoning in 1-2 sentences.
Return strict JSON: {"category": "Man"|"Machine"|"Material"|"Method"|"Environment",
"explanation": "..."}"""


@traced_node("suggest_root_cause")
async def suggest_root_cause(state: ComplaintAgentState) -> dict:
    fields = state["extracted_fields"] or {}
    description = (fields.get("description") or {}).get("value") or state["raw_text"]
    result = await call_groq_json(SYSTEM_PROMPT, description)
    return {"root_cause": result}
