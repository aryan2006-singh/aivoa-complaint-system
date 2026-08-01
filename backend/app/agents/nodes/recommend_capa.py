from app.agents.state import ComplaintAgentState
from app.agents.tracing import traced_node
from app.services.groq_client import call_groq_json

SYSTEM_PROMPT = """You are a pharmaceutical QMS specialist writing CAPA
(Corrective and Preventive Action) recommendations. Given a complaint and its
likely root cause, propose one concrete corrective action and one concrete
preventive action. Return strict JSON: {"corrective": "...", "preventive": "..."}"""


@traced_node("recommend_capa")
async def recommend_capa(state: ComplaintAgentState) -> dict:
    fields = state["extracted_fields"] or {}
    description = (fields.get("description") or {}).get("value") or state["raw_text"]
    root_cause = state.get("root_cause") or {}
    prompt = f"Complaint: {description}\nRoot cause: {root_cause.get('category')} - {root_cause.get('explanation')}"
    result = await call_groq_json(SYSTEM_PROMPT, prompt)
    return {"capa": result}
