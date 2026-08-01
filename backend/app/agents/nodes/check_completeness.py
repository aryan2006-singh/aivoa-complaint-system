from app.agents.state import ComplaintAgentState
from app.agents.tracing import traced_node

REQUIRED_FIELDS = ["product_name", "batch_lot_number", "description", "date_received"]


@traced_node("check_completeness")
async def check_completeness(state: ComplaintAgentState) -> dict:
    fields = state["extracted_fields"] or {}
    missing = [
        name for name in REQUIRED_FIELDS
        if not (fields.get(name) or {}).get("value")
    ]
    score = round(1 - len(missing) / len(REQUIRED_FIELDS), 2)
    return {"completeness": {"score": score, "missing_fields": missing}}
