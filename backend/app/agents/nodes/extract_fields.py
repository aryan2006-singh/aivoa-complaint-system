from app.agents.state import ComplaintAgentState
from app.agents.tracing import traced_node
from app.services.groq_client import call_groq_json

SYSTEM_PROMPT = """You extract structured fields from a pharmaceutical product
complaint. Return strict JSON with this shape, where every field has a "value"
(string or null) and a "confidence" (0.0-1.0 float reflecting how certain you are
the value is correct and present in the text):
{"product_name": {"value": ..., "confidence": ...},
 "batch_lot_number": {"value": ..., "confidence": ...},
 "customer_name": {"value": ..., "confidence": ...},
 "customer_contact": {"value": ..., "confidence": ...},
 "date_received": {"value": ..., "confidence": ...},
 "description": {"value": ..., "confidence": ...},
 "category": {"value": ..., "confidence": ...},
 "source": {"value": ..., "confidence": ...}}
If a field is genuinely absent from the text, set value to null and confidence to 0.0.
"""


@traced_node("extract_fields")
async def extract_fields(state: ComplaintAgentState) -> dict:
    fields = await call_groq_json(SYSTEM_PROMPT, state["raw_text"])
    return {"extracted_fields": fields}
