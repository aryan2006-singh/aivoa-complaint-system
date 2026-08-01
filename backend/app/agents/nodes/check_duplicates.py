from app.agents.state import ComplaintAgentState
from app.agents.tracing import traced_node
from app.config import settings
from app.db.repository import find_similar_complaints
from app.db.session import SessionLocal
from app.services.embeddings import get_embedding


@traced_node("check_duplicates")
async def check_duplicates(state: ComplaintAgentState) -> dict:
    fields = state["extracted_fields"] or {}
    description = (fields.get("description") or {}).get("value") or state["raw_text"]
    embedding = get_embedding(description)

    product = (fields.get("product_name") or {}).get("value")
    batch = (fields.get("batch_lot_number") or {}).get("value")

    matches: list[tuple] = []
    if product and batch:
        async with SessionLocal() as db:
            matches = await find_similar_complaints(
                db, embedding, product, batch, settings.duplicate_window_days
            )

    if matches:
        best_complaint, best_score = matches[0]
        is_duplicate = best_score > settings.duplicate_sim_threshold
        return {
            "duplicate": {"complaint_id": str(best_complaint.id), "confidence": round(best_score, 3)},
            "is_duplicate": is_duplicate,
            "embedding": embedding,
        }
    return {"duplicate": {"complaint_id": None, "confidence": 0.0}, "is_duplicate": False, "embedding": embedding}
