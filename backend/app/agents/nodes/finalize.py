import uuid
from datetime import datetime

from langgraph.config import get_stream_writer
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.agents.state import ComplaintAgentState
from app.db.models import AIAssessment, Complaint
from app.db.session import SessionLocal

MAX_COMPLAINT_NUMBER_ATTEMPTS = 3


async def _next_complaint_number(db) -> str:
    year = datetime.utcnow().year
    count = (await db.execute(select(func.count()).select_from(Complaint))).scalar_one()
    return f"CMPL-{year}-{count + 1:06d}"


def _field(fields: dict, name: str) -> str | None:
    return (fields.get(name) or {}).get("value")


async def finalize(state: ComplaintAgentState) -> dict:
    writer = get_stream_writer()
    fields = state.get("extracted_fields") or {}
    async with SessionLocal() as db:
        # `complaint_number` is unique, and `_next_complaint_number` derives it from
        # a plain COUNT(*) -- two concurrent intakes can race and read the same
        # count, so the flush below can raise IntegrityError. This is a minimal
        # retry (regenerate the number, try again) rather than a DB-sequence-based
        # solution, consistent with this project's "no Alembic migrations" scope.
        for attempt in range(1, MAX_COMPLAINT_NUMBER_ATTEMPTS + 1):
            complaint = Complaint(
                id=uuid.uuid4(),
                complaint_number=await _next_complaint_number(db),
                product_name=_field(fields, "product_name") or "Unknown",
                batch_lot_number=_field(fields, "batch_lot_number") or "Unknown",
                customer_name=_field(fields, "customer_name"),
                customer_contact=_field(fields, "customer_contact"),
                source=_field(fields, "source") or "manual",
                description=_field(fields, "description") or state["raw_text"],
                category=_field(fields, "category"),
                severity=(state.get("risk") or {}).get("classification"),
                status="New",
            )
            db.add(complaint)
            try:
                await db.flush()
            except IntegrityError:
                await db.rollback()
                if attempt == MAX_COMPLAINT_NUMBER_ATTEMPTS:
                    raise
                continue
            else:
                break
        db.add(
            AIAssessment(
                complaint_id=complaint.id,
                extracted_fields=fields,
                completeness_score=(state.get("completeness") or {}).get("score", 0.0),
                missing_fields=(state.get("completeness") or {}).get("missing_fields", []),
                duplicate_confidence=(state.get("duplicate") or {}).get("confidence"),
                embedding=state.get("embedding"),
                risk_classification=(state.get("risk") or {}).get("classification"),
                risk_rationale=(state.get("risk") or {}).get("rationale"),
                regulatory_reportable=(state.get("regulatory") or {}).get("reportable"),
                regulatory_rationale=(state.get("regulatory") or {}).get("rationale"),
                root_cause_suggestion=(state.get("root_cause") or {}).get("explanation"),
                capa_recommendation=(state.get("capa") or {}).get("corrective"),
                summary=state.get("summary"),
                agent_trace=state.get("trace", []),
                model_used="gemma2-9b-it",
            )
        )
        await db.commit()
        writer({"step": "finalize", "status": "completed", "data": {"complaint_id": str(complaint.id)}})
        return {"complaint_id": str(complaint.id)}


async def finalize_duplicate(state: ComplaintAgentState) -> dict:
    writer = get_stream_writer()
    dup_id = (state.get("duplicate") or {}).get("complaint_id")
    writer({"step": "finalize_duplicate", "status": "completed", "data": {"duplicate_of": dup_id}})
    return {"complaint_id": None, "duplicate_of": dup_id}
