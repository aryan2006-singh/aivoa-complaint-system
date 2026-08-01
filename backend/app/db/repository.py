from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AIAssessment, Complaint


async def find_similar_complaints(
    db: AsyncSession,
    embedding: list[float],
    product_name: str,
    batch_lot_number: str,
    window_days: int,
    limit: int = 5,
) -> list[tuple[Complaint, float]]:
    cutoff = datetime.utcnow() - timedelta(days=window_days)
    distance = AIAssessment.embedding.cosine_distance(embedding)
    stmt = (
        select(Complaint, distance.label("distance"))
        .join(AIAssessment, AIAssessment.complaint_id == Complaint.id)
        .where(
            Complaint.product_name == product_name,
            Complaint.batch_lot_number == batch_lot_number,
            Complaint.date_received >= cutoff,
            AIAssessment.embedding.is_not(None),
        )
        .order_by(distance)
        .limit(limit)
    )
    rows = (await db.execute(stmt)).all()
    return [(complaint, 1 - dist) for complaint, dist in rows]
