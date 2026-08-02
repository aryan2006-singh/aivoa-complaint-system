import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import AIAssessment, Complaint, ComplaintStatusHistory
from app.db.session import get_db
from app.schemas import ComplaintRead, ComplaintUpdate

router = APIRouter(prefix="/api/complaints", tags=["complaints"])


@router.get("", response_model=list[ComplaintRead])
async def list_complaints(
    status: str | None = None,
    severity: str | None = None,
    category: str | None = None,
    search: str | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Complaint).options(selectinload(Complaint.assessment))
    if status:
        stmt = stmt.where(Complaint.status == status)
    if severity:
        stmt = stmt.where(Complaint.severity == severity)
    if category:
        stmt = stmt.where(Complaint.category == category)
    if search:
        like = f"%{search}%"
        stmt = stmt.where(or_(Complaint.description.ilike(like), Complaint.product_name.ilike(like)))
    stmt = stmt.order_by(Complaint.date_received.desc()).offset((page - 1) * page_size).limit(page_size)
    return (await db.execute(stmt)).scalars().all()


@router.get("/{complaint_id}", response_model=ComplaintRead)
async def get_complaint(complaint_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    stmt = select(Complaint).options(selectinload(Complaint.assessment)).where(Complaint.id == complaint_id)
    complaint = (await db.execute(stmt)).scalar_one_or_none()
    if complaint is None:
        raise HTTPException(status_code=404, detail="Complaint not found")
    return complaint


@router.patch("/{complaint_id}", response_model=ComplaintRead)
async def update_complaint(complaint_id: uuid.UUID, payload: ComplaintUpdate, db: AsyncSession = Depends(get_db)):
    stmt = select(Complaint).options(selectinload(Complaint.assessment)).where(Complaint.id == complaint_id)
    complaint = (await db.execute(stmt)).scalar_one_or_none()
    if complaint is None:
        raise HTTPException(status_code=404, detail="Complaint not found")

    updates = payload.model_dump(exclude={"changed_by"}, exclude_unset=True)
    for field, new_value in updates.items():
        old_value = getattr(complaint, field)
        if old_value != new_value:
            db.add(
                ComplaintStatusHistory(
                    complaint_id=complaint.id,
                    changed_field=field,
                    old_value=str(old_value) if old_value is not None else None,
                    new_value=str(new_value) if new_value is not None else None,
                    changed_by=payload.changed_by,
                )
            )
            setattr(complaint, field, new_value)
    await db.commit()
    await db.refresh(complaint)
    return complaint


@router.get("/{complaint_id}/trace")
async def get_trace(complaint_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    stmt = select(AIAssessment).where(AIAssessment.complaint_id == complaint_id)
    assessment = (await db.execute(stmt)).scalar_one_or_none()
    if assessment is None:
        raise HTTPException(status_code=404, detail="No assessment for this complaint")
    return {"trace": assessment.agent_trace}


@router.get("/{complaint_id}/history")
async def get_history(complaint_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    stmt = (
        select(ComplaintStatusHistory)
        .where(ComplaintStatusHistory.complaint_id == complaint_id)
        .order_by(ComplaintStatusHistory.changed_at.desc())
    )
    rows = (await db.execute(stmt)).scalars().all()
    return [
        {
            "field": r.changed_field,
            "old_value": r.old_value,
            "new_value": r.new_value,
            "changed_by": r.changed_by,
            "changed_at": r.changed_at.isoformat(),
        }
        for r in rows
    ]
