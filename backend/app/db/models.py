import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, ForeignKey, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class Complaint(Base):
    __tablename__ = "complaints"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    complaint_number: Mapped[str] = mapped_column(String(32), unique=True)
    product_name: Mapped[str] = mapped_column(String(255))
    batch_lot_number: Mapped[str] = mapped_column(String(100))
    customer_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    customer_contact: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source: Mapped[str] = mapped_column(String(20), default="manual")
    date_received: Mapped[datetime] = mapped_column(server_default=func.now())
    description: Mapped[str] = mapped_column(String)
    category: Mapped[str | None] = mapped_column(String(100), nullable=True)
    severity: Mapped[str | None] = mapped_column(String(20), nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="New")
    assigned_to: Mapped[str | None] = mapped_column(String(255), nullable=True)
    raw_input_type: Mapped[str] = mapped_column(String(20), default="text")
    raw_input_reference: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())

    attachments: Mapped[list["ComplaintAttachment"]] = relationship(back_populates="complaint")
    assessment: Mapped["AIAssessment | None"] = relationship(
        back_populates="complaint", uselist=False, foreign_keys="AIAssessment.complaint_id"
    )
    history: Mapped[list["ComplaintStatusHistory"]] = relationship(back_populates="complaint")


class ComplaintAttachment(Base):
    __tablename__ = "complaint_attachments"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    complaint_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("complaints.id"))
    filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(100))
    storage_path: Mapped[str] = mapped_column(String(500))
    extracted_text: Mapped[str | None] = mapped_column(String, nullable=True)

    complaint: Mapped["Complaint"] = relationship(back_populates="attachments")


class AIAssessment(Base):
    __tablename__ = "ai_assessments"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    complaint_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("complaints.id"), unique=True)
    extracted_fields: Mapped[dict] = mapped_column(JSON)
    completeness_score: Mapped[float] = mapped_column(default=0.0)
    missing_fields: Mapped[list] = mapped_column(JSON, default=list)
    duplicate_of_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("complaints.id"), nullable=True)
    duplicate_confidence: Mapped[float | None] = mapped_column(nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(384), nullable=True)
    risk_classification: Mapped[str | None] = mapped_column(String(20), nullable=True)
    risk_rationale: Mapped[str | None] = mapped_column(String, nullable=True)
    regulatory_reportable: Mapped[bool | None] = mapped_column(nullable=True)
    regulatory_rationale: Mapped[str | None] = mapped_column(String, nullable=True)
    root_cause_suggestion: Mapped[str | None] = mapped_column(String, nullable=True)
    capa_recommendation: Mapped[str | None] = mapped_column(String, nullable=True)
    summary: Mapped[str | None] = mapped_column(String, nullable=True)
    agent_trace: Mapped[list] = mapped_column(JSON, default=list)
    model_used: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    complaint: Mapped["Complaint"] = relationship(back_populates="assessment", foreign_keys=[complaint_id])


class ComplaintStatusHistory(Base):
    __tablename__ = "complaint_status_history"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    complaint_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("complaints.id"))
    changed_field: Mapped[str] = mapped_column(String(50))
    old_value: Mapped[str | None] = mapped_column(String, nullable=True)
    new_value: Mapped[str | None] = mapped_column(String, nullable=True)
    changed_by: Mapped[str] = mapped_column(String(255), default="system")
    changed_at: Mapped[datetime] = mapped_column(server_default=func.now())

    complaint: Mapped["Complaint"] = relationship(back_populates="history")
