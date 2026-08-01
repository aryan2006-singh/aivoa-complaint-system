import uuid
from datetime import datetime

from pydantic import BaseModel


class ExtractedField(BaseModel):
    value: str | None = None
    confidence: float = 0.0


class ExtractedFields(BaseModel):
    product_name: ExtractedField = ExtractedField()
    batch_lot_number: ExtractedField = ExtractedField()
    customer_name: ExtractedField = ExtractedField()
    customer_contact: ExtractedField = ExtractedField()
    date_received: ExtractedField = ExtractedField()
    description: ExtractedField = ExtractedField()
    category: ExtractedField = ExtractedField()
    source: ExtractedField = ExtractedField()


class ComplaintCreate(BaseModel):
    text: str


class ComplaintUpdate(BaseModel):
    product_name: str | None = None
    batch_lot_number: str | None = None
    customer_name: str | None = None
    customer_contact: str | None = None
    description: str | None = None
    category: str | None = None
    status: str | None = None
    assigned_to: str | None = None
    changed_by: str = "reviewer"


class AIAssessmentRead(BaseModel):
    completeness_score: float
    missing_fields: list[str]
    duplicate_of_id: uuid.UUID | None
    duplicate_confidence: float | None
    risk_classification: str | None
    risk_rationale: str | None
    regulatory_reportable: bool | None
    regulatory_rationale: str | None
    root_cause_suggestion: str | None
    capa_recommendation: str | None
    summary: str | None
    model_config = {"from_attributes": True}


class ComplaintRead(BaseModel):
    id: uuid.UUID
    complaint_number: str
    product_name: str
    batch_lot_number: str
    customer_name: str | None
    customer_contact: str | None
    source: str
    date_received: datetime
    description: str
    category: str | None
    severity: str | None
    status: str
    assigned_to: str | None
    assessment: AIAssessmentRead | None = None
    model_config = {"from_attributes": True}
