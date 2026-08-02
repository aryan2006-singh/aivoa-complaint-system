import operator
from typing import Annotated, TypedDict


class ComplaintAgentState(TypedDict, total=False):
    raw_text: str
    extracted_fields: dict | None
    completeness: dict | None
    duplicate: dict | None
    is_duplicate: bool
    embedding: list[float] | None
    risk: dict | None
    regulatory: dict | None
    root_cause: dict | None
    capa: dict | None
    summary: str | None
    complaint_id: str | None
    duplicate_of: str | None
    trace: Annotated[list[dict], operator.add]
    errors: Annotated[list[str], operator.add]
