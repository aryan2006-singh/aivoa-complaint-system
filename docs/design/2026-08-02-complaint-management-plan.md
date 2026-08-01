# AI Complaint Management System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the AI-powered Customer Complaint Management System described in
`docs/design/2026-08-02-complaint-management-system-design.md`: a LangGraph agent
pipeline (extraction, completeness, semantic duplicate detection, risk +
regulatory-reportability classification, root cause, CAPA, summary) behind a FastAPI
SSE endpoint, with a React/Redux frontend that shows the agent thinking live and lets
a reviewer edit and save the result.

**Architecture:** Backend-first, bottom-up: DB layer → individual LangGraph nodes
(each independently testable with a mocked Groq client) → full graph wiring → API →
frontend, finishing with full docker-compose integration, seed data, and README.

**Tech Stack:** Python 3.11+, FastAPI, SQLAlchemy 2.0 (async, `asyncpg`), Pydantic v2,
LangGraph >=0.2.60, `groq` Python SDK, `sentence-transformers`, `pgvector`, `pypdf`,
`tenacity`, `pytest` + `pytest-asyncio` + `httpx`. React 18 + TypeScript + Vite,
Redux Toolkit + RTK Query, Tailwind CSS + shadcn/ui, `vitest` + React Testing
Library. Postgres 16 via the `pgvector/pgvector:pg16` Docker image.

## Global Constraints

- Duplicate detection: 90-day window, 0.85 cosine-similarity short-circuit threshold,
  0.70 "possible duplicate" warning threshold — all three env-configurable, not
  hardcoded (`DUPLICATE_WINDOW_DAYS`, `DUPLICATE_SIM_THRESHOLD`,
  `DUPLICATE_POSSIBLE_THRESHOLD`).
- LLM: Groq `gemma2-9b-it` by default, model name env-configurable
  (`GROQ_MODEL=gemma2-9b-it`). Always use JSON mode (`response_format:
  {"type": "json_object"}`) for structured extraction/classification calls.
- Font: Google Inter, loaded via `@fontsource/inter`.
- No Alembic — tables are created via `Base.metadata.create_all` on backend startup.
  This is a deliberate scope cut for the one-day budget; note it in the README as a
  documented fast-follow (a real production system would use migrations).
- No auth/RBAC, no PDF export — documented fast-follows per the design spec, not
  built in this plan.
- Every task must leave `main` runnable: backend tasks end with passing `pytest`,
  frontend tasks end with passing `vitest` (once the frontend exists) and a clean
  `npm run build`.

---

## Task 1: Backend scaffolding — config, DB models, schemas, docker-compose

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/app/__init__.py`
- Create: `backend/app/config.py`
- Create: `backend/app/db/__init__.py`
- Create: `backend/app/db/session.py`
- Create: `backend/app/db/models.py`
- Create: `backend/app/schemas.py`
- Create: `backend/app/main.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/conftest.py`
- Create: `backend/tests/test_health.py`
- Create: `docker-compose.yml`
- Create: `.env.example`
- Create: `.gitignore`

**Interfaces:**
- Produces: `Settings` class (`backend/app/config.py`) with fields
  `groq_api_key: str`, `groq_model: str = "gemma2-9b-it"`,
  `database_url: str`, `duplicate_window_days: int = 90`,
  `duplicate_sim_threshold: float = 0.85`,
  `duplicate_possible_threshold: float = 0.70`; module-level `settings = Settings()`.
- Produces: `get_db()` async generator dependency (`backend/app/db/session.py`)
  yielding `AsyncSession`.
- Produces: SQLAlchemy models `Complaint`, `ComplaintAttachment`, `AIAssessment`,
  `ComplaintStatusHistory` (`backend/app/db/models.py`), and `Base`.
- Produces: Pydantic schemas in `backend/app/schemas.py`: `ComplaintCreate`,
  `ComplaintRead`, `ComplaintUpdate`, `AIAssessmentRead`, `ExtractedField` (`value:
  str | None`, `confidence: float`), `ExtractedFields` (product_name,
  batch_lot_number, customer_name, customer_contact, date_received, description,
  category, source — each an `ExtractedField`).
- Produces: FastAPI app instance `app` (`backend/app/main.py`) with `GET /api/health`
  returning `{"status": "ok"}`, and a startup hook that runs
  `Base.metadata.create_all`.

- [ ] **Step 1: Scaffold the Python project**

Run:
```bash
mkdir -p backend/app/db backend/app/agents backend/app/services backend/app/api backend/tests
cd backend
```

Create `backend/pyproject.toml`:
```toml
[project]
name = "aivoa-complaint-backend"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "fastapi>=0.115",
  "uvicorn[standard]>=0.30",
  "sqlalchemy>=2.0",
  "asyncpg>=0.29",
  "pydantic>=2.7",
  "pydantic-settings>=2.3",
  "pgvector>=0.3.6",
  "langgraph>=0.2.60",
  "groq>=0.11",
  "sentence-transformers>=3.0",
  "pypdf>=4.3",
  "tenacity>=8.5",
  "python-multipart>=0.0.9",
]

[project.optional-dependencies]
dev = ["pytest>=8.3", "pytest-asyncio>=0.24", "httpx>=0.27"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

- [ ] **Step 2: Install dependencies**

Run: `pip install -e ".[dev]"` (from `backend/`)
Expected: install completes with no errors.

- [ ] **Step 3: Write `backend/app/config.py`**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str
    groq_model: str = "gemma2-9b-it"
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/complaints"
    duplicate_window_days: int = 90
    duplicate_sim_threshold: float = 0.85
    duplicate_possible_threshold: float = 0.70


settings = Settings()
```

- [ ] **Step 4: Write `backend/app/db/session.py`**

```python
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

engine = create_async_engine(settings.database_url, echo=False)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        yield session
```

- [ ] **Step 5: Write `backend/app/db/models.py`**

```python
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
    assessment: Mapped["AIAssessment | None"] = relationship(back_populates="complaint", uselist=False)
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
```

- [ ] **Step 6: Write `backend/app/schemas.py`**

```python
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
```

- [ ] **Step 7: Write `backend/app/main.py`**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.models import Base
from app.db.session import engine

app = FastAPI(title="AIVOA Complaint Management API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(lambda sync_conn: sync_conn.execute(__import__("sqlalchemy").text("CREATE EXTENSION IF NOT EXISTS vector")))
        await conn.run_sync(Base.metadata.create_all)


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok"}
```

- [ ] **Step 8: Write `docker-compose.yml`**

```yaml
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: complaints
    ports: ["5432:5432"]
    volumes: ["pgdata:/var/lib/postgresql/data"]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 10

volumes:
  pgdata:
```

(Backend/frontend services are added in Task 11 once both apps exist.)

- [ ] **Step 9: Write `.env.example` and `.gitignore`**

`.env.example`:
```
GROQ_API_KEY=your-groq-api-key-here
GROQ_MODEL=gemma2-9b-it
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/complaints
DUPLICATE_WINDOW_DAYS=90
DUPLICATE_SIM_THRESHOLD=0.85
DUPLICATE_POSSIBLE_THRESHOLD=0.70
```

`.gitignore`:
```
__pycache__/
*.pyc
.venv/
.env
node_modules/
dist/
.pytest_cache/
```

- [ ] **Step 10: Start the DB and write the health test**

Run: `docker compose up -d db` (from repo root), wait for it healthy
(`docker compose ps`).

Create `backend/tests/conftest.py`:
```python
import os

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/complaints"
)
```

Create `backend/tests/test_health.py`:
```python
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_health() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 11: Run the test**

Run: `pytest backend/tests/test_health.py -v` (from repo root, with `backend` on
`PYTHONPATH` — run from inside `backend/` or set `PYTHONPATH=backend`)
Expected: PASS. The startup hook will create the `vector` extension and all four
tables against the running `db` container.

- [ ] **Step 12: Commit**

```bash
git add backend docker-compose.yml .env.example .gitignore
git commit -m "feat: backend scaffolding, DB models, docker-compose Postgres+pgvector"
```

---

## Task 2: Groq client wrapper, agent state, and `extract_fields` node

**Files:**
- Create: `backend/app/services/groq_client.py`
- Create: `backend/app/agents/state.py`
- Create: `backend/app/agents/tracing.py`
- Create: `backend/app/agents/nodes/extract_fields.py`
- Create: `backend/tests/agents/test_extract_fields.py`

**Interfaces:**
- Consumes: `settings` from `app.config` (Task 1).
- Produces: `async def call_groq_json(system_prompt: str, user_prompt: str) -> dict`
  (`backend/app/services/groq_client.py`) — later nodes call this.
- Produces: `async def call_groq_stream(system_prompt: str, user_prompt: str) ->
  AsyncIterator[str]` (same file) — used by the `summarize` node in Task 5.
- Produces: `ComplaintAgentState` TypedDict (`backend/app/agents/state.py`) with keys:
  `raw_text: str`, `extracted_fields: dict | None`, `completeness: dict | None`,
  `duplicate: dict | None`, `is_duplicate: bool`, `risk: dict | None`,
  `regulatory: dict | None`, `root_cause: dict | None`, `capa: dict | None`,
  `summary: str | None`, `trace: Annotated[list[dict], operator.add]`,
  `errors: Annotated[list[str], operator.add]`.
- Produces: `traced_node(name: str)` decorator (`backend/app/agents/tracing.py`) that
  wraps an `async def node(state) -> dict` function, emits
  `{"step": name, "status": "started"}` via `get_stream_writer()` on entry and
  `{"step": name, "status": "completed", "data": <result>}` on success (or
  `"status": "error"` on exception, appending to `errors` and returning `{}` so the
  graph continues) — every node in Tasks 2-5 is wrapped with this.
- Produces: `extract_fields(state: ComplaintAgentState) -> dict` node
  (`backend/app/agents/nodes/extract_fields.py`) returning
  `{"extracted_fields": {...}}` matching the `ExtractedFields` schema shape.

- [ ] **Step 1: Write the Groq client wrapper**

```python
# backend/app/services/groq_client.py
import json
from collections.abc import AsyncIterator

from groq import AsyncGroq
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

_client = AsyncGroq(api_key=settings.groq_api_key)


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.5, max=4))
async def call_groq_json(system_prompt: str, user_prompt: str) -> dict:
    response = await _client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        repair = await _client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": "Fix the following into valid JSON only, no prose."},
                {"role": "user", "content": raw},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return json.loads(repair.choices[0].message.content)


async def call_groq_stream(system_prompt: str, user_prompt: str) -> AsyncIterator[str]:
    stream = await _client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
```

- [ ] **Step 2: Write the agent state**

```python
# backend/app/agents/state.py
import operator
from typing import Annotated, TypedDict


class ComplaintAgentState(TypedDict, total=False):
    raw_text: str
    extracted_fields: dict | None
    completeness: dict | None
    duplicate: dict | None
    is_duplicate: bool
    risk: dict | None
    regulatory: dict | None
    root_cause: dict | None
    capa: dict | None
    summary: str | None
    trace: Annotated[list[dict], operator.add]
    errors: Annotated[list[str], operator.add]
```

- [ ] **Step 3: Write the tracing decorator**

```python
# backend/app/agents/tracing.py
import functools
from collections.abc import Awaitable, Callable

from langgraph.config import get_stream_writer

from app.agents.state import ComplaintAgentState


def traced_node(name: str):
    def decorator(fn: Callable[[ComplaintAgentState], Awaitable[dict]]):
        @functools.wraps(fn)
        async def wrapper(state: ComplaintAgentState) -> dict:
            writer = get_stream_writer()
            writer({"step": name, "status": "started"})
            try:
                result = await fn(state)
            except Exception as exc:  # noqa: BLE001 - node failures degrade, not crash
                writer({"step": name, "status": "error", "data": str(exc)})
                return {"errors": [f"{name}: {exc}"], "trace": [{"step": name, "status": "error"}]}
            writer({"step": name, "status": "completed", "data": result})
            return {**result, "trace": [{"step": name, "status": "completed"}]}

        return wrapper

    return decorator
```

- [ ] **Step 4: Write the `extract_fields` node**

```python
# backend/app/agents/nodes/extract_fields.py
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
```

- [ ] **Step 5: Write the test with a mocked Groq client**

```python
# backend/tests/agents/test_extract_fields.py
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.nodes.extract_fields import extract_fields


@pytest.mark.asyncio
async def test_extract_fields_returns_structured_data() -> None:
    fake_response = {
        "product_name": {"value": "Amoxicillin 500mg", "confidence": 0.95},
        "batch_lot_number": {"value": "B12345", "confidence": 0.9},
        "customer_name": {"value": None, "confidence": 0.0},
        "customer_contact": {"value": None, "confidence": 0.0},
        "date_received": {"value": "2026-08-01", "confidence": 0.8},
        "description": {"value": "Tablet discoloration reported.", "confidence": 0.9},
        "category": {"value": "Quality", "confidence": 0.85},
        "source": {"value": "email", "confidence": 0.7},
    }
    with (
        patch("app.agents.nodes.extract_fields.call_groq_json", new=AsyncMock(return_value=fake_response)),
        patch("app.agents.tracing.get_stream_writer", return_value=lambda *_: None),
    ):
        result = await extract_fields({"raw_text": "Batch B12345 of Amoxicillin discolored."})
    assert result["extracted_fields"]["product_name"]["value"] == "Amoxicillin 500mg"
    assert result["trace"][0]["status"] == "completed"
```

- [ ] **Step 6: Run the test**

Run: `pytest backend/tests/agents/test_extract_fields.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/app/services backend/app/agents backend/tests/agents
git commit -m "feat: Groq client wrapper, agent state, tracing decorator, extract_fields node"
```

---

## Task 3: `check_completeness` and `check_duplicates` nodes (pgvector semantic search)

**Files:**
- Create: `backend/app/services/embeddings.py`
- Create: `backend/app/db/repository.py`
- Create: `backend/app/agents/nodes/check_completeness.py`
- Create: `backend/app/agents/nodes/check_duplicates.py`
- Create: `backend/tests/agents/test_check_completeness.py`
- Create: `backend/tests/agents/test_check_duplicates.py`

**Interfaces:**
- Consumes: `ComplaintAgentState`, `traced_node`, `call_groq_json` (Task 2);
  `Complaint`, `AIAssessment`, `SessionLocal` (Task 1).
- Produces: `def get_embedding(text: str) -> list[float]`
  (`backend/app/services/embeddings.py`), 384-dim, loads
  `sentence-transformers/all-MiniLM-L6-v2` once at module import.
- Produces: `async def find_similar_complaints(db, embedding, product_name,
  batch_lot_number, window_days, limit=5) -> list[tuple[Complaint, float]]`
  (`backend/app/db/repository.py`) — returns `(complaint, cosine_similarity)` pairs
  ordered by similarity desc, similarity computed as `1 - cosine_distance`.
- Produces: `check_completeness(state) -> dict` returning
  `{"completeness": {"score": float, "missing_fields": list[str]}}`.
- Produces: `check_duplicates(state) -> dict` returning
  `{"duplicate": {"complaint_id": str | None, "confidence": float}, "is_duplicate":
  bool}` — `is_duplicate=True` only when confidence >
  `settings.duplicate_sim_threshold`.

- [ ] **Step 1: Write the embedding service**

```python
# backend/app/services/embeddings.py
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text: str) -> list[float]:
    return _model.encode(text, normalize_embeddings=True).tolist()
```

- [ ] **Step 2: Write the duplicate-search repository function**

```python
# backend/app/db/repository.py
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
```

- [ ] **Step 3: Write `check_completeness`**

```python
# backend/app/agents/nodes/check_completeness.py
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
```

- [ ] **Step 4: Write `check_duplicates`**

```python
# backend/app/agents/nodes/check_duplicates.py
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
```

Note: `embedding` is also added to `ComplaintAgentState` (add it to the TypedDict in
`state.py` alongside the other keys — `embedding: list[float] | None`) so the
`finalize` node in Task 5 can persist it onto the `AIAssessment` row without
recomputing it.

- [ ] **Step 5: Write completeness test**

```python
# backend/tests/agents/test_check_completeness.py
import pytest

from app.agents.nodes.check_completeness import check_completeness


@pytest.mark.asyncio
async def test_flags_missing_batch_number() -> None:
    state = {
        "extracted_fields": {
            "product_name": {"value": "Amoxicillin", "confidence": 0.9},
            "batch_lot_number": {"value": None, "confidence": 0.0},
            "description": {"value": "Tablet discoloration", "confidence": 0.9},
            "date_received": {"value": "2026-08-01", "confidence": 0.8},
        }
    }
    result = await check_completeness(state)
    assert "batch_lot_number" in result["completeness"]["missing_fields"]
    assert result["completeness"]["score"] == 0.75
```

- [ ] **Step 6: Run it**

Run: `pytest backend/tests/agents/test_check_completeness.py -v`
Expected: PASS.

- [ ] **Step 7: Write duplicates test (uses the real test DB from Task 1)**

```python
# backend/tests/agents/test_check_duplicates.py
import uuid

import pytest
from sqlalchemy import delete

from app.agents.nodes.check_duplicates import check_duplicates
from app.db.models import AIAssessment, Complaint
from app.db.session import SessionLocal
from app.services.embeddings import get_embedding


@pytest.mark.asyncio
async def test_detects_near_duplicate_by_semantic_similarity() -> None:
    async with SessionLocal() as db:
        await db.execute(delete(AIAssessment))
        await db.execute(delete(Complaint))
        existing = Complaint(
            id=uuid.uuid4(),
            complaint_number="CMPL-TEST-0001",
            product_name="Amoxicillin 500mg",
            batch_lot_number="B12345",
            source="manual",
            description="Tablets show yellow discoloration in the blister pack.",
            status="New",
        )
        db.add(existing)
        await db.flush()
        db.add(
            AIAssessment(
                complaint_id=existing.id,
                extracted_fields={},
                model_used="test",
                embedding=get_embedding("Tablets show yellow discoloration in the blister pack."),
            )
        )
        await db.commit()

    state = {
        "raw_text": "Noticed yellow discoloration on the tablets inside the blister.",
        "extracted_fields": {
            "product_name": {"value": "Amoxicillin 500mg", "confidence": 0.9},
            "batch_lot_number": {"value": "B12345", "confidence": 0.9},
            "description": {"value": "Noticed yellow discoloration on the tablets inside the blister.", "confidence": 0.9},
        },
    }
    result = await check_duplicates(state)
    assert result["is_duplicate"] is True
    assert result["duplicate"]["confidence"] > 0.85
```

- [ ] **Step 8: Run it**

Run: `pytest backend/tests/agents/test_check_duplicates.py -v`
Expected: PASS (requires `docker compose up -d db` running).

- [ ] **Step 9: Commit**

```bash
git add backend/app/services/embeddings.py backend/app/db/repository.py backend/app/agents/nodes/check_completeness.py backend/app/agents/nodes/check_duplicates.py backend/tests/agents/test_check_completeness.py backend/tests/agents/test_check_duplicates.py
git commit -m "feat: completeness check and pgvector-based semantic duplicate detection"
```

---

## Task 4: `classify_risk`, `regulatory_reportability`, `suggest_root_cause`, `recommend_capa` nodes

**Files:**
- Create: `backend/app/agents/nodes/classify_risk.py`
- Create: `backend/app/agents/nodes/regulatory_reportability.py`
- Create: `backend/app/agents/nodes/suggest_root_cause.py`
- Create: `backend/app/agents/nodes/recommend_capa.py`
- Create: `backend/tests/agents/test_classification_nodes.py`

**Interfaces:**
- Consumes: `ComplaintAgentState`, `traced_node`, `call_groq_json` (Task 2).
- Produces: `classify_risk(state) -> {"risk": {"classification": "Critical"|"Major"|
  "Minor", "rationale": str}}`.
- Produces: `regulatory_reportability(state) -> {"regulatory": {"reportable": bool,
  "rationale": str}}` — consumes `state["risk"]`.
- Produces: `suggest_root_cause(state) -> {"root_cause": {"category":
  "Man"|"Machine"|"Material"|"Method"|"Environment", "explanation": str}}`.
- Produces: `recommend_capa(state) -> {"capa": {"corrective": str, "preventive":
  str}}` — consumes `state["root_cause"]`.

- [ ] **Step 1: Write `classify_risk`**

```python
# backend/app/agents/nodes/classify_risk.py
from app.agents.state import ComplaintAgentState
from app.agents.tracing import traced_node
from app.services.groq_client import call_groq_json

SYSTEM_PROMPT = """You are a pharmaceutical QMS risk assessor. Classify the patient
safety risk of this complaint as one of "Critical", "Major", or "Minor" based on
potential patient harm, and explain why in 1-2 sentences. Return strict JSON:
{"classification": "Critical"|"Major"|"Minor", "rationale": "..."}"""


@traced_node("classify_risk")
async def classify_risk(state: ComplaintAgentState) -> dict:
    fields = state["extracted_fields"] or {}
    description = (fields.get("description") or {}).get("value") or state["raw_text"]
    result = await call_groq_json(SYSTEM_PROMPT, description)
    return {"risk": result}
```

- [ ] **Step 2: Write `regulatory_reportability`**

```python
# backend/app/agents/nodes/regulatory_reportability.py
from app.agents.state import ComplaintAgentState
from app.agents.tracing import traced_node
from app.services.groq_client import call_groq_json

SYSTEM_PROMPT = """You are a pharmaceutical regulatory affairs assistant. Given a
complaint description and its assessed risk level, decide whether this complaint
pattern would typically warrant a regulatory report (e.g. an adverse event or field
alert report) under standard pharma QMS practice, and explain why in 1-2 sentences.
Return strict JSON: {"reportable": true|false, "rationale": "..."}"""


@traced_node("regulatory_reportability")
async def regulatory_reportability(state: ComplaintAgentState) -> dict:
    fields = state["extracted_fields"] or {}
    description = (fields.get("description") or {}).get("value") or state["raw_text"]
    risk = state.get("risk") or {}
    prompt = f"Complaint: {description}\nAssessed risk: {risk.get('classification')} - {risk.get('rationale')}"
    result = await call_groq_json(SYSTEM_PROMPT, prompt)
    return {"regulatory": result}
```

- [ ] **Step 3: Write `suggest_root_cause`**

```python
# backend/app/agents/nodes/suggest_root_cause.py
from app.agents.state import ComplaintAgentState
from app.agents.tracing import traced_node
from app.services.groq_client import call_groq_json

SYSTEM_PROMPT = """You are a pharmaceutical manufacturing quality engineer. Using the
Man/Machine/Material/Method/Environment framework, suggest the single most likely
root cause category for this complaint and explain your reasoning in 1-2 sentences.
Return strict JSON: {"category": "Man"|"Machine"|"Material"|"Method"|"Environment",
"explanation": "..."}"""


@traced_node("suggest_root_cause")
async def suggest_root_cause(state: ComplaintAgentState) -> dict:
    fields = state["extracted_fields"] or {}
    description = (fields.get("description") or {}).get("value") or state["raw_text"]
    result = await call_groq_json(SYSTEM_PROMPT, description)
    return {"root_cause": result}
```

- [ ] **Step 4: Write `recommend_capa`**

```python
# backend/app/agents/nodes/recommend_capa.py
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
```

- [ ] **Step 5: Write tests for all four nodes**

```python
# backend/tests/agents/test_classification_nodes.py
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.nodes.classify_risk import classify_risk
from app.agents.nodes.recommend_capa import recommend_capa
from app.agents.nodes.regulatory_reportability import regulatory_reportability
from app.agents.nodes.suggest_root_cause import suggest_root_cause


@pytest.mark.asyncio
async def test_classify_risk() -> None:
    fake = {"classification": "Major", "rationale": "Discoloration may indicate degradation."}
    with (
        patch("app.agents.nodes.classify_risk.call_groq_json", new=AsyncMock(return_value=fake)),
        patch("app.agents.tracing.get_stream_writer", return_value=lambda *_: None),
    ):
        result = await classify_risk({"raw_text": "x", "extracted_fields": {}})
    assert result["risk"]["classification"] == "Major"


@pytest.mark.asyncio
async def test_regulatory_reportability_uses_risk_context() -> None:
    fake = {"reportable": True, "rationale": "Potential product degradation affecting efficacy."}
    with (
        patch("app.agents.nodes.regulatory_reportability.call_groq_json", new=AsyncMock(return_value=fake)) as mock_call,
        patch("app.agents.tracing.get_stream_writer", return_value=lambda *_: None),
    ):
        result = await regulatory_reportability(
            {"raw_text": "x", "extracted_fields": {}, "risk": {"classification": "Major", "rationale": "..."}}
        )
    assert result["regulatory"]["reportable"] is True
    assert "Major" in mock_call.call_args.args[1]


@pytest.mark.asyncio
async def test_suggest_root_cause() -> None:
    fake = {"category": "Material", "explanation": "Likely raw material degradation."}
    with (
        patch("app.agents.nodes.suggest_root_cause.call_groq_json", new=AsyncMock(return_value=fake)),
        patch("app.agents.tracing.get_stream_writer", return_value=lambda *_: None),
    ):
        result = await suggest_root_cause({"raw_text": "x", "extracted_fields": {}})
    assert result["root_cause"]["category"] == "Material"


@pytest.mark.asyncio
async def test_recommend_capa_uses_root_cause_context() -> None:
    fake = {"corrective": "Quarantine batch B12345.", "preventive": "Add incoming material stability testing."}
    with (
        patch("app.agents.nodes.recommend_capa.call_groq_json", new=AsyncMock(return_value=fake)) as mock_call,
        patch("app.agents.tracing.get_stream_writer", return_value=lambda *_: None),
    ):
        result = await recommend_capa(
            {"raw_text": "x", "extracted_fields": {}, "root_cause": {"category": "Material", "explanation": "..."}}
        )
    assert "Quarantine" in result["capa"]["corrective"]
    assert "Material" in mock_call.call_args.args[1]
```

- [ ] **Step 6: Run the tests**

Run: `pytest backend/tests/agents/test_classification_nodes.py -v`
Expected: PASS (4 tests).

- [ ] **Step 7: Commit**

```bash
git add backend/app/agents/nodes/classify_risk.py backend/app/agents/nodes/regulatory_reportability.py backend/app/agents/nodes/suggest_root_cause.py backend/app/agents/nodes/recommend_capa.py backend/tests/agents/test_classification_nodes.py
git commit -m "feat: risk, regulatory reportability, root cause, and CAPA nodes"
```

---

## Task 5: `summarize`/`finalize` nodes, full graph wiring, integration tests

**Files:**
- Create: `backend/app/agents/nodes/summarize.py`
- Create: `backend/app/agents/nodes/finalize.py`
- Create: `backend/app/agents/graph.py`
- Create: `backend/tests/agents/test_graph.py`

**Interfaces:**
- Consumes: every node from Tasks 2-4; `call_groq_stream` (Task 2);
  `SessionLocal`, `Complaint`, `AIAssessment`, `ComplaintStatusHistory` (Task 1).
- Produces: `summarize(state) -> {"summary": str}`, streaming tokens via
  `get_stream_writer()` as `{"step": "summarize", "status": "token", "data": token}`
  in addition to the wrapping `traced_node` start/complete events.
- Produces: `finalize(state) -> {"complaint_id": str}` and
  `finalize_duplicate(state) -> {"complaint_id": None, "duplicate_of": str}` —
  persist to DB, **do not** wrap these two in `traced_node` (they run after the last
  traced step; their own started/completed events are emitted manually so the final
  SSE event always carries the created `complaint_id`).
- Produces: `build_graph()` (`backend/app/agents/graph.py`) returning a compiled
  LangGraph graph with entry `extract_fields`, fan-out to `check_completeness` +
  `check_duplicates`, a `merge_checks` join, a conditional edge routing duplicates to
  `finalize_duplicate`, fan-out from `merge_checks` to `classify_risk` and
  `suggest_root_cause`, `classify_risk -> regulatory_reportability`,
  `suggest_root_cause -> recommend_capa`, both joining into `summarize -> finalize`.
  This is what Task 6's intake endpoint imports and calls `.astream(...,
  stream_mode="custom")` on.

- [ ] **Step 1: Write `summarize` (token-streaming)**

```python
# backend/app/agents/nodes/summarize.py
from langgraph.config import get_stream_writer

from app.agents.state import ComplaintAgentState
from app.services.groq_client import call_groq_stream

SYSTEM_PROMPT = """You are a QMS reviewer assistant. Write a concise 2-3 sentence
summary of this complaint and the AI assessment for a human reviewer to skim."""


async def summarize(state: ComplaintAgentState) -> dict:
    writer = get_stream_writer()
    writer({"step": "summarize", "status": "started"})
    fields = state["extracted_fields"] or {}
    description = (fields.get("description") or {}).get("value") or state["raw_text"]
    context = (
        f"Complaint: {description}\n"
        f"Risk: {(state.get('risk') or {}).get('classification')}\n"
        f"Root cause: {(state.get('root_cause') or {}).get('category')}\n"
        f"CAPA: {(state.get('capa') or {}).get('corrective')}"
    )
    text = ""
    async for token in call_groq_stream(SYSTEM_PROMPT, context):
        text += token
        writer({"step": "summarize", "status": "token", "data": token})
    writer({"step": "summarize", "status": "completed", "data": text})
    return {"summary": text, "trace": [{"step": "summarize", "status": "completed"}]}
```

- [ ] **Step 2: Write `finalize` and `finalize_duplicate`**

```python
# backend/app/agents/nodes/finalize.py
import uuid
from datetime import datetime

from langgraph.config import get_stream_writer
from sqlalchemy import func, select

from app.agents.state import ComplaintAgentState
from app.db.models import AIAssessment, Complaint
from app.db.session import SessionLocal


async def _next_complaint_number(db) -> str:
    year = datetime.utcnow().year
    count = (await db.execute(select(func.count()).select_from(Complaint))).scalar_one()
    return f"CMPL-{year}-{count + 1:06d}"


def _field(fields: dict, name: str) -> str | None:
    return (fields.get(name) or {}).get("value")


async def finalize(state: ComplaintAgentState) -> dict:
    writer = get_stream_writer()
    fields = state["extracted_fields"] or {}
    async with SessionLocal() as db:
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
        await db.flush()
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
```

- [ ] **Step 3: Wire the full graph**

```python
# backend/app/agents/graph.py
from typing import Literal

from langgraph.graph import END, StateGraph

from app.agents.nodes.check_completeness import check_completeness
from app.agents.nodes.check_duplicates import check_duplicates
from app.agents.nodes.classify_risk import classify_risk
from app.agents.nodes.extract_fields import extract_fields
from app.agents.nodes.finalize import finalize, finalize_duplicate
from app.agents.nodes.recommend_capa import recommend_capa
from app.agents.nodes.regulatory_reportability import regulatory_reportability
from app.agents.nodes.suggest_root_cause import suggest_root_cause
from app.agents.nodes.summarize import summarize
from app.agents.state import ComplaintAgentState


async def merge_checks(state: ComplaintAgentState) -> dict:
    return {}


def route_after_duplicate_check(state: ComplaintAgentState) -> Literal["duplicate", "continue"]:
    return "duplicate" if state.get("is_duplicate") else "continue"


def build_graph():
    graph = StateGraph(ComplaintAgentState)

    graph.add_node("extract_fields", extract_fields)
    graph.add_node("check_completeness", check_completeness)
    graph.add_node("check_duplicates", check_duplicates)
    graph.add_node("merge_checks", merge_checks)
    graph.add_node("classify_risk", classify_risk)
    graph.add_node("regulatory_reportability", regulatory_reportability)
    graph.add_node("suggest_root_cause", suggest_root_cause)
    graph.add_node("recommend_capa", recommend_capa)
    graph.add_node("summarize", summarize)
    graph.add_node("finalize", finalize)
    graph.add_node("finalize_duplicate", finalize_duplicate)

    graph.set_entry_point("extract_fields")
    graph.add_edge("extract_fields", "check_completeness")
    graph.add_edge("extract_fields", "check_duplicates")
    graph.add_edge("check_completeness", "merge_checks")
    graph.add_conditional_edges(
        "check_duplicates",
        route_after_duplicate_check,
        {"duplicate": "finalize_duplicate", "continue": "merge_checks"},
    )
    graph.add_edge("merge_checks", "classify_risk")
    graph.add_edge("merge_checks", "suggest_root_cause")
    graph.add_edge("classify_risk", "regulatory_reportability")
    graph.add_edge("suggest_root_cause", "recommend_capa")
    graph.add_edge("regulatory_reportability", "summarize")
    graph.add_edge("recommend_capa", "summarize")
    graph.add_edge("summarize", "finalize")
    graph.add_edge("finalize", END)
    graph.add_edge("finalize_duplicate", END)

    return graph.compile()
```

Also update `backend/app/agents/state.py` to add `embedding: list[float] | None` and
`complaint_id: str | None` and `duplicate_of: str | None` keys (flagged in Task 3,
done here since this is where they're first consumed end-to-end).

- [ ] **Step 4: Write the integration test covering both paths**

```python
# backend/tests/agents/test_graph.py
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import delete

from app.agents.graph import build_graph
from app.db.models import AIAssessment, Complaint
from app.db.session import SessionLocal

FAKE_EXTRACT = {
    "product_name": {"value": "Amoxicillin 500mg", "confidence": 0.9},
    "batch_lot_number": {"value": "B99999", "confidence": 0.9},
    "customer_name": {"value": "Jane Doe", "confidence": 0.8},
    "customer_contact": {"value": "jane@example.com", "confidence": 0.8},
    "date_received": {"value": "2026-08-02", "confidence": 0.8},
    "description": {"value": "Tablets are cracked in several blister cells.", "confidence": 0.9},
    "category": {"value": "Quality", "confidence": 0.8},
    "source": {"value": "email", "confidence": 0.7},
}
FAKE_RISK = {"classification": "Major", "rationale": "Cracked tablets may affect dosing."}
FAKE_REGULATORY = {"reportable": False, "rationale": "No patient harm reported."}
FAKE_ROOT_CAUSE = {"category": "Machine", "explanation": "Likely tableting press fault."}
FAKE_CAPA = {"corrective": "Inspect press tooling.", "preventive": "Add in-process cracked-tablet detection."}


async def _fake_stream(*_args, **_kwargs):
    for token in ["Summary ", "of ", "the ", "complaint."]:
        yield token


@pytest.mark.asyncio
async def test_full_graph_normal_path_persists_complaint() -> None:
    async with SessionLocal() as db:
        await db.execute(delete(AIAssessment))
        await db.execute(delete(Complaint))
        await db.commit()

    with (
        patch("app.agents.nodes.extract_fields.call_groq_json", new=AsyncMock(return_value=FAKE_EXTRACT)),
        patch("app.agents.nodes.classify_risk.call_groq_json", new=AsyncMock(return_value=FAKE_RISK)),
        patch("app.agents.nodes.regulatory_reportability.call_groq_json", new=AsyncMock(return_value=FAKE_REGULATORY)),
        patch("app.agents.nodes.suggest_root_cause.call_groq_json", new=AsyncMock(return_value=FAKE_ROOT_CAUSE)),
        patch("app.agents.nodes.recommend_capa.call_groq_json", new=AsyncMock(return_value=FAKE_CAPA)),
        patch("app.agents.nodes.summarize.call_groq_stream", new=_fake_stream),
    ):
        graph = build_graph()
        events = []
        async for event in graph.astream(
            {"raw_text": "Tablets cracked in blister.", "trace": [], "errors": []},
            stream_mode="custom",
        ):
            events.append(event)

    steps_seen = {e["step"] for e in events}
    assert "extract_fields" in steps_seen
    assert "summarize" in steps_seen
    assert any(e["step"] == "summarize" and e["status"] == "token" for e in events)

    async with SessionLocal() as db:
        from sqlalchemy import select

        complaint = (await db.execute(select(Complaint))).scalar_one()
        assert complaint.batch_lot_number == "B99999"
        assessment = (await db.execute(select(AIAssessment))).scalar_one()
        assert assessment.risk_classification == "Major"
        assert assessment.regulatory_reportable is False


@pytest.mark.asyncio
async def test_full_graph_short_circuits_on_duplicate() -> None:
    with (
        patch("app.agents.nodes.extract_fields.call_groq_json", new=AsyncMock(return_value=FAKE_EXTRACT)),
        patch(
            "app.agents.nodes.check_duplicates.check_duplicates.__wrapped__",
            new=AsyncMock(return_value={"duplicate": {"complaint_id": "fake-id", "confidence": 0.97}, "is_duplicate": True, "embedding": [0.0] * 384}),
        ),
    ):
        graph = build_graph()
        events = []
        async for event in graph.astream(
            {"raw_text": "Tablets cracked in blister.", "trace": [], "errors": []},
            stream_mode="custom",
        ):
            events.append(event)

    steps_seen = {e["step"] for e in events}
    assert "finalize_duplicate" in steps_seen
    assert "classify_risk" not in steps_seen
```

- [ ] **Step 5: Run the tests**

Run: `pytest backend/tests/agents/test_graph.py -v`
Expected: PASS (2 tests). If `check_duplicates.__wrapped__` patching doesn't apply
cleanly because of how `functools.wraps` exposes it, patch
`app.agents.graph.check_duplicates` directly instead (import the module, patch the
name bound in `graph.py`) — either approach is valid, pick whichever the installed
`langgraph`/`functools` behavior makes work and keep it consistent.

- [ ] **Step 6: Run full backend test suite**

Run: `pytest backend/tests -v`
Expected: all tests from Tasks 1-5 PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/app/agents/nodes/summarize.py backend/app/agents/nodes/finalize.py backend/app/agents/graph.py backend/app/agents/state.py backend/tests/agents/test_graph.py
git commit -m "feat: summarize/finalize nodes and full LangGraph pipeline wiring with integration tests"
```

---

## Task 6: Intake API (SSE) with PDF/email parsing

**Files:**
- Create: `backend/app/services/file_parsing.py`
- Create: `backend/app/api/intake.py`
- Modify: `backend/app/main.py` — register the intake router
- Create: `backend/tests/api/test_intake.py`

**Interfaces:**
- Consumes: `build_graph()` (Task 5).
- Produces: `extract_text_from_pdf(data: bytes) -> str`, `extract_text_from_eml(data:
  bytes) -> str` (`backend/app/services/file_parsing.py`).
- Produces: `POST /api/complaints/intake` (`backend/app/api/intake.py`), accepting
  either `multipart/form-data` with a `text` field OR a `file` field (`.pdf`/`.eml`),
  returning `text/event-stream`. Each SSE frame is `data: {json}\n\n` where the JSON
  matches the custom-stream events from Task 5's nodes, plus a final
  `{"step": "done", "status": "completed", "data": {"complaint_id": ..., "duplicate_of":
  ...}}` frame.

- [ ] **Step 1: Write file parsing helpers**

```python
# backend/app/services/file_parsing.py
import email
from io import BytesIO

from pypdf import PdfReader


def extract_text_from_pdf(data: bytes) -> str:
    reader = PdfReader(BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text_from_eml(data: bytes) -> str:
    message = email.message_from_bytes(data)
    parts = [f"Subject: {message.get('Subject', '')}", f"From: {message.get('From', '')}"]
    if message.is_multipart():
        for part in message.walk():
            if part.get_content_type() == "text/plain":
                parts.append(part.get_payload(decode=True).decode(errors="replace"))
    else:
        parts.append(message.get_payload(decode=True).decode(errors="replace"))
    return "\n".join(parts)
```

- [ ] **Step 2: Write the SSE intake endpoint**

```python
# backend/app/api/intake.py
import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.agents.graph import build_graph
from app.services.file_parsing import extract_text_from_eml, extract_text_from_pdf

router = APIRouter(prefix="/api/complaints", tags=["intake"])
_graph = build_graph()


async def _resolve_raw_text(text: str | None, file: UploadFile | None) -> str:
    if file is not None:
        data = await file.read()
        if file.filename and file.filename.lower().endswith(".pdf"):
            return extract_text_from_pdf(data)
        if file.filename and file.filename.lower().endswith(".eml"):
            return extract_text_from_eml(data)
        raise HTTPException(status_code=400, detail="Unsupported file type; use .pdf or .eml")
    if text:
        return text
    raise HTTPException(status_code=400, detail="Provide either 'text' or 'file'")


@router.post("/intake")
async def intake(text: str | None = Form(default=None), file: UploadFile | None = File(default=None)):
    raw_text = await _resolve_raw_text(text, file)

    async def event_stream():
        async for event in _graph.astream(
            {"raw_text": raw_text, "trace": [], "errors": []}, stream_mode="custom"
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

- [ ] **Step 3: Register the router**

Modify `backend/app/main.py`, adding after the `app = FastAPI(...)` line:
```python
from app.api.intake import router as intake_router

app.include_router(intake_router)
```

- [ ] **Step 4: Write the endpoint test**

```python
# backend/tests/api/test_intake.py
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.tests_helpers import FAKE_EXTRACT, FAKE_RISK, FAKE_REGULATORY, FAKE_ROOT_CAUSE, FAKE_CAPA  # noqa
```

Note: rather than duplicating the fake fixtures, move `FAKE_EXTRACT`, `FAKE_RISK`,
`FAKE_REGULATORY`, `FAKE_ROOT_CAUSE`, `FAKE_CAPA`, and `_fake_stream` from Task 5's
`test_graph.py` into a shared `backend/tests/agents/fixtures.py` module in this step
(small refactor), and import from there in both files:

```python
# backend/tests/agents/fixtures.py
FAKE_EXTRACT = {
    "product_name": {"value": "Amoxicillin 500mg", "confidence": 0.9},
    "batch_lot_number": {"value": "B99999", "confidence": 0.9},
    "customer_name": {"value": "Jane Doe", "confidence": 0.8},
    "customer_contact": {"value": "jane@example.com", "confidence": 0.8},
    "date_received": {"value": "2026-08-02", "confidence": 0.8},
    "description": {"value": "Tablets are cracked in several blister cells.", "confidence": 0.9},
    "category": {"value": "Quality", "confidence": 0.8},
    "source": {"value": "email", "confidence": 0.7},
}
FAKE_RISK = {"classification": "Major", "rationale": "Cracked tablets may affect dosing."}
FAKE_REGULATORY = {"reportable": False, "rationale": "No patient harm reported."}
FAKE_ROOT_CAUSE = {"category": "Machine", "explanation": "Likely tableting press fault."}
FAKE_CAPA = {"corrective": "Inspect press tooling.", "preventive": "Add in-process cracked-tablet detection."}


async def fake_stream(*_args, **_kwargs):
    for token in ["Summary ", "of ", "the ", "complaint."]:
        yield token
```

Update `backend/tests/agents/test_graph.py` to import these from
`tests.agents.fixtures` instead of defining them inline (delete the now-duplicate
module-level definitions).

```python
# backend/tests/api/test_intake.py
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from tests.agents.fixtures import FAKE_CAPA, FAKE_EXTRACT, FAKE_REGULATORY, FAKE_RISK, FAKE_ROOT_CAUSE, fake_stream


@pytest.mark.asyncio
async def test_intake_streams_sse_events_for_pasted_text() -> None:
    with (
        patch("app.agents.nodes.extract_fields.call_groq_json", new=AsyncMock(return_value=FAKE_EXTRACT)),
        patch("app.agents.nodes.classify_risk.call_groq_json", new=AsyncMock(return_value=FAKE_RISK)),
        patch("app.agents.nodes.regulatory_reportability.call_groq_json", new=AsyncMock(return_value=FAKE_REGULATORY)),
        patch("app.agents.nodes.suggest_root_cause.call_groq_json", new=AsyncMock(return_value=FAKE_ROOT_CAUSE)),
        patch("app.agents.nodes.recommend_capa.call_groq_json", new=AsyncMock(return_value=FAKE_CAPA)),
        patch("app.agents.nodes.summarize.call_groq_stream", new=fake_stream),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            async with client.stream(
                "POST", "/api/complaints/intake", data={"text": "Tablets cracked in blister, batch B99999."}
            ) as response:
                body = b""
                async for chunk in response.aiter_bytes():
                    body += chunk
    assert response.status_code == 200
    assert b'"step": "extract_fields"' in body or b'"step":"extract_fields"' in body
    assert b"summarize" in body


@pytest.mark.asyncio
async def test_intake_rejects_missing_text_and_file() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/complaints/intake", data={})
    assert response.status_code == 400
```

- [ ] **Step 5: Run the tests**

Run: `pytest backend/tests -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/file_parsing.py backend/app/api/intake.py backend/app/main.py backend/tests/agents/fixtures.py backend/tests/agents/test_graph.py backend/tests/api/test_intake.py
git commit -m "feat: SSE intake endpoint with PDF/email parsing"
```

---

## Task 7: CRUD endpoints (list, detail, update, trace, history)

**Files:**
- Create: `backend/app/api/complaints.py`
- Modify: `backend/app/main.py` — register the router
- Create: `backend/tests/api/test_complaints.py`

**Interfaces:**
- Consumes: `Complaint`, `AIAssessment`, `ComplaintStatusHistory`, `get_db` (Task 1);
  `ComplaintRead`, `ComplaintUpdate` (Task 1).
- Produces: `GET /api/complaints?status=&severity=&category=&search=&page=&page_size=`
  → `list[ComplaintRead]`; `GET /api/complaints/{id}` → `ComplaintRead`;
  `PATCH /api/complaints/{id}` (body `ComplaintUpdate`) → `ComplaintRead`, writing one
  `ComplaintStatusHistory` row per changed field; `GET /api/complaints/{id}/trace` →
  the stored `agent_trace` JSON; `GET /api/complaints/{id}/history` → list of history
  rows.

- [ ] **Step 1: Write the router**

```python
# backend/app/api/complaints.py
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
```

- [ ] **Step 2: Register the router**

Modify `backend/app/main.py`:
```python
from app.api.complaints import router as complaints_router

app.include_router(complaints_router)
```

- [ ] **Step 3: Write endpoint tests**

```python
# backend/tests/api/test_complaints.py
import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete

from app.db.models import AIAssessment, Complaint, ComplaintStatusHistory
from app.db.session import SessionLocal
from app.main import app


async def _seed_complaint() -> uuid.UUID:
    async with SessionLocal() as db:
        await db.execute(delete(ComplaintStatusHistory))
        await db.execute(delete(AIAssessment))
        await db.execute(delete(Complaint))
        complaint = Complaint(
            id=uuid.uuid4(),
            complaint_number="CMPL-TEST-0002",
            product_name="Ibuprofen 200mg",
            batch_lot_number="B55555",
            source="manual",
            description="Bottle seal broken on arrival.",
            status="New",
        )
        db.add(complaint)
        await db.commit()
        return complaint.id


@pytest.mark.asyncio
async def test_list_and_get_and_patch_complaint() -> None:
    complaint_id = await _seed_complaint()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        listed = await client.get("/api/complaints")
        assert listed.status_code == 200
        assert len(listed.json()) == 1

        detail = await client.get(f"/api/complaints/{complaint_id}")
        assert detail.status_code == 200
        assert detail.json()["product_name"] == "Ibuprofen 200mg"

        patched = await client.patch(f"/api/complaints/{complaint_id}", json={"status": "Under Review"})
        assert patched.status_code == 200
        assert patched.json()["status"] == "Under Review"

        history = await client.get(f"/api/complaints/{complaint_id}/history")
        assert history.status_code == 200
        assert history.json()[0]["field"] == "status"


@pytest.mark.asyncio
async def test_get_missing_complaint_returns_404() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/api/complaints/{uuid.uuid4()}")
    assert response.status_code == 404
```

- [ ] **Step 4: Run the tests**

Run: `pytest backend/tests -v`
Expected: all PASS. Backend is now feature-complete.

- [ ] **Step 5: Commit**

```bash
git add backend/app/api/complaints.py backend/app/main.py backend/tests/api/test_complaints.py
git commit -m "feat: complaint CRUD, trace replay, and audit history endpoints"
```

---

## Task 8: Frontend scaffold (Vite/Redux/Tailwind/shadcn) + Dashboard

**Files:**
- Create: `frontend/` (Vite scaffold — many generated files)
- Create: `frontend/src/services/api.ts`
- Create: `frontend/src/store.ts`
- Create: `frontend/src/pages/Dashboard.tsx`
- Create: `frontend/src/App.tsx` (modify generated version)
- Create: `frontend/src/pages/Dashboard.test.tsx`

**Interfaces:**
- Produces: `api` RTK Query service (`frontend/src/services/api.ts`) with
  `useGetComplaintsQuery(params)`, `useGetComplaintQuery(id)`,
  `useUpdateComplaintMutation()`, `useGetComplaintHistoryQuery(id)` — matches Task 7's
  endpoints exactly (`GET /api/complaints`, `GET /api/complaints/{id}`,
  `PATCH /api/complaints/{id}`, `GET /api/complaints/{id}/history`).
- Produces: `store` (`frontend/src/store.ts`) combining `api.reducer` under
  `api.reducerPath` and the `intake` reducer added in Task 9.
- Produces: `<Dashboard />` page component rendering the complaint list.

- [ ] **Step 1: Scaffold the Vite project**

Run (from repo root):
```bash
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
npm install @reduxjs/toolkit react-redux react-router-dom @fontsource/inter
npm install -D tailwindcss postcss autoprefixer vitest @testing-library/react @testing-library/jest-dom jsdom
npx tailwindcss init -p
```

- [ ] **Step 2: Configure Tailwind**

Edit `frontend/tailwind.config.js` `content` array to
`["./index.html", "./src/**/*.{ts,tsx}"]`. Replace `frontend/src/index.css` with:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```
In `frontend/src/main.tsx`, add `import "@fontsource/inter";` above the existing
imports, and set `body { font-family: 'Inter', sans-serif; }` via a small addition to
`index.css`.

- [ ] **Step 3: Init shadcn/ui**

Run: `npx shadcn@latest init -d` (from `frontend/`), then
`npx shadcn@latest add button table badge tabs card input textarea` to pull in the
components used by later tasks.

- [ ] **Step 4: Write the RTK Query API service**

```typescript
// frontend/src/services/api.ts
import { createApi, fetchBaseQuery } from "@reduxjs/toolkit/query/react";

export interface ExtractedField {
  value: string | null;
  confidence: number;
}

export interface AIAssessment {
  completeness_score: number;
  missing_fields: string[];
  duplicate_of_id: string | null;
  duplicate_confidence: number | null;
  risk_classification: string | null;
  risk_rationale: string | null;
  regulatory_reportable: boolean | null;
  regulatory_rationale: string | null;
  root_cause_suggestion: string | null;
  capa_recommendation: string | null;
  summary: string | null;
}

export interface Complaint {
  id: string;
  complaint_number: string;
  product_name: string;
  batch_lot_number: string;
  customer_name: string | null;
  customer_contact: string | null;
  source: string;
  date_received: string;
  description: string;
  category: string | null;
  severity: string | null;
  status: string;
  assigned_to: string | null;
  assessment: AIAssessment | null;
}

export interface HistoryEntry {
  field: string;
  old_value: string | null;
  new_value: string | null;
  changed_by: string;
  changed_at: string;
}

export const api = createApi({
  reducerPath: "api",
  baseQuery: fetchBaseQuery({ baseUrl: "/api" }),
  tagTypes: ["Complaint"],
  endpoints: (builder) => ({
    getComplaints: builder.query<Complaint[], Record<string, string | number | undefined>>({
      query: (params) => ({ url: "/complaints", params }),
      providesTags: ["Complaint"],
    }),
    getComplaint: builder.query<Complaint, string>({
      query: (id) => `/complaints/${id}`,
      providesTags: ["Complaint"],
    }),
    updateComplaint: builder.mutation<Complaint, { id: string; patch: Partial<Complaint> & { changed_by?: string } }>({
      query: ({ id, patch }) => ({ url: `/complaints/${id}`, method: "PATCH", body: patch }),
      invalidatesTags: ["Complaint"],
    }),
    getComplaintHistory: builder.query<HistoryEntry[], string>({
      query: (id) => `/complaints/${id}/history`,
    }),
  }),
});

export const {
  useGetComplaintsQuery,
  useGetComplaintQuery,
  useUpdateComplaintMutation,
  useGetComplaintHistoryQuery,
} = api;
```

- [ ] **Step 5: Write the store (intake slice stub, filled in Task 9)**

```typescript
// frontend/src/store.ts
import { configureStore } from "@reduxjs/toolkit";

import { api } from "./services/api";
import intakeReducer from "./features/intake/intakeSlice";

export const store = configureStore({
  reducer: {
    [api.reducerPath]: api.reducer,
    intake: intakeReducer,
  },
  middleware: (getDefaultMiddleware) => getDefaultMiddleware().concat(api.middleware),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
```

Create a minimal placeholder `frontend/src/features/intake/intakeSlice.ts` now (full
version replaces it in Task 9):
```typescript
import { createSlice } from "@reduxjs/toolkit";

const intakeSlice = createSlice({
  name: "intake",
  initialState: {},
  reducers: {},
});

export default intakeSlice.reducer;
```

- [ ] **Step 6: Write the Dashboard page**

```tsx
// frontend/src/pages/Dashboard.tsx
import { Link } from "react-router-dom";

import { useGetComplaintsQuery } from "../services/api";

const severityColor: Record<string, string> = {
  Critical: "bg-red-100 text-red-800",
  Major: "bg-amber-100 text-amber-800",
  Minor: "bg-emerald-100 text-emerald-800",
};

export default function Dashboard() {
  const { data: complaints, isLoading } = useGetComplaintsQuery({});

  if (isLoading) return <p className="p-6">Loading complaints...</p>;

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-2xl font-semibold">Customer Complaints</h1>
        <Link to="/intake" className="rounded bg-slate-900 text-white px-4 py-2">
          + Log Complaint
        </Link>
      </div>
      <table className="w-full text-left border-collapse">
        <thead>
          <tr className="border-b text-sm text-slate-500">
            <th className="py-2">Complaint #</th>
            <th>Product</th>
            <th>Batch/Lot</th>
            <th>Severity</th>
            <th>Status</th>
            <th>Date</th>
          </tr>
        </thead>
        <tbody>
          {(complaints ?? []).map((c) => (
            <tr key={c.id} className="border-b hover:bg-slate-50">
              <td className="py-2">
                <Link to={`/complaints/${c.id}`} className="text-blue-600">
                  {c.complaint_number}
                </Link>
              </td>
              <td>{c.product_name}</td>
              <td>{c.batch_lot_number}</td>
              <td>
                <span className={`rounded px-2 py-1 text-xs ${severityColor[c.severity ?? ""] ?? "bg-slate-100 text-slate-700"}`}>
                  {c.severity ?? "Unclassified"}
                </span>
              </td>
              <td>{c.status}</td>
              <td>{new Date(c.date_received).toLocaleDateString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

- [ ] **Step 7: Write a Dashboard test**

```tsx
// frontend/src/pages/Dashboard.test.tsx
import { render, screen } from "@testing-library/react";
import { Provider } from "react-redux";
import { MemoryRouter } from "react-router-dom";
import { configureStore } from "@reduxjs/toolkit";
import { describe, expect, it, vi } from "vitest";

import { api } from "../services/api";
import Dashboard from "./Dashboard";

vi.mock("../services/api", async () => {
  const actual = await vi.importActual<typeof import("../services/api")>("../services/api");
  return {
    ...actual,
    useGetComplaintsQuery: () => ({
      data: [
        {
          id: "1",
          complaint_number: "CMPL-2026-000001",
          product_name: "Amoxicillin",
          batch_lot_number: "B123",
          severity: "Major",
          status: "New",
          date_received: "2026-08-01T00:00:00Z",
        },
      ],
      isLoading: false,
    }),
  };
});

describe("Dashboard", () => {
  it("renders the complaint list", () => {
    const store = configureStore({ reducer: { [api.reducerPath]: api.reducer } });
    render(
      <Provider store={store}>
        <MemoryRouter>
          <Dashboard />
        </MemoryRouter>
      </Provider>
    );
    expect(screen.getByText("CMPL-2026-000001")).toBeInTheDocument();
    expect(screen.getByText("Major")).toBeInTheDocument();
  });
});
```

Add to `frontend/vite.config.ts`: a `test` block
(`test: { environment: "jsdom", globals: true, setupFiles: "./src/setupTests.ts" }`),
and create `frontend/src/setupTests.ts` with `import "@testing-library/jest-dom";`.
Add a `"test": "vitest run"` script to `frontend/package.json`.

- [ ] **Step 8: Run the frontend test**

Run: `npm run test` (from `frontend/`)
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add frontend
git commit -m "feat: frontend scaffold (Vite/Redux/Tailwind/shadcn) with Dashboard page"
```

---

## Task 9: Intake page — SSE streaming hook + live AI Copilot stepper

**Files:**
- Create: `frontend/src/features/intake/intakeSlice.ts` (replaces Task 8 stub)
- Create: `frontend/src/features/intake/useIntakeStream.ts`
- Create: `frontend/src/features/intake/AiCopilotStepper.tsx`
- Create: `frontend/src/pages/Intake.tsx`
- Create: `frontend/src/features/intake/useIntakeStream.test.ts`

**Interfaces:**
- Produces: `intakeSlice` reducer with state `{status: "idle"|"streaming"|"done"|
  "error", steps: Record<string, {status: string; data?: unknown}>, summaryText:
  string, resultComplaintId: string | null, duplicateOf: string | null, error: string
  | null}`, actions `stepUpdated`, `tokenReceived`, `intakeDone`, `intakeError`,
  `intakeReset`.
- Produces: `useIntakeStream()` hook (`frontend/src/features/intake/
  useIntakeStream.ts`) returning `{start(payload: {text?: string; file?: File}):
  void}` — internally does a manual `fetch` + `ReadableStream` read (NOT
  `EventSource`, since the endpoint is `POST` with an optional file body which
  `EventSource` cannot send) against `POST /api/complaints/intake`, parses
  `data: {...}\n\n` frames, and dispatches the actions above.
- Produces: `<AiCopilotStepper />` reading `state.intake` and rendering each pipeline
  step with a status icon plus the live-streamed summary text.

- [ ] **Step 1: Write the intake slice**

```typescript
// frontend/src/features/intake/intakeSlice.ts
import { createSlice, type PayloadAction } from "@reduxjs/toolkit";

export interface StepState {
  status: "pending" | "started" | "completed" | "error";
  data?: unknown;
}

interface IntakeState {
  status: "idle" | "streaming" | "done" | "error";
  steps: Record<string, StepState>;
  summaryText: string;
  resultComplaintId: string | null;
  duplicateOf: string | null;
  error: string | null;
}

const initialState: IntakeState = {
  status: "idle",
  steps: {},
  summaryText: "",
  resultComplaintId: null,
  duplicateOf: null,
  error: null,
};

const intakeSlice = createSlice({
  name: "intake",
  initialState,
  reducers: {
    intakeStarted(state) {
      Object.assign(state, initialState, { status: "streaming" as const });
    },
    stepUpdated(state, action: PayloadAction<{ step: string; status: StepState["status"]; data?: unknown }>) {
      const { step, status, data } = action.payload;
      state.steps[step] = { status, data };
    },
    tokenReceived(state, action: PayloadAction<string>) {
      state.summaryText += action.payload;
    },
    intakeDone(state, action: PayloadAction<{ complaintId: string | null; duplicateOf: string | null }>) {
      state.status = "done";
      state.resultComplaintId = action.payload.complaintId;
      state.duplicateOf = action.payload.duplicateOf;
    },
    intakeError(state, action: PayloadAction<string>) {
      state.status = "error";
      state.error = action.payload;
    },
    intakeReset() {
      return initialState;
    },
  },
});

export const { intakeStarted, stepUpdated, tokenReceived, intakeDone, intakeError, intakeReset } = intakeSlice.actions;
export default intakeSlice.reducer;
```

Update `frontend/src/store.ts` import path — it already points at
`./features/intake/intakeSlice`, so no change needed there.

- [ ] **Step 2: Write the streaming hook**

```typescript
// frontend/src/features/intake/useIntakeStream.ts
import { useCallback } from "react";
import { useDispatch } from "react-redux";

import type { AppDispatch } from "../../store";
import { intakeDone, intakeError, intakeStarted, stepUpdated, tokenReceived } from "./intakeSlice";

interface StartPayload {
  text?: string;
  file?: File;
}

export function useIntakeStream() {
  const dispatch = useDispatch<AppDispatch>();

  const start = useCallback(
    async (payload: StartPayload) => {
      dispatch(intakeStarted());
      const body = new FormData();
      if (payload.file) body.append("file", payload.file);
      if (payload.text) body.append("text", payload.text);

      try {
        const response = await fetch("/api/complaints/intake", { method: "POST", body });
        if (!response.ok || !response.body) {
          throw new Error(`Intake failed with status ${response.status}`);
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const frames = buffer.split("\n\n");
          buffer = frames.pop() ?? "";
          for (const frame of frames) {
            const line = frame.trim();
            if (!line.startsWith("data:")) continue;
            const event = JSON.parse(line.slice("data:".length).trim());
            if (event.step === "finalize" || event.step === "finalize_duplicate") {
              dispatch(
                intakeDone({
                  complaintId: event.data?.complaint_id ?? null,
                  duplicateOf: event.data?.duplicate_of ?? null,
                })
              );
            } else if (event.step === "summarize" && event.status === "token") {
              dispatch(tokenReceived(event.data as string));
            } else {
              dispatch(stepUpdated(event));
            }
          }
        }
      } catch (err) {
        dispatch(intakeError(err instanceof Error ? err.message : "Unknown error"));
      }
    },
    [dispatch]
  );

  return { start };
}
```

- [ ] **Step 3: Write the AI Copilot stepper**

```tsx
// frontend/src/features/intake/AiCopilotStepper.tsx
import { useSelector } from "react-redux";

import type { RootState } from "../../store";

const PIPELINE_STEPS = [
  "extract_fields",
  "check_completeness",
  "check_duplicates",
  "classify_risk",
  "regulatory_reportability",
  "suggest_root_cause",
  "recommend_capa",
  "summarize",
];

const ICON: Record<string, string> = { pending: "○", started: "◐", completed: "●", error: "✕" };

export default function AiCopilotStepper() {
  const { steps, summaryText, status } = useSelector((state: RootState) => state.intake);

  return (
    <div className="rounded border p-4 space-y-2">
      <h2 className="font-semibold">AI Copilot</h2>
      <ul className="space-y-1 text-sm">
        {PIPELINE_STEPS.map((step) => {
          const s = steps[step]?.status ?? "pending";
          return (
            <li key={step} className="flex items-center gap-2">
              <span>{ICON[s]}</span>
              <span className={s === "completed" ? "text-slate-900" : "text-slate-400"}>
                {step.replaceAll("_", " ")}
              </span>
            </li>
          );
        })}
      </ul>
      {summaryText && (
        <div className="mt-3 rounded bg-slate-50 p-3 text-sm">
          <strong>Summary:</strong> {summaryText}
        </div>
      )}
      {status === "error" && <p className="text-red-600 text-sm">Something went wrong processing this complaint.</p>}
    </div>
  );
}
```

- [ ] **Step 4: Write the Intake page**

```tsx
// frontend/src/pages/Intake.tsx
import { useState } from "react";
import { useSelector } from "react-redux";
import { useNavigate } from "react-router-dom";

import AiCopilotStepper from "../features/intake/AiCopilotStepper";
import { useIntakeStream } from "../features/intake/useIntakeStream";
import type { RootState } from "../store";

export default function Intake() {
  const [text, setText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const { start } = useIntakeStream();
  const { status, resultComplaintId } = useSelector((state: RootState) => state.intake);
  const navigate = useNavigate();

  if (status === "done" && resultComplaintId) {
    navigate(`/complaints/${resultComplaintId}`);
  }

  return (
    <div className="p-6 grid grid-cols-2 gap-6">
      <div>
        <h1 className="text-2xl font-semibold mb-4">Log Customer Complaint</h1>
        <textarea
          className="w-full border rounded p-3 h-40"
          placeholder="Paste the complaint text (email, portal message, transcript)..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <input
          type="file"
          accept=".pdf,.eml"
          className="mt-3"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        <button
          className="mt-4 rounded bg-slate-900 text-white px-4 py-2 disabled:opacity-50"
          disabled={status === "streaming" || (!text && !file)}
          onClick={() => start({ text: text || undefined, file: file || undefined })}
        >
          {status === "streaming" ? "Processing..." : "Submit"}
        </button>
      </div>
      <AiCopilotStepper />
    </div>
  );
}
```

- [ ] **Step 5: Write a test for the streaming hook**

```typescript
// frontend/src/features/intake/useIntakeStream.test.ts
import { configureStore } from "@reduxjs/toolkit";
import { renderHook, waitFor } from "@testing-library/react";
import { Provider } from "react-redux";
import { describe, expect, it, vi } from "vitest";

import intakeReducer from "./intakeSlice";
import { useIntakeStream } from "./useIntakeStream";

function sseBody(events: object[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  const frames = events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join("");
  return new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode(frames));
      controller.close();
    },
  });
}

describe("useIntakeStream", () => {
  it("parses SSE frames and dispatches completion", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        body: sseBody([
          { step: "extract_fields", status: "completed" },
          { step: "summarize", status: "token", data: "Hello " },
          { step: "finalize", status: "completed", data: { complaint_id: "abc-123" } },
        ]),
      })
    );

    const store = configureStore({ reducer: { intake: intakeReducer } });
    const wrapper = ({ children }: { children: React.ReactNode }) => <Provider store={store}>{children}</Provider>;
    const { result } = renderHook(() => useIntakeStream(), { wrapper });

    await result.current.start({ text: "test complaint" });

    await waitFor(() => {
      expect(store.getState().intake.status).toBe("done");
    });
    expect(store.getState().intake.resultComplaintId).toBe("abc-123");
    expect(store.getState().intake.summaryText).toBe("Hello ");
  });
});
```

- [ ] **Step 6: Run frontend tests**

Run: `npm run test` (from `frontend/`)
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/features/intake frontend/src/pages/Intake.tsx frontend/src/store.ts
git commit -m "feat: intake page with SSE streaming hook and live AI Copilot stepper"
```

---

## Task 10: Review form + Risk Assessment panel, Detail page, audit trail

**Files:**
- Create: `frontend/src/features/complaints/ComplaintForm.tsx`
- Create: `frontend/src/features/complaints/RiskAssessmentPanel.tsx`
- Create: `frontend/src/features/complaints/AuditTrailTimeline.tsx`
- Create: `frontend/src/pages/ComplaintDetail.tsx`
- Modify: `frontend/src/App.tsx` — add routes
- Create: `frontend/src/features/complaints/ComplaintForm.test.tsx`

**Interfaces:**
- Consumes: `useGetComplaintQuery`, `useUpdateComplaintMutation`,
  `useGetComplaintHistoryQuery` (Task 8); `Complaint`, `AIAssessment`,
  `HistoryEntry` types (Task 8).
- Produces: `<ComplaintForm complaint={Complaint} />` — editable fields, calls
  `useUpdateComplaintMutation` on save.
- Produces: `<RiskAssessmentPanel assessment={AIAssessment} />` — read-only display
  of completeness, duplicate warning, risk, regulatory flag, root cause, CAPA,
  summary.
- Produces: `<AuditTrailTimeline complaintId={string} />`.
- Produces: `<ComplaintDetail />` page composing all three, routed at
  `/complaints/:id`.

- [ ] **Step 1: Write `ComplaintForm`**

```tsx
// frontend/src/features/complaints/ComplaintForm.tsx
import { useState } from "react";

import { useUpdateComplaintMutation, type Complaint } from "../../services/api";

export default function ComplaintForm({ complaint }: { complaint: Complaint }) {
  const [fields, setFields] = useState({
    product_name: complaint.product_name,
    batch_lot_number: complaint.batch_lot_number,
    customer_name: complaint.customer_name ?? "",
    description: complaint.description,
    status: complaint.status,
  });
  const [updateComplaint, { isLoading }] = useUpdateComplaintMutation();

  const set = (key: keyof typeof fields) => (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) =>
    setFields((f) => ({ ...f, [key]: e.target.value }));

  return (
    <div className="space-y-3">
      <h2 className="font-semibold">Log Customer Complaint</h2>
      <label className="block text-sm">
        Product Name
        <input className="w-full border rounded p-2" value={fields.product_name} onChange={set("product_name")} />
      </label>
      <label className="block text-sm">
        Batch/Lot Number
        <input className="w-full border rounded p-2" value={fields.batch_lot_number} onChange={set("batch_lot_number")} />
      </label>
      <label className="block text-sm">
        Customer Name
        <input className="w-full border rounded p-2" value={fields.customer_name} onChange={set("customer_name")} />
      </label>
      <label className="block text-sm">
        Description
        <textarea className="w-full border rounded p-2 h-24" value={fields.description} onChange={set("description")} />
      </label>
      <label className="block text-sm">
        Status
        <select className="w-full border rounded p-2" value={fields.status} onChange={set("status")}>
          {["New", "Under Review", "CAPA Initiated", "Closed"].map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      </label>
      <button
        className="rounded bg-slate-900 text-white px-4 py-2 disabled:opacity-50"
        disabled={isLoading}
        onClick={() => updateComplaint({ id: complaint.id, patch: { ...fields, changed_by: "reviewer" } })}
      >
        Save Complaint
      </button>
    </div>
  );
}
```

- [ ] **Step 2: Write `RiskAssessmentPanel`**

```tsx
// frontend/src/features/complaints/RiskAssessmentPanel.tsx
import { Link } from "react-router-dom";

import type { AIAssessment } from "../../services/api";

export default function RiskAssessmentPanel({ assessment }: { assessment: AIAssessment | null }) {
  if (!assessment) return <p className="text-sm text-slate-500">No AI assessment available.</p>;

  return (
    <div className="rounded border p-4 space-y-3">
      <h2 className="font-semibold">AI Copilot Risk Assessment</h2>

      <div>
        <p className="text-sm font-medium">Completeness: {Math.round(assessment.completeness_score * 100)}%</p>
        {assessment.missing_fields.length > 0 && (
          <p className="text-xs text-amber-700">Missing: {assessment.missing_fields.join(", ")}</p>
        )}
      </div>

      {assessment.duplicate_of_id && (
        <p className="text-sm text-amber-700">
          Possible duplicate of{" "}
          <Link className="underline" to={`/complaints/${assessment.duplicate_of_id}`}>
            existing complaint
          </Link>{" "}
          ({Math.round((assessment.duplicate_confidence ?? 0) * 100)}% match)
        </p>
      )}

      <div>
        <p className="text-sm font-medium">Risk: {assessment.risk_classification ?? "Unclassified"}</p>
        <p className="text-xs text-slate-600">{assessment.risk_rationale}</p>
      </div>

      <div>
        <p className="text-sm font-medium">
          Regulatory Reportable: {assessment.regulatory_reportable === null ? "Unknown" : assessment.regulatory_reportable ? "Yes" : "No"}
        </p>
        <p className="text-xs text-slate-600">{assessment.regulatory_rationale}</p>
      </div>

      <div>
        <p className="text-sm font-medium">Root Cause</p>
        <p className="text-xs text-slate-600">{assessment.root_cause_suggestion}</p>
      </div>

      <div>
        <p className="text-sm font-medium">CAPA Recommendation</p>
        <p className="text-xs text-slate-600">{assessment.capa_recommendation}</p>
      </div>

      <div>
        <p className="text-sm font-medium">Summary</p>
        <p className="text-xs text-slate-600">{assessment.summary}</p>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Write `AuditTrailTimeline`**

```tsx
// frontend/src/features/complaints/AuditTrailTimeline.tsx
import { useGetComplaintHistoryQuery } from "../../services/api";

export default function AuditTrailTimeline({ complaintId }: { complaintId: string }) {
  const { data: history } = useGetComplaintHistoryQuery(complaintId);

  if (!history || history.length === 0) return null;

  return (
    <div className="rounded border p-4">
      <h2 className="font-semibold mb-2">Audit Trail</h2>
      <ul className="space-y-1 text-xs text-slate-600">
        {history.map((h, i) => (
          <li key={i}>
            {new Date(h.changed_at).toLocaleString()} — {h.changed_by} changed {h.field} from "{h.old_value}" to "{h.new_value}"
          </li>
        ))}
      </ul>
    </div>
  );
}
```

- [ ] **Step 4: Write the Detail page and wire routes**

```tsx
// frontend/src/pages/ComplaintDetail.tsx
import { useParams } from "react-router-dom";

import AuditTrailTimeline from "../features/complaints/AuditTrailTimeline";
import ComplaintForm from "../features/complaints/ComplaintForm";
import RiskAssessmentPanel from "../features/complaints/RiskAssessmentPanel";
import { useGetComplaintQuery } from "../services/api";

export default function ComplaintDetail() {
  const { id } = useParams<{ id: string }>();
  const { data: complaint, isLoading } = useGetComplaintQuery(id!);

  if (isLoading || !complaint) return <p className="p-6">Loading...</p>;

  return (
    <div className="p-6 grid grid-cols-2 gap-6">
      <ComplaintForm complaint={complaint} />
      <div className="space-y-4">
        <RiskAssessmentPanel assessment={complaint.assessment} />
        <AuditTrailTimeline complaintId={complaint.id} />
      </div>
    </div>
  );
}
```

Modify `frontend/src/App.tsx`:
```tsx
import { BrowserRouter, Route, Routes } from "react-router-dom";

import ComplaintDetail from "./pages/ComplaintDetail";
import Dashboard from "./pages/Dashboard";
import Intake from "./pages/Intake";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/intake" element={<Intake />} />
        <Route path="/complaints/:id" element={<ComplaintDetail />} />
      </Routes>
    </BrowserRouter>
  );
}
```

Ensure `frontend/src/main.tsx` wraps `<App />` in `<Provider store={store}>`.

- [ ] **Step 5: Write a `ComplaintForm` test**

```tsx
// frontend/src/features/complaints/ComplaintForm.test.tsx
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Provider } from "react-redux";
import { configureStore } from "@reduxjs/toolkit";
import { describe, expect, it, vi } from "vitest";

import { api } from "../../services/api";
import ComplaintForm from "./ComplaintForm";

const complaint = {
  id: "1",
  complaint_number: "CMPL-2026-000001",
  product_name: "Amoxicillin",
  batch_lot_number: "B123",
  customer_name: null,
  customer_contact: null,
  source: "manual",
  date_received: "2026-08-01T00:00:00Z",
  description: "Tablets discolored.",
  category: null,
  severity: "Major",
  status: "New",
  assigned_to: null,
  assessment: null,
};

describe("ComplaintForm", () => {
  it("lets a reviewer edit the product name field", async () => {
    const store = configureStore({
      reducer: { [api.reducerPath]: api.reducer },
      middleware: (getDefault) => getDefault().concat(api.middleware),
    });
    render(
      <Provider store={store}>
        <ComplaintForm complaint={complaint} />
      </Provider>
    );
    const input = screen.getByDisplayValue("Amoxicillin") as HTMLInputElement;
    await userEvent.clear(input);
    await userEvent.type(input, "Ibuprofen");
    expect(input.value).toBe("Ibuprofen");
  });
});
```

- [ ] **Step 6: Run frontend tests and build**

Run: `npm run test` then `npm run build` (from `frontend/`)
Expected: tests PASS, build succeeds with no TypeScript errors.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/features/complaints frontend/src/pages/ComplaintDetail.tsx frontend/src/App.tsx frontend/src/main.tsx
git commit -m "feat: complaint review form, AI risk assessment panel, audit trail, detail page"
```

---

## Task 11: Full docker-compose wiring, seed script, README

**Files:**
- Create: `backend/Dockerfile`
- Create: `frontend/Dockerfile`
- Modify: `docker-compose.yml` — add `backend` and `frontend` services
- Create: `backend/scripts/seed.py`
- Create: `README.md`

**Interfaces:**
- Consumes: everything from Tasks 1-10.
- Produces: `docker-compose up --build` starts `db`, `backend` (port 8000),
  `frontend` (port 5173) as one command.
- Produces: `python backend/scripts/seed.py` inserts ~6 realistic demo complaints
  directly via the DB models, including one deliberately near-duplicate pair (same
  product/batch, semantically similar description) so `check_duplicates` has
  something to demonstrate when a 7th similar complaint is submitted live through the
  UI during the demo video.

- [ ] **Step 1: Write `backend/Dockerfile`**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Write `frontend/Dockerfile`**

```dockerfile
FROM node:20-slim
WORKDIR /app
COPY package*.json .
RUN npm install
COPY . .
EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
```

- [ ] **Step 3: Extend `docker-compose.yml`**

```yaml
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: complaints
    ports: ["5432:5432"]
    volumes: ["pgdata:/var/lib/postgresql/data"]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 10

  backend:
    build: ./backend
    env_file: .env
    environment:
      DATABASE_URL: postgresql+asyncpg://postgres:postgres@db:5432/complaints
    ports: ["8000:8000"]
    depends_on:
      db:
        condition: service_healthy

  frontend:
    build: ./frontend
    ports: ["5173:5173"]
    environment:
      VITE_API_PROXY_TARGET: http://backend:8000
    depends_on: ["backend"]

volumes:
  pgdata:
```

Add a Vite dev-server proxy so relative `/api` calls from the frontend reach the
backend container. Modify `frontend/vite.config.ts` to add:
```typescript
server: {
  proxy: { "/api": "http://localhost:8000" },
},
```
(For the Docker Compose case this should read `process.env.VITE_API_PROXY_TARGET ??
"http://localhost:8000"` so it resolves to the `backend` service name inside the
compose network but still defaults sensibly for local `npm run dev`.)

- [ ] **Step 4: Write the seed script**

```python
# backend/scripts/seed.py
import asyncio
import uuid

from app.db.models import AIAssessment, Complaint
from app.db.models import Base
from app.db.session import SessionLocal, engine
from app.services.embeddings import get_embedding

DEMO_COMPLAINTS = [
    dict(product_name="Amoxicillin 500mg", batch_lot_number="B10021", category="Quality",
         description="Several tablets in the blister pack show yellow discoloration and an unusual odor.",
         severity="Major"),
    dict(product_name="Amoxicillin 500mg", batch_lot_number="B10021", category="Quality",
         description="Customer reports yellow-tinted tablets with an off smell from the same batch.",
         severity="Major"),  # deliberate near-duplicate of the row above
    dict(product_name="Metformin 1000mg", batch_lot_number="B20044", category="Packaging",
         description="Outer carton was crushed during shipping; blister strips intact.",
         severity="Minor"),
    dict(product_name="Insulin Glargine", batch_lot_number="B30099", category="Adverse Event",
         description="Patient reported injection site reaction and reduced efficacy; possible cold-chain excursion.",
         severity="Critical"),
    dict(product_name="Ibuprofen 200mg", batch_lot_number="B40012", category="Quality",
         description="Bottle seal was broken on arrival; tablet count appears correct.",
         severity="Minor"),
    dict(product_name="Losartan 50mg", batch_lot_number="B50077", category="Efficacy",
         description="Patient reports no change in blood pressure after two weeks on this batch.",
         severity="Major"),
]


async def main() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with SessionLocal() as db:
        for i, row in enumerate(DEMO_COMPLAINTS, start=1):
            complaint = Complaint(
                id=uuid.uuid4(),
                complaint_number=f"CMPL-2026-{i:06d}",
                product_name=row["product_name"],
                batch_lot_number=row["batch_lot_number"],
                source="manual",
                description=row["description"],
                category=row["category"],
                severity=row["severity"],
                status="New",
            )
            db.add(complaint)
            await db.flush()
            db.add(
                AIAssessment(
                    complaint_id=complaint.id,
                    extracted_fields={},
                    completeness_score=1.0,
                    missing_fields=[],
                    embedding=get_embedding(row["description"]),
                    risk_classification=row["severity"],
                    summary=row["description"][:120],
                    model_used="seed-script",
                )
            )
        await db.commit()
    print(f"Seeded {len(DEMO_COMPLAINTS)} demo complaints.")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 5: Run the seed script and verify**

Run: `docker compose up -d db && python backend/scripts/seed.py` (with
`GROQ_API_KEY`/`DATABASE_URL` env vars set, e.g. via `set -a; source .env; set +a` on
the shell, or exported directly)
Expected: prints `Seeded 6 demo complaints.`; `GET /api/complaints` (once the backend
is running) returns 6 rows.

- [ ] **Step 6: Write the README**

Write `README.md` at the repo root with, at minimum, these sections (real content,
not placeholders):
- **What this is** — one paragraph, references the AIVOA assignment.
- **Architecture diagram** — a Mermaid `graph TD` block mirroring the LangGraph
  pipeline diagram from the design spec (`docs/design/2026-08-02-complaint-management-system-design.md`).
- **Quickstart** — `cp .env.example .env` (fill in `GROQ_API_KEY`), then
  `docker compose up --build`, then `python backend/scripts/seed.py`, then open
  `http://localhost:5173`.
- **Running tests** — `cd backend && pytest`, `cd frontend && npm run test`.
- **Rubric mapping** — a table with one row per mandatory requirement and each bonus
  feature (Completeness Checker, Root Cause Recommendation, Duplicate Complaint
  Detection, CAPA Recommendation, Complaint Summary, AI Risk Classification, plus the
  Regulatory Reportability addition), each linking to the exact file implementing it
  (e.g. `backend/app/agents/nodes/check_duplicates.py`).
- **Design decisions worth calling out** — the LangGraph branching/parallel design,
  pgvector semantic duplicate search vs. naive matching, SSE token streaming.
- **Documented fast-follows** — auth/RBAC, PDF export, Alembic migrations — explicitly
  named as scope cuts made to hit the one-day budget, not oversights.

- [ ] **Step 7: Full end-to-end manual verification**

Run: `docker compose up --build`, wait for all three services healthy, open
`http://localhost:5173`, submit a pasted-text complaint through `/intake`, confirm
the stepper lights up live and the summary streams token-by-token, confirm the
resulting complaint appears on the Dashboard, then submit a near-duplicate of one of
the seeded complaints and confirm the duplicate warning appears.
Expected: full flow works without errors end to end.

- [ ] **Step 8: Commit**

```bash
git add backend/Dockerfile frontend/Dockerfile docker-compose.yml backend/scripts/seed.py README.md
git commit -m "feat: full docker-compose stack, seed data, and README with rubric mapping"
```

---

## Plan self-review notes

- **Spec coverage:** mandatory stack (React/Redux, FastAPI, LangGraph, Groq,
  Postgres+pgvector, Inter font) — Tasks 1, 8. All six bonus features — Tasks 3-4.
  Regulatory reportability, audit trail, semantic duplicate search, token streaming,
  confidence/explainability — Tasks 3-5, 9-10. PDF/email upload — Task 6.
  Two-video/GitHub-repo/rubric-mapping deliverable support — Task 11 README.
  Fast-follow items (auth, PDF export, Alembic) are explicitly documented, not
  silently dropped.
- **Type consistency:** `ComplaintAgentState` keys introduced in Task 2 (`extracted_fields`,
  `completeness`, `duplicate`, `is_duplicate`, `risk`, `regulatory`, `root_cause`,
  `capa`, `summary`, `trace`, `errors`) are extended in Tasks 3 (`embedding`) and 5
  (`complaint_id`, `duplicate_of`) and used consistently by every downstream node and
  by `graph.py`. Frontend `Complaint`/`AIAssessment` TypeScript interfaces (Task 8)
  match `ComplaintRead`/`AIAssessmentRead` Pydantic schemas (Task 1) field-for-field.
  SSE event shape (`{step, status, data}`) is produced once in `tracing.py` (Task 2)
  and consumed identically by the intake endpoint (Task 6) and `useIntakeStream`
  (Task 9).
- **Granularity note:** given the one-day budget stated for this project, steps are
  grouped at "one node / one endpoint / one component" granularity with its own
  test, rather than atomized to a single assertion per step — each step is still
  independently runnable and testable, per the Task Right-Sizing guidance.
