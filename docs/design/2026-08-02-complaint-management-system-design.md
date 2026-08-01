# AI-Powered Customer Complaint Management System — Design Spec

**Context:** AIVOA Round 1 AI Product Engineer (Interns) assignment. Build an AI-powered
Customer Complaint Management System for the pharmaceutical manufacturing industry
(API/FDF QMS domain), matching the mandated stack and the demonstrated workflow
(prompt or PDF/email upload → structured "Log Customer Complaint" form → "AI Copilot
Risk Assessment").

**Goal of this build:** satisfy every mandatory requirement solidly, implement all six
suggested bonus features as one coherent AI pipeline rather than six bolted-on
features, and add domain-authentic depth (semantic duplicate search, regulatory
reportability signal, audit trail, live token streaming, confidence/explainability)
that goes beyond what the brief literally asks for.

**Time budget:** one day. Scope is chosen so the mandatory workflow is rock-solid
first, with enhancements layered on as long as they don't put that at risk. Auth/RBAC
and PDF export are explicitly deferred (documented, not built) to protect the budget.

## Architecture

Monorepo:

```
aivoa-complaint-system/
├── backend/         FastAPI + LangGraph agent + SQLAlchemy/Postgres
├── frontend/         React + Redux Toolkit + Tailwind + shadcn/ui
├── docker-compose.yml  postgres(pgvector) + backend + frontend, one-command startup
└── docs/
```

REST for CRUD. One SSE endpoint streams the LangGraph agent's execution live
(per-node status transitions AND token-level streaming of generated rationale/summary
text) to the frontend during complaint intake.

## Data model (Postgres, pgvector extension enabled)

- **complaints**: id (uuid), complaint_number (e.g. `CMPL-2026-000123`), product_name,
  batch_lot_number, customer_name, customer_contact, source
  (email/phone/portal/manual), date_received, description, category, severity,
  status (New / Under Review / CAPA Initiated / Closed), assigned_to, created_at,
  updated_at, raw_input_type, raw_input_reference.
- **complaint_attachments**: id, complaint_id FK, filename, content_type,
  storage_path, extracted_text.
- **ai_assessments**: id, complaint_id FK, extracted_fields (jsonb, with a per-field
  confidence score), completeness_score, missing_fields (jsonb), duplicate_of_id
  (nullable FK), duplicate_confidence, embedding (vector(384), pgvector), risk_classification,
  risk_rationale, regulatory_reportable (bool), regulatory_rationale,
  root_cause_suggestion, capa_recommendation, summary, agent_trace (jsonb — full
  step-by-step record for replay), model_used, created_at.
- **complaint_status_history**: id, complaint_id FK, changed_field, old_value,
  new_value, changed_by, changed_at — the audit trail.

## LangGraph pipeline

```
extract_fields
   ├─▶ check_completeness ──────────────┐
   └─▶ check_duplicates (pgvector kNN) ─┤
                                          ├── if duplicate_confidence > threshold:
                                          │     → finalize_duplicate (short-circuit)
                                          └── else, run in parallel:
                                                classify_risk ────────────────┐
                                                  └─▶ regulatory_reportability┤
                                                suggest_root_cause ─▶ recommend_capa ─┤
                                                                                       ▼
                                                                                  summarize ─▶ finalize
```

- **extract_fields**: Groq `gemma2-9b-it` call with JSON-mode structured output;
  returns each field plus a 0–1 confidence score. Malformed JSON triggers one
  self-repair retry (ask the model to fix its own output against the schema) before
  falling back to a partial result with an `error` flag on that node.
- **check_completeness**: rule-based check against mandatory QMS fields (product,
  batch/lot, description, date, reporter) plus an LLM pass for ambiguous/low-quality
  free text; produces a completeness_score and missing_fields list.
- **check_duplicates**: embed the complaint description (local sentence-transformer,
  `all-MiniLM-L6-v2`, 384-dim) and run a pgvector cosine-similarity kNN query against
  existing complaints' embeddings, pre-filtered by same product + batch/lot within a
  90-day window (both the window and the threshold below are env-configurable, not
  hardcoded). This replaces naive fuzzy/keyword matching with real semantic
  similarity. Above a 0.85 cosine-similarity confidence threshold, the graph
  short-circuits to `finalize_duplicate` and skips the rest of the pipeline; between
  0.70–0.85 it's surfaced as a "possible duplicate" warning without short-circuiting,
  so a borderline match doesn't silently swallow a real complaint.
- **classify_risk**: LLM classifies severity (Critical/Major/Minor) with rationale,
  referencing patient-safety-impact framing appropriate to pharma QMS.
- **regulatory_reportability**: LLM flags whether the complaint pattern-matches
  something that would typically trigger a regulatory report (e.g. adverse event,
  field alert), with rationale. This is the 7th, non-rubric-listed signal.
- **suggest_root_cause**: LLM proposes likely root cause category (Man/Machine/
  Material/Method/Environment framing) from description + category.
- **recommend_capa**: LLM proposes Corrective and Preventive Action recommendations
  tied to the suggested root cause.
- **summarize**: LLM produces a concise reviewer-facing summary; this node's tokens
  are streamed live to the client (not just a "done" event) for visible effect.
- **finalize / finalize_duplicate**: persists the `ai_assessments` row (including full
  step trace) and emits the final SSE event.

Each node emits `{step, status: started|completed|error, data}` over SSE as it runs;
the `summarize` node additionally streams token deltas.

## API

- `POST /api/complaints/intake` — body is either `{text}` (paste) or multipart file
  (PDF via `pypdf`, or `.eml` via Python's `email` module). Returns
  `text/event-stream` of agent progress, ending in the saved complaint + assessment.
- `GET /api/complaints` — list with filters (status, severity, category, search) +
  pagination.
- `GET /api/complaints/{id}` — detail incl. latest `ai_assessment`.
- `PATCH /api/complaints/{id}` — reviewer edits to AI-extracted fields, status, or
  assignment; writes to `complaint_status_history`.
- `GET /api/complaints/{id}/trace` — replay the stored agent trace without
  re-invoking the LLM (fast, deterministic, good for the demo video).
- `GET /api/complaints/{id}/history` — audit trail.
- `GET /api/health`.

## Frontend (React + Redux Toolkit, Tailwind + shadcn/ui, Inter font)

- **Dashboard**: complaint list/table, severity badges, filters, search.
- **Intake**: tabs for Paste Text / Upload PDF / Upload Email; submitting opens an SSE
  connection driving a live "AI Copilot" stepper (Extract → Completeness → Duplicate
  Check → Risk → Regulatory → Root Cause → CAPA → Summary) with the summary panel's
  text appearing token-by-token as it streams.
- **Review**: auto-populated, editable "Log Customer Complaint" form next to the
  "AI Copilot Risk Assessment" panel (completeness flags, duplicate warning with link
  to the original complaint if found, risk classification + rationale, regulatory
  reportability flag + rationale, root cause, CAPA, summary, and a per-field
  confidence indicator). Reviewer edits are explicit overrides of the AI output, saved
  via `PATCH`.
- **Detail view**: saved complaint, AI assessment, and an audit-trail timeline
  (`complaint_status_history`).

Redux slices: `complaintsSlice` (list/detail via RTK Query), `intakeSlice` (SSE
connection state, live step data). A small custom hook wraps `EventSource` and
dispatches actions per event.

## Error handling

- LLM calls: Groq JSON mode + `tenacity` retry with backoff; a failed node is marked
  `error` in the trace and the graph degrades gracefully (continues where it can)
  rather than hard-failing the whole intake.
- File parsing failures (corrupt PDF, unsupported type): clear 4xx response, no
  partial DB writes.
- SSE disconnects: processing continues and persists server-side regardless of
  client connection state, so a refreshed complaint list always reflects the true
  outcome even if the live view was interrupted.
- Consistent `{error: {code, message}}` envelope via FastAPI exception handlers.

## Testing

Backend-weighted, since that's what the assignment evaluates:
- pytest per LangGraph node (mocked Groq client), one full-graph integration test
  covering both the normal path and the duplicate short-circuit path.
- API endpoint tests via `httpx`/`TestClient` (intake happy path, duplicate path,
  malformed file upload).
- A handful of frontend RTL tests (intake form, AI Copilot stepper with a mocked
  `EventSource`) — not exhaustive, time-budgeted.

## Deliverables / docs

- README with a Mermaid architecture diagram, one-command `docker-compose up` setup,
  `.env.example` (`GROQ_API_KEY`, DB URL), and an explicit **rubric-mapping section**
  tying every mandatory requirement and bonus feature to its file/line — doubles as
  the script for the required explanation video.
- Seed script generating realistic demo complaints, including one deliberately
  planted near-duplicate pair so duplicate detection has something to demonstrate.
- Documented "fast-follow, not built" section: auth/RBAC, PDF export.

## Explicitly out of scope for this build

- Production-grade OCR/document parsing (assignment explicitly says not required).
- Auth/RBAC (documented as fast-follow to protect the one-day budget).
- PDF export of the investigation report (documented as fast-follow).
- Multi-tenant support.
