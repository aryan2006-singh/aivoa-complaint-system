import pytest
from sqlalchemy import select, text

from app.db.models import Complaint
from app.db.session import SessionLocal
from app.main import app


@pytest.mark.asyncio
async def test_startup_creates_extension_and_tables() -> None:
    """Drives the real ASGI startup hook (which `AsyncClient` +
    `ASGITransport` never triggers) against the live Postgres container.

    This exercises: the `CREATE EXTENSION IF NOT EXISTS vector` statement,
    `Base.metadata.create_all`, and — by issuing an ORM query afterwards —
    SQLAlchemy mapper configuration, which fails loudly if any relationship
    (e.g. `Complaint.assessment` / `AIAssessment.complaint_id` vs.
    `duplicate_of_id`) is ambiguous.
    """
    await app.router.startup()

    async with SessionLocal() as session:
        # The startup hook must have created the pgvector extension.
        ext_result = await session.execute(
            text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        )
        assert ext_result.scalar() == 1

        # All four tables must exist and be queryable via raw SQL.
        for table_name in (
            "complaints",
            "complaint_attachments",
            "ai_assessments",
            "complaint_status_history",
        ):
            count_result = await session.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            assert count_result.scalar() is not None

        # Issue a real ORM query. This forces SQLAlchemy to configure all
        # mappers in the registry, surfacing AmbiguousForeignKeysError (or
        # any other mapper-configuration error) if it's present. We only
        # assert that the query executes and returns a list — the table's
        # contents are not this test's concern, since other tests legitimately
        # write rows to the shared live database.
        orm_result = await session.execute(select(Complaint).limit(1))
        assert isinstance(orm_result.scalars().all(), list)
