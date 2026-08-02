from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.models import Base
from app.db.session import engine

app = FastAPI(title="AIVOA Complaint Management API")

from app.api.intake import router as intake_router

app.include_router(intake_router)

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
