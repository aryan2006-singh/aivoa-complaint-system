import os

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/complaints"
)
