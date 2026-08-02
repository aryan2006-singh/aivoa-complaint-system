from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str
    groq_model: str = "llama-3.1-8b-instant"
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/complaints"
    duplicate_window_days: int = 90
    duplicate_sim_threshold: float = 0.85
    duplicate_possible_threshold: float = 0.70

    @field_validator("database_url")
    @classmethod
    def _use_asyncpg_driver(cls, value: str) -> str:
        # Managed Postgres providers (Render, Heroku, Railway, ...) hand out a
        # bare `postgres://`/`postgresql://` connection string with no async
        # driver specified; SQLAlchemy's async engine needs `+asyncpg` or it
        # fails to resolve a dialect at startup.
        if value.startswith("postgres://"):
            return "postgresql+asyncpg://" + value[len("postgres://"):]
        if value.startswith("postgresql://"):
            return "postgresql+asyncpg://" + value[len("postgresql://"):]
        return value


settings = Settings()
