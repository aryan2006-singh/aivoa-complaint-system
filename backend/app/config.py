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
