"""
Application configuration using pydantic-settings.

Loads environment variables from a .env file at the project root.
Fails fast on startup if required variables are missing.
Exposes a single cached `get_settings()` accessor for the entire app.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration – one source of truth for all env vars."""

    # --- Required (app will refuse to start if absent) ---
    DATABASE_URL: str  # PostgreSQL connection string
    SUPABASE_JWT_SECRET: str  # Used to verify Supabase-issued JWTs

    # --- Optional (sensible default provided) ---
    ENVIRONMENT: str = "development"  # "development" | "staging" | "production"

    # --- ML risk thresholds (tunable via env vars) ---
    HIGH_RISK_THRESHOLD: float = 0.7
    MEDIUM_RISK_THRESHOLD: float = 0.4

    # Load from .env at the project root; silently ignore any extra vars.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (created once, reused everywhere)."""
    return Settings()
