"""Settings read from the environment, shared by the API, the runner and the agent."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """One object for every knob the harness reads at start-up."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = Field(
        default="postgresql+psycopg://harness:harness@localhost:5432/harness",
        description="SQLAlchemy URL of the PostgreSQL database that holds runs and results.",
    )
    openai_api_key: str = Field(default="", description="Key sent to the model endpoint.")
    openai_base_url: str | None = Field(
        default=None,
        description="Base URL of an OpenAI-compatible endpoint; empty means the default.",
    )
    model: str = Field(default="gpt-4o-mini", description="Model under test.")
    judge_model: str = Field(default="gpt-4o-mini", description="Model used as the judge.")
    prompt_version: str = Field(default="v1", description="Label stored on each run.")
    pass_threshold: float = Field(
        default=0.75, ge=0.0, le=1.0, description="Score a task must reach to count as passed."
    )
    max_steps: int = Field(default=6, ge=1, le=20, description="Tool-calling turns allowed per task.")
    request_timeout: float = Field(default=30.0, gt=0, description="Seconds before a model call fails.")
    suite_path: str = Field(default="data/suite.json", description="Task suite loaded by the runner.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, built once from the environment."""

    return Settings()
