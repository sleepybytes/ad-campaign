from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    llm_provider: str = "anthropic"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    aws_region: str = "us-east-1"

    parser_model: str = "claude-haiku-4-5-20251001"
    planner_model: str = "claude-sonnet-4-6"
    parser_max_tokens: int = 800
    planner_max_tokens: int = 4000
    scorer_max_tokens: int = 3500
    creative_max_tokens: int = 1500

    data_dir: str = "data"

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
