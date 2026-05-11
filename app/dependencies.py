from functools import lru_cache
from app.config import get_settings
from providers.base import LLMClient
from providers.factory import build_llm_client


@lru_cache
def get_llm_client() -> LLMClient:
    return build_llm_client(get_settings())
