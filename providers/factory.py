from app.config import Settings
from .base import LLMClient


def build_llm_client(settings: Settings) -> LLMClient:
    registry = {
        "anthropic": _build_anthropic,
        "openai":    _build_openai,
    }
    builder = registry.get(settings.llm_provider)
    if builder is None:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{settings.llm_provider}'. "
            f"Valid options: {list(registry)}"
        )
    return builder(settings)


def _build_anthropic(settings: Settings) -> LLMClient:
    from .anthropic_provider import AnthropicClient
    if not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY is required for the anthropic provider")
    return AnthropicClient(api_key=settings.anthropic_api_key)


def _build_openai(settings: Settings) -> LLMClient:
    from .openai_provider import OpenAIClient
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for the openai provider")
    return OpenAIClient(api_key=settings.openai_api_key)
