import anthropic
from .base import LLMClient, LLMMessage, LLMResponse


class AnthropicClient(LLMClient):
    def __init__(self, api_key: str):
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def complete(self, *, system, messages, model, max_tokens, temperature=0.3) -> LLMResponse:
        response = await self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        return LLMResponse(
            text=response.content[0].text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
        )

    def model_for_task(self, task: str) -> str:
        return {
            "parse": "claude-haiku-4-5-20251001",
            "plan":  "claude-sonnet-4-6",
        }.get(task, "claude-sonnet-4-20250514")
