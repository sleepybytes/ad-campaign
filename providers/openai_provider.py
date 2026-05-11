try:
    from openai import AsyncOpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

from .base import LLMClient, LLMMessage, LLMResponse


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str):
        if not _OPENAI_AVAILABLE:
            raise ImportError("pip install openai to use the OpenAI provider")
        self._client = AsyncOpenAI(api_key=api_key)

    async def complete(self, *, system, messages, model, max_tokens, temperature=0.3) -> LLMResponse:
        all_messages = [{"role": "system", "content": system}]
        all_messages += [{"role": m.role, "content": m.content} for m in messages]
        response = await self._client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=all_messages,
        )
        choice = response.choices[0]
        return LLMResponse(
            text=choice.message.content,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            model=response.model,
        )

    def model_for_task(self, task: str) -> str:
        return {"parse": "gpt-4o-mini", "plan": "gpt-4o"}.get(task, "gpt-4o")
