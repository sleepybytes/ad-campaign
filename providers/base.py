from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMMessage:
    role: str      # "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


class LLMClient(ABC):
    @abstractmethod
    async def complete(
        self,
        *,
        system: str,
        messages: list[LLMMessage],
        model: str,
        max_tokens: int,
        temperature: float = 0.3,
    ) -> LLMResponse: ...

    @abstractmethod
    def model_for_task(self, task: str) -> str:
        """Map task name to provider-specific model string.
        task is one of: "parse", "plan"
        """
        ...
