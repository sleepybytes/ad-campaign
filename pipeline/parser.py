import json
import re
import logging
from providers.base import LLMClient, LLMMessage
from prompts import BRIEF_PARSER_SYSTEM
from app.config import Settings

logger = logging.getLogger(__name__)


def _strip_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())


async def parse_brief(
    brief: str,
    client: LLMClient,
    settings: Settings,
    trace_id: str = "",
    max_retries: int = 2,
) -> dict:
    model = settings.parser_model or client.model_for_task("parse")

    for attempt in range(max_retries + 1):
        if attempt == 0:
            user_content = brief
        else:
            user_content = (
                f"{brief}\n\n"
                "IMPORTANT: Your previous response was not valid JSON. "
                "Return ONLY a raw JSON object — no markdown, no explanation."
            )

        response = await client.complete(
            system=BRIEF_PARSER_SYSTEM,
            messages=[LLMMessage(role="user", content=user_content)],
            model=model,
            max_tokens=settings.parser_max_tokens,
            temperature=0.1,
        )
        logger.info(
            "parser_call",
            extra={
                "trace_id": trace_id,
                "attempt": attempt,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "model": response.model,
            },
        )

        try:
            return json.loads(_strip_fences(response.text))
        except json.JSONDecodeError:
            if attempt == max_retries:
                raise ValueError(
                    f"Parser returned invalid JSON after {max_retries + 1} attempts. "
                    f"trace_id={trace_id} last_response={response.text[:300]}"
                )
