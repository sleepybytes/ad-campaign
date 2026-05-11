import json
import pytest
from pipeline.parser import parse_brief
from providers.base import LLMResponse
from app.config import Settings


@pytest.fixture
def settings():
    return Settings(
        llm_provider="anthropic",
        anthropic_api_key="test",
        parser_model="mock-model",
        parser_max_tokens=800,
    )


@pytest.mark.asyncio
async def test_parse_brief_happy_path(mock_llm_client, settings):
    expected = {
        "category": "pet_food",
        "product_summary": "Senior dog food.",
        "target_audience": {"age_range": "30-55", "gender_skew": "balanced",
                            "income_tier": "mid-high", "life_stage": None},
        "price_tier": "premium",
        "business_model": ["subscription"],
        "key_values": ["vet-formulated"],
        "channel_fit": {"is_b2b": False, "is_giftable": False,
                        "is_subscription_friendly": True},
        "confidence": "high",
        "ambiguity_notes": None,
    }
    mock_llm_client.complete.return_value = LLMResponse(
        text=json.dumps(expected), input_tokens=100, output_tokens=200, model="mock"
    )
    result = await parse_brief("Senior dog food brief", mock_llm_client, settings)
    assert result["category"] == "pet_food"
    assert result["channel_fit"]["is_b2b"] is False


@pytest.mark.asyncio
async def test_parse_brief_retries_on_invalid_json(mock_llm_client, settings):
    valid = json.dumps({"category": "pet_food", "product_summary": "x",
                        "target_audience": {}, "price_tier": None,
                        "business_model": [], "key_values": [],
                        "channel_fit": {"is_b2b": False, "is_giftable": False,
                                        "is_subscription_friendly": False},
                        "confidence": "low", "ambiguity_notes": None})
    mock_llm_client.complete.side_effect = [
        LLMResponse(text="not json :(", input_tokens=10, output_tokens=5, model="mock"),
        LLMResponse(text=valid, input_tokens=10, output_tokens=50, model="mock"),
    ]
    result = await parse_brief("some brief", mock_llm_client, settings)
    assert mock_llm_client.complete.call_count == 2
    assert result["category"] == "pet_food"


@pytest.mark.asyncio
async def test_parse_brief_raises_after_max_retries(mock_llm_client, settings):
    mock_llm_client.complete.return_value = LLMResponse(
        text="still not json", input_tokens=10, output_tokens=5, model="mock"
    )
    with pytest.raises(ValueError, match="invalid JSON"):
        await parse_brief("bad brief", mock_llm_client, settings, max_retries=1)


@pytest.mark.asyncio
async def test_parse_brief_strips_markdown_fences(mock_llm_client, settings):
    payload = {"category": "beauty", "product_summary": "x",
               "target_audience": {}, "price_tier": None, "business_model": [],
               "key_values": [], "channel_fit": {"is_b2b": False, "is_giftable": True,
               "is_subscription_friendly": False}, "confidence": "high", "ambiguity_notes": None}
    mock_llm_client.complete.return_value = LLMResponse(
        text=f"```json\n{json.dumps(payload)}\n```",
        input_tokens=10, output_tokens=30, model="mock",
    )
    result = await parse_brief("beauty brief", mock_llm_client, settings)
    assert result["category"] == "beauty"
