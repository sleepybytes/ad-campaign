import pytest
from unittest.mock import AsyncMock
from providers.base import LLMClient, LLMResponse


@pytest.fixture
def mock_llm_client():
    client = AsyncMock(spec=LLMClient)
    client.model_for_task.return_value = "mock-model"
    return client


@pytest.fixture
def sample_enriched_brief():
    return {
        "category": "pet_food",
        "product_summary": "Premium grain-free dog food for senior dogs.",
        "target_audience": {
            "age_range": "30-55",
            "gender_skew": "balanced",
            "income_tier": "mid-high",
            "life_stage": None,
        },
        "price_tier": "premium",
        "business_model": ["subscription", "dtc"],
        "key_values": ["vet-formulated", "grain-free", "senior health"],
        "channel_fit": {
            "is_b2b": False,
            "is_giftable": False,
            "is_subscription_friendly": True,
        },
        "confidence": "high",
        "ambiguity_notes": None,
    }
