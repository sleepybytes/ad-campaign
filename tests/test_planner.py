import json
import pytest
from pipeline.planner import generate_plan, _validate_budget_pct
from providers.base import LLMResponse
from app.config import Settings


@pytest.fixture
def settings():
    return Settings(
        llm_provider="anthropic",
        anthropic_api_key="test",
        planner_model="mock-model",
        planner_max_tokens=4000,
    )


@pytest.fixture
def minimal_plan():
    return {
        "plan_version": "1.0",
        "generated_at": "2026-01-01T00:00:00Z",
        "advertiser_summary": "Premium senior dog food.",
        "confidence": "high",
        "no_match_reason": None,
        "publishers": {
            "recommended": [
                {"publisher_id": "pub_009", "publisher_name": "Ruffco",
                 "overall_score": 8.5,
                 "score_breakdown": {"category_relevance": 10, "audience_age_fit": 7,
                                     "audience_gender_fit": 6, "income_fit": 7,
                                     "aov_compatibility": 8},
                 "fit_rationale": "Largest pet audience.", "recommended_budget_pct": 100}
            ],
            "excluded": []
        },
        "personas": {
            "selected": [{"persona_id": "persona_004", "persona_name": "The Pet Parent",
                          "plausibility_rationale": "Reads ingredient labels."}],
            "excluded": []
        },
        "creatives": [
            {"variant_id": "creative_001", "persona_id": "persona_004",
             "persona_name": "The Pet Parent", "headline": "Science-backed nutrition for senior dogs.",
             "body_copy": "Vet-formulated. Grain-free. Delivered monthly.",
             "persona_rationale": "Vet endorsement resonates."}
        ],
        "campaign_config": {
            "meta": {"campaign_name": "test", "status": "draft",
                     "created_at": "2026-01-01T00:00:00Z",
                     "flight": {"start_date": "2026-01-02", "end_date": "2026-02-01",
                                "duration_days": 30}},
            "budget": {"total_usd": None, "currency": "USD", "pacing": "even",
                       "allocations": [{"publisher_id": "pub_009", "publisher_name": "Ruffco",
                                        "budget_pct": 100, "estimated_impressions": 500000,
                                        "cpm_floor_usd": 8.0}]},
            "targeting": {"age_range": {"min": 30, "max": 55}, "gender": ["female", "male"],
                          "income_tier": ["mid-high"], "geos": ["nationwide"],
                          "device": ["mobile", "desktop"], "time_of_day": None},
            "bidding": {"strategy": "CPA", "optimization_goal": "subscription_start",
                        "cpa_target_usd": None, "roas_target": None,
                        "cpm_floor_usd": 8.0, "cpm_ceiling_usd": 12.0},
            "creative_assignments": [{"publisher_id": "pub_009",
                                      "variant_ids": ["creative_001"],
                                      "rotation": "even", "rationale": "Only variant."}],
            "frequency_cap": {"impressions_per_user": 3, "window_hours": 24},
            "brand_safety": {"content_exclusions": ["adult", "gambling", "political"],
                             "publisher_allowlist_only": True}
        }
    }


def test_validate_budget_pct_valid(minimal_plan):
    _validate_budget_pct(minimal_plan)  # should not raise


def test_validate_budget_pct_invalid():
    plan = {"publishers": {"recommended": [
        {"recommended_budget_pct": 60},
        {"recommended_budget_pct": 30},
    ]}}
    with pytest.raises(ValueError, match="sum to 90"):
        _validate_budget_pct(plan)


def test_validate_budget_pct_no_recommended():
    _validate_budget_pct({"publishers": {"recommended": []}})  # no raise


async def test_generate_plan_happy_path(mock_llm_client, settings, minimal_plan,
                                        sample_enriched_brief):
    scorer_result = {
        "advertiser_summary": "Premium senior dog food.",
        "confidence": "high",
        "no_match_reason": None,
        "publishers": minimal_plan["publishers"],
        "personas": minimal_plan["personas"],
    }
    creative_result = {"creatives": minimal_plan["creatives"]}

    mock_llm_client.complete.side_effect = [
        LLMResponse(text=json.dumps(scorer_result), input_tokens=500, output_tokens=500, model="mock"),
        LLMResponse(text=json.dumps(creative_result), input_tokens=300, output_tokens=200, model="mock"),
    ]
    result = await generate_plan(sample_enriched_brief, [], [], mock_llm_client, settings)
    assert result["confidence"] == "high"
    assert result["publishers"]["recommended"][0]["publisher_id"] == "pub_009"
    assert len(result["creatives"]) == 1
