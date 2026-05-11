import asyncio
import json
import re
import logging
from datetime import date, datetime, timedelta, timezone
from providers.base import LLMClient, LLMMessage
from prompts import PUBLISHER_SCORER_SYSTEM, CREATIVE_WRITER_SYSTEM
from app.config import Settings

logger = logging.getLogger(__name__)


def _strip_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())


def _validate_budget_pct(plan: dict) -> None:
    """Raise ValueError if recommended publisher budget_pct values don't sum to 100."""
    recommended = plan.get("publishers", {}).get("recommended", [])
    if not recommended:
        return
    total = sum(p.get("recommended_budget_pct", 0) for p in recommended)
    if total != 100:
        raise ValueError(
            f"budget_pct values sum to {total}, expected 100. "
            "The planner returned an invalid allocation."
        )


async def _score_publishers(
    payload: str,
    client: LLMClient,
    model: str,
    max_tokens: int,
    trace_id: str,
    max_retries: int,
) -> dict:
    for attempt in range(max_retries + 1):
        user_content = payload if attempt == 0 else (
            f"{payload}\n\nIMPORTANT: Your previous response was not valid JSON. "
            "Return ONLY the raw JSON object matching the schema exactly."
        )
        response = await client.complete(
            system=PUBLISHER_SCORER_SYSTEM,
            messages=[LLMMessage(role="user", content=user_content)],
            model=model,
            max_tokens=max_tokens,
            temperature=0.3,
        )
        logger.info("scorer_call", extra={
            "trace_id": trace_id, "attempt": attempt,
            "input_tokens": response.input_tokens, "output_tokens": response.output_tokens,
        })
        try:
            result = json.loads(_strip_fences(response.text))
            _validate_budget_pct(result)
            return result
        except (json.JSONDecodeError, ValueError) as exc:
            if attempt == max_retries:
                raise ValueError(
                    f"Scorer failed after {max_retries + 1} attempts: {exc}. "
                    f"trace_id={trace_id} last_response={response.text[:300]}"
                )


async def _write_creatives(
    payload: str,
    client: LLMClient,
    model: str,
    max_tokens: int,
    trace_id: str,
    max_retries: int,
) -> list[dict]:
    for attempt in range(max_retries + 1):
        user_content = payload if attempt == 0 else (
            f"{payload}\n\nIMPORTANT: Your previous response was not valid JSON. "
            "Return ONLY the raw JSON object matching the schema exactly."
        )
        response = await client.complete(
            system=CREATIVE_WRITER_SYSTEM,
            messages=[LLMMessage(role="user", content=user_content)],
            model=model,
            max_tokens=max_tokens,
            temperature=0.4,
        )
        logger.info("creative_call", extra={
            "trace_id": trace_id, "attempt": attempt,
            "input_tokens": response.input_tokens, "output_tokens": response.output_tokens,
        })
        try:
            result = json.loads(_strip_fences(response.text))
            return result.get("creatives", [])
        except json.JSONDecodeError as exc:
            if attempt == max_retries:
                raise ValueError(
                    f"Creative writer failed after {max_retries + 1} attempts: {exc}. "
                    f"trace_id={trace_id} last_response={response.text[:300]}"
                )


def _assemble_campaign_config(
    enriched_brief: dict,
    recommended_publishers: list[dict],
    creatives: list[dict],
    publishers_by_id: dict,
) -> dict:
    today = date.today()
    start = today + timedelta(days=1)
    end = start + timedelta(days=30)

    business_model = enriched_brief.get("business_model", [])
    if "subscription" in business_model:
        strategy, opt_goal = "CPA", "subscription_start"
    elif "one-time" in business_model:
        strategy, opt_goal = "ROAS", "purchase"
    else:
        strategy, opt_goal = "CPM", "awareness"

    cpm_floor_map = {"high": 15.0, "mid-high": 10.0, "mid": 6.5, "low": 4.0}
    income_tier = (enriched_brief.get("target_audience", {}) or {}).get("income_tier") or "mid"
    cpm_floor = cpm_floor_map.get(income_tier, 6.5)

    allocations = []
    for pub in recommended_publishers:
        pub_data = publishers_by_id.get(pub["publisher_id"], {})
        monthly_imp = pub_data.get("monthly_impressions", 0)
        pct = pub["recommended_budget_pct"]
        pub_income = (pub_data.get("audience", {}) or {}).get("income_tier", "mid")
        allocations.append({
            "publisher_id": pub["publisher_id"],
            "publisher_name": pub["publisher_name"],
            "budget_pct": pct,
            "estimated_impressions": int(monthly_imp * pct / 100),
            "cpm_floor_usd": cpm_floor_map.get(pub_income, 6.5),
        })

    creative_ids = [c["variant_id"] for c in creatives]
    creative_assignments = [
        {
            "publisher_id": pub["publisher_id"],
            "variant_ids": creative_ids,
            "rotation": "even",
            "rationale": f"All {len(creative_ids)} variant(s) rotated evenly.",
        }
        for pub in recommended_publishers
    ]

    audience = enriched_brief.get("target_audience", {}) or {}
    age_str = audience.get("age_range") or "18-65"
    try:
        lo, hi = age_str.split("-")
        age_min, age_max = int(lo), int(hi)
    except Exception:
        age_min, age_max = 18, 65

    gender_skew = audience.get("gender_skew")
    if gender_skew == "female":
        genders = ["female"]
    elif gender_skew == "male":
        genders = ["male"]
    else:
        genders = ["female", "male"]

    category = enriched_brief.get("category", "campaign").replace("_", " ").title()

    return {
        "meta": {
            "campaign_name": f"{category} Campaign",
            "status": "draft",
            "created_at": f"{today.isoformat()}T00:00:00Z",
            "flight": {
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "duration_days": 30,
            },
        },
        "budget": {
            "total_usd": None,
            "currency": "USD",
            "pacing": "even",
            "allocations": allocations,
        },
        "targeting": {
            "age_range": {"min": age_min, "max": age_max},
            "gender": genders,
            "income_tier": [income_tier],
            "geos": ["nationwide"],
            "device": ["mobile", "desktop"],
            "time_of_day": None,
        },
        "bidding": {
            "strategy": strategy,
            "optimization_goal": opt_goal,
            "cpa_target_usd": None,
            "roas_target": None,
            "cpm_floor_usd": cpm_floor,
            "cpm_ceiling_usd": round(cpm_floor * 1.5, 2),
        },
        "creative_assignments": creative_assignments,
        "frequency_cap": {"impressions_per_user": 3, "window_hours": 24},
        "brand_safety": {
            "content_exclusions": ["adult", "gambling", "political"],
            "publisher_allowlist_only": True,
        },
    }


async def generate_plan(
    enriched_brief: dict,
    publishers: list[dict],
    personas: list[dict],
    client: LLMClient,
    settings: Settings,
    trace_id: str = "",
    max_retries: int = 2,
) -> dict:
    model = settings.planner_model or client.model_for_task("plan")
    publishers_by_id = {p["id"]: p for p in publishers}

    scorer_payload = json.dumps({
        "advertiser_brief": enriched_brief,
        "publishers": publishers,
        "personas": personas,
    })
    creative_payload = json.dumps({
        "advertiser_brief": enriched_brief,
        "personas": personas,
    })

    scorer_result, creatives = await asyncio.gather(
        _score_publishers(scorer_payload, client, model, settings.scorer_max_tokens, trace_id, max_retries),
        _write_creatives(creative_payload, client, model, settings.creative_max_tokens, trace_id, max_retries),
    )

    # Keep only creatives whose persona was also selected by the scorer
    selected_ids = {p["persona_id"] for p in scorer_result.get("personas", {}).get("selected", [])}
    filtered_creatives = [c for c in creatives if c.get("persona_id") in selected_ids] or creatives

    # Renumber variant_ids sequentially after filtering
    for i, creative in enumerate(filtered_creatives, 1):
        creative["variant_id"] = f"creative_{i:03d}"

    recommended = scorer_result.get("publishers", {}).get("recommended", [])
    campaign_config = _assemble_campaign_config(
        enriched_brief, recommended, filtered_creatives, publishers_by_id
    )

    return {
        "plan_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "advertiser_summary": scorer_result.get("advertiser_summary", ""),
        "confidence": scorer_result.get("confidence", "medium"),
        "no_match_reason": scorer_result.get("no_match_reason"),
        "publishers": scorer_result.get("publishers", {}),
        "personas": scorer_result.get("personas", {}),
        "creatives": filtered_creatives,
        "campaign_config": campaign_config,
    }
