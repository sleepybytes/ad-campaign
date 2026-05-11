import json
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch
from app.main import app
from providers.base import LLMResponse


@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_plan_b2b_brief():
    b2b_parsed = {
        "category": "saas", "product_summary": "B2B SaaS for dentists.",
        "target_audience": {}, "price_tier": None, "business_model": ["b2b"],
        "key_values": [], "channel_fit": {"is_b2b": True, "is_giftable": False,
        "is_subscription_friendly": True}, "confidence": "high", "ambiguity_notes": None,
    }
    with patch("app.routes.campaign.parse_brief", return_value=b2b_parsed):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            r = await ac.post("/campaign/plan", json={"brief": "B2B SaaS for dental practices."})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert "B2B" in body["error"]


@pytest.mark.asyncio
async def test_plan_request_too_short():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.post("/campaign/plan", json={"brief": "hi"})
    assert r.status_code == 422
