import json
import uuid
import logging
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from app.config import Settings, get_settings
from app.dependencies import get_llm_client
from app.schemas import PlanRequest, PlanResponse
from providers.base import LLMClient
from pipeline.parser import parse_brief
from pipeline.planner import generate_plan

router = APIRouter(prefix="/campaign", tags=["campaign"])
logger = logging.getLogger(__name__)


def _load_catalog(settings: Settings) -> tuple[list, list]:
    data = Path(settings.data_dir)
    publishers = json.loads((data / "publishers.json").read_text())
    personas   = json.loads((data / "personas.json").read_text())
    return publishers, personas


@router.post("/plan", response_model=PlanResponse)
async def create_plan(
    body: PlanRequest,
    settings: Settings  = Depends(get_settings),
    llm: LLMClient      = Depends(get_llm_client),
):
    trace_id = str(uuid.uuid4())
    logger.info("plan_request_start", extra={"trace_id": trace_id, "brief_length": len(body.brief)})

    publishers, personas = _load_catalog(settings)

    try:
        enriched = await parse_brief(body.brief, llm, settings, trace_id=trace_id)
    except ValueError as e:
        logger.error("parser_failed", extra={"trace_id": trace_id, "error": str(e)})
        raise HTTPException(status_code=422, detail=str(e))

    if enriched.get("channel_fit", {}).get("is_b2b"):
        return PlanResponse(
            ok=False,
            trace_id=trace_id,
            parse_result=enriched,
            error=(
                "This appears to be a B2B product. "
                "The publisher catalog covers consumer audiences only."
            ),
        )

    try:
        plan = await generate_plan(
            enriched, publishers, personas, llm, settings, trace_id=trace_id
        )
    except ValueError as e:
        logger.error("planner_failed", extra={"trace_id": trace_id, "error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))

    logger.info("plan_request_complete", extra={"trace_id": trace_id})
    return PlanResponse(ok=True, trace_id=trace_id, plan=plan, parse_result=enriched)
