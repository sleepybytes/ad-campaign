import logging
import json
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.routes.campaign import router as campaign_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    if settings.llm_provider == "anthropic" and not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    if settings.llm_provider == "openai" and not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    data = Path(settings.data_dir)
    for fname in ("publishers.json", "personas.json"):
        if not (data / fname).exists():
            raise RuntimeError(f"Required data file missing: {data / fname}")

    logger.info(f"Startup complete. provider={settings.llm_provider}")
    yield
    logger.info("Shutdown.")


app = FastAPI(
    title="Ad Campaign Planner",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(campaign_router)


@app.get("/health", tags=["infra"])
async def health():
    settings = get_settings()
    return {"status": "ok", "provider": settings.llm_provider}
