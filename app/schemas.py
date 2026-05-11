from pydantic import BaseModel, Field


class PlanRequest(BaseModel):
    brief: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Free-text advertiser description",
        examples=[
            "Premium dog food for senior dogs. Grain-free, vet-formulated, subscription-based."
        ],
    )


class PlanResponse(BaseModel):
    ok: bool
    trace_id: str
    plan: dict | None = None
    parse_result: dict | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    provider: str
