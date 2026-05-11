# Ad Campaign Planner

A FastAPI service that takes a free-text advertiser description and returns a ranked publisher list, persona-targeted ad creative variants, and a structured campaign config — ready for an ad system to ingest.

---

## How to run it

**With Docker (recommended):**
```bash
cp .env.example .env        # add your ANTHROPIC_API_KEY
docker compose up --build
```

**Locally:**
```bash
python3 -m venv .venv   
source .venv/bin/activate   
pip install -r requirements.txt
cp .env.example .env        # add your ANTHROPIC_API_KEY
uvicorn app.main:app --reload
```

**Run tests:**
```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -v
```

### API

`POST /campaign/plan` — takes a free-text brief, returns a full plan:
```bash
curl -X POST http://localhost:8000/campaign/plan \
  -H "Content-Type: application/json" \
  -d '{"brief": "Premium dog food for senior dogs. Grain-free, vet-formulated, subscription-based."}'
```

`GET /health` — returns `{"status": "ok", "provider": "anthropic"}`.

**To switch LLM provider:** set `LLM_PROVIDER=openai` and `OPENAI_API_KEY=sk-...` in `.env`. No code changes.

---

## What I built

Two LLM calls run in sequence behind a single endpoint:

1. **Parser** (fast/cheap model — Haiku): extracts structured attributes from the raw brief (category, audience, price tier, channel fit, business model). Retries once with a repair prompt on invalid JSON.

2. **Planner** (quality model — Sonnet): scores all publishers on five weighted dimensions, selects 2–4 matching personas, writes persona-specific ad creatives with varied sentence structure, and assembles a full campaign config including budget allocations, targeting, bidding strategy, frequency caps, and creative assignments.

The LLM provider is fully swappable via environment variable — the pipeline only talks to the `LLMClient` interface. Adding a new provider (Bedrock, Gemini) means one new file in `providers/` and one dict entry in `factory.py`.

---

## What I would do next (one more week)

**Catalog retrieval at scale.** Right now the full publisher and persona catalogs go into every prompt. At 20 publishers that's fine; at 2,000 it breaks context windows and burns tokens. The fix is embedding-based retrieval: embed each brief and each publisher profile, retrieve the top-N by cosine similarity, and send only those to the planner. This is the first thing I'd ship.

**Structured output validation.** The planner response is passed through as a raw `dict`. I'd add full Pydantic models for the nested plan — that catches LLM schema drift immediately instead of silently propagating bad data to downstream systems.

**SSE streaming endpoint.** The planner call takes 10–20 seconds. A `/plan/stream` endpoint using server-sent events would let the frontend show incremental progress (parse result first, then publisher scores, then creatives). The `LLMClient` base class already has a placeholder for a `stream()` method.

**Prompt templates with few-shot examples.** Publisher scores cluster at 7/10 without examples to anchor calibration. I'd move prompts to Jinja2 templates and inject 3–5 annotated brief→score pairs per advertiser category. The annotations become a dataset that improves over time.

**Prompt versioning and A/B testing.** Right now prompts are flat text files with no history and no way to know if a change made things better or worse. I'd version prompts explicitly, run two variants in shadow mode against the same brief, and score outputs with an LLM judge. Without this, any prompt edit is just a guess.

**Feedback loop.** Connect actual campaign performance (CTR, CPA, ROAS) back to the scoring model. Even a simple re-ranker trained on 500 examples would outperform the zero-shot LLM scorer on in-distribution advertisers.

---

## What I cut

- **Auth** — no API key middleware. Add it in `app/main.py` as a dependency on `APIRouter`.
- **Persistent storage** — plans are returned, not saved. A Postgres table with `(trace_id, brief, plan_json, created_at)` is the obvious v2 addition.
- **Bedrock provider** — same interface as Anthropic, just different init. Skipped to keep scope tight.
- **UI** — JSON API only. Any frontend wires directly to `/campaign/plan`.
- **Startup catalog caching** — publishers and personas are read from disk on every request. In production, load once into `app.state` in the lifespan handler and inject via dependency.

---

## What's actually hard here

**Easy:** the API plumbing, the provider abstraction, the retry logic, wiring FastAPI. These are solved problems with well-known patterns. 

**Genuinely hard:**

*Prompt calibration without ground truth.* "Score this publisher 0–10" is underspecified. Without annotated examples the model anchors around 7 for everything that isn't an obvious mismatch. Getting calibrated scores that reflect real campaign performance requires a feedback dataset you don't have at launch — you have to ship something suboptimal and close the loop.

*Ambiguous briefs.* "We help people feel better" has no right answer. The current system makes its best guess and flags `confidence: low`. The better behavior is to ask a clarifying question — but that requires a multi-turn interaction model the single-shot API doesn't support. Deciding where the boundary is between "parse it and flag uncertainty" and "ask before proceeding" is a product question with real UX consequences.

*Scale and retrieval quality.* Embedding-based retrieval solves the context-window problem but introduces a new one: embedding similarity doesn't capture category nuance well. A beauty brand and a wellness brand may be close in embedding space but need completely different publisher mixes. The interesting engineering is in the retrieval layer — hybrid BM25 + dense retrieval, re-ranking, or a dedicated scoring model — not in the LLM call itself.

*The evaluation problem.* This is the thing I keep coming back to. Without a labeled golden set for publisher ranking, an LLM-as-judge setup for creative quality, and online signals from actual campaign performance (CTR, CPA, conversion rate by publisher and persona), you have no idea whether the system is good or getting worse. Building that feedback loop is where the interesting engineering actually lives — and none of it exists yet.
