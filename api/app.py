import logging
import os
import re
import sys
import time
from typing import List

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from semantic_matcher.retrieval import SemanticMatcher
from semantic_matcher.settings import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", settings.artifacts_dir)
MODEL_NAME = os.getenv("MODEL_NAME")
INDEX_VERSION = os.getenv("INDEX_VERSION", settings.index_version)

if not os.path.isabs(ARTIFACTS_DIR):
    ARTIFACTS_DIR = os.path.join(ROOT, ARTIFACTS_DIR)

app = FastAPI(title="NLP Semantic Matcher", version="1.0.0")

STATIC_DIR = os.path.join(ROOT, "api", "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

matcher = SemanticMatcher.from_artifacts(ARTIFACTS_DIR, model_name=MODEL_NAME, index_version=INDEX_VERSION)

Instrumentator().instrument(app).expose(app)


class MatchRequest(BaseModel):
    designation: str = Field(..., min_length=2)
    top_k: int = Field(5, ge=1, le=20)


class MatchItem(BaseModel):
    item_id: str
    official_name: str
    score: float
    confidence: float
    explanation: str


class MatchResponse(BaseModel):
    query: str
    results: List[MatchItem]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/match", response_model=MatchResponse)
async def match(request: MatchRequest) -> MatchResponse:
    start_time = time.time()
    query_length = len(request.designation)
    
    try:
        results = matcher.match(request.designation, top_k=request.top_k)
        
        mode = "unknown"
        if results:
            explanation = results[0].get("explanation", "")
            mode_match = re.search(r"mode=(\w+)", explanation)
            if mode_match:
                mode = mode_match.group(1)
        
        latency_ms = (time.time() - start_time) * 1000
        
        logger.info(
            f"match_request: query_length={query_length}, top_k={request.top_k}, mode={mode}, results_count={len(results)}, latency_ms={round(latency_ms, 2)}"
        )
        
        return MatchResponse(query=request.designation, results=results)
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error(
            f"match_error: query_length={query_length}, top_k={request.top_k}, error={str(e)}, latency_ms={round(latency_ms, 2)}",
            exc_info=True,
        )
        raise
