"""FastAPI backend.

Run:  uvicorn src.api:app --reload
Then: http://127.0.0.1:8000
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import DEFAULT_STRATEGY, DISTANCE_THRESHOLD, STRATEGIES, TOP_K
from .qa import answer

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"

app = FastAPI(title="Document Q&A with Citations")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    strategy: str = DEFAULT_STRATEGY
    k: int = TOP_K
    threshold: float | None = None


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "strategies": list(STRATEGIES),
        "default_strategy": DEFAULT_STRATEGY,
        "threshold": DISTANCE_THRESHOLD,
        "k": TOP_K,
    }


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    if request.strategy not in STRATEGIES:
        raise HTTPException(400, f"strategy must be one of {list(STRATEGIES)}")
    try:
        return answer(
            request.question,
            strategy=request.strategy,
            k=request.k,
            threshold=request.threshold,
        )
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc


if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(FRONTEND / "index.html")