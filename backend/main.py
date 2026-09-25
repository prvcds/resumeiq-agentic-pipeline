"""ResumeIQ — FastAPI application entry point.

Routes:
    GET  /                  Serve the frontend (redirect to static HTML)
    GET  /health            Health check
    GET  /candidate         Return the loaded candidate data
    POST /qa                Natural-language Q&A over the candidate
    POST /evaluate          Run the full agentic evaluation pipeline
    GET  /download/{name}   Download a generated evaluation file
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.agent_workflow import run_evaluation
from backend.models import Candidate, QARequest, QAResponse
from backend.parsing import compute_timeline, load_candidate
from backend.qa_engine import ask

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-24s %(levelname)-5s %(message)s",
)
logger = logging.getLogger("resumeiq.main")

app = FastAPI(
    title="ResumeIQ",
    description="AI-powered resume Q&A and agentic evaluation system",
    version="1.0.0",
)

# CORS — permissive for local development (file:// origin + localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend static files
_FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if _FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="frontend")

# Output directory for downloads
_OUTPUT_DIR = Path(__file__).parent.parent / "output"

# ---------------------------------------------------------------------------
# Singleton candidate (loaded once at startup)
# ---------------------------------------------------------------------------

_candidate: Candidate | None = None


def _get_candidate() -> Candidate:
    global _candidate
    if _candidate is None:
        _candidate = load_candidate()
        logger.info("Loaded candidate: %s", _candidate.full_name)
    return _candidate


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Redirect to the frontend page."""
    return FileResponse(str(_FRONTEND_DIR / "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok", "llm_provider": os.environ.get("LLM_PROVIDER", "auto")}


@app.get("/candidate")
async def get_candidate():
    """Return the loaded candidate data and pre-computed timeline."""
    candidate = _get_candidate()
    timeline = compute_timeline(candidate)
    return {
        "candidate": candidate.model_dump(),
        "timeline": timeline.model_dump(),
    }


@app.post("/qa", response_model=QAResponse)
async def qa_endpoint(request: QARequest):
    """Answer a natural-language question about the candidate."""
    candidate = _get_candidate()
    try:
        response = ask(candidate, request.question)
        return response
    except Exception as exc:
        logger.exception("Q&A failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/evaluate")
async def evaluate_endpoint(
    export_format: str = Query(default="json", regex="^(json|pdf)$"),
    send_email: bool = Query(default=True),
):
    """Run the full agentic evaluation pipeline and return results."""
    candidate = _get_candidate()
    try:
        result = run_evaluation(
            candidate,
            export_format=export_format,
            send_email=send_email,
        )
        return result
    except Exception as exc:
        logger.exception("Evaluation pipeline failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download a generated evaluation file from the output directory."""
    path = _OUTPUT_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    return FileResponse(
        str(path),
        filename=filename,
        media_type="application/octet-stream",
    )


@app.get("/output-files")
async def list_output_files():
    """List all generated output files."""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = [
        f.name for f in _OUTPUT_DIR.iterdir()
        if f.is_file() and f.name != ".gitkeep"
    ]
    return {"files": sorted(files, reverse=True)}
