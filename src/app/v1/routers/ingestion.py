from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks

from src.app.v1.models.ingestion import IngestionStatusResponse
from src.app.components.ingestion_pipeline import run_ingestion, IngestionResult
from src.app.common.settings import settings
from src.app.common.logger import get_logger
from src.app.common.exceptions import IngestionError

logger=get_logger()


router=APIRouter()

# In-memory job store (replace with Redis for production)
_jobs: dict[str, IngestionResult] = {}

def _ingest_background(job_id: str, pdf_path: Path) -> None:
    """
    Background task — runs in uvicorn's thread pool via BackgroundTasks.
    run_ingestion() handles the async event loop internally via nest_asyncio.
    We do NOT clean up the pdf_path here — ingestion_pipeline.run_ingestion()
    deletes it in its finally block.
    """
    job = _jobs[job_id]
    job.status = "running"
    try:
        result = run_ingestion(pdf_path)
        job.status       = result.status
        job.total_pages  = result.total_pages
        job.total_chunks = result.total_chunks
        job.total_images = result.total_images
        job.total_tables = result.total_tables
        job.error        = result.error
        logger.info(
            "Job %s complete: %d pages | %d chunks | %d images | %d tables",
            job_id[:8], job.total_pages, job.total_chunks,
            job.total_images, job.total_tables,
        )
    except IngestionError as exc:
        job.status = "error"
        job.error  = str(exc)
        logger.error("Job %s failed: %s", job_id[:8], exc)
    except Exception as exc:
        job.status = "error"
        job.error  = f"Unexpected error: {exc}"
        logger.exception("Job %s unexpected failure", job_id[:8])


@router.post(
    "/ingest",
    response_model=IngestionStatusResponse,
    status_code=202,
    tags=["ingestion"],
)
async def ingest_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to ingest"),
) -> IngestionStatusResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    job_id = str(uuid.uuid4())

    data_dir = Path(settings.ingestion_data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = data_dir / f"upload_{job_id}.pdf"
    content  = await file.read()
    tmp_path.write_bytes(content)

    result = IngestionResult(status="pending")
    _jobs[job_id] = result
    background_tasks.add_task(_ingest_background, job_id, tmp_path)

    logger.info("Ingestion job %s queued for %s", job_id, file.filename)
    return IngestionStatusResponse(job_id=job_id, status="pending")


@router.get(
    "/ingest/{job_id}",
    response_model=IngestionStatusResponse,
    tags=["ingestion"],
)
def ingest_status(job_id: str) -> IngestionStatusResponse:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return IngestionStatusResponse(
        job_id=job_id,
        status=job.status,          # type: ignore[arg-type]
        total_pages=job.total_pages,
        total_chunks=job.total_chunks,
        total_images=job.total_images,
        total_tables=job.total_tables,
        error=job.error,
    )
