from __future__ import annotations
from fastapi import APIRouter

from src.app.v1.models.health_check import HealthResponse
from src.app.database.qdrant_client import get_qdrant_client
from src.app.components.bm25_index import get_bm25_index


router=APIRouter()

@router.get("/health", response_model=HealthResponse, tags=["system"])
def health_check() -> HealthResponse:
    qdrant_ok = False
    try:
        get_qdrant_client().get_collections()
        qdrant_ok = True
    except Exception:
        pass

    bm25_ready = get_bm25_index().is_ready()

    status = "ok" if (qdrant_ok and bm25_ready) else "degraded"
    return HealthResponse(
        status=status,
        qdrant=qdrant_ok,
        bm25_ready=bm25_ready,
    )