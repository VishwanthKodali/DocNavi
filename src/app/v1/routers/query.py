from __future__ import annotations
from fastapi import APIRouter, HTTPException

from src.app.v1.models.query import QueryRequest, QueryResponse,CitationModel
from src.app.components.query_pipeline import run_query
from src.app.common.logger import get_logger

router=APIRouter()

logger=get_logger()

@router.post("/query", response_model=QueryResponse, tags=["query"])
def query_handbook(body: QueryRequest) -> QueryResponse:
    try:
        state = run_query(body.query)
    except Exception as exc:
        logger.exception("Query pipeline failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")

    citations = [
        CitationModel(
            section_id=c.get("section_id", "?"),
            page_num=c.get("page_num", "?"),
            confidence=c.get("confidence", 0.0),
            source_type=c.get("source_type", "text"),
        )
        for c in state.get("citations", [])
    ]

    return QueryResponse(
        query=body.query,
        answer=state.get("answer", ""),
        citations=citations,
        intent=state.get("intent", ""),
        confidence_score=state.get("confidence_score", 0.0),
        expanded_query=state.get("expanded_query", ""),
    )
