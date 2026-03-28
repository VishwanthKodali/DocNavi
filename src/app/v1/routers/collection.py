from __future__ import annotations
from fastapi import APIRouter,HTTPException
from src.app.v1.models.collection import CollectionsResponse, CollectionInfo
from src.app.database.qdrant_client import get_qdrant_client


router=APIRouter()

@router.get("/collections", response_model=CollectionsResponse, tags=["system"])
def list_collections() -> CollectionsResponse:
    try:
        client = get_qdrant_client()
        cols = client.get_collections().collections
        result: list[CollectionInfo] = []
        for col in cols:
            info = client.get_collection(col.name)
            result.append(CollectionInfo(
                name=col.name,
                vectors_count=info.vectors_count or 0,
                status=str(info.status),
            ))
        return CollectionsResponse(collections=result)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Qdrant unavailable: {exc}")