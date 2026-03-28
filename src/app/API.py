from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.app.database.qdrant_client import bootstrap_collections
from src.app.components.bm25_index import get_bm25_index
from src.app.common.settings import settings
from src.app.common.logger import get_logger
from src.app.v1.routers import ingestion,query,collection,health_check

logger = get_logger("API")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ────────────────────────────────────────────────────────────
    logger.info("Starting %s...", settings.app_name)

    # Bootstrap Qdrant collections (idempotent)
    try:
        bootstrap_collections()
    except Exception as exc:
        logger.error("Qdrant bootstrap failed: %s", exc)

    # Load BM25 index from disk if available
    bm25 = get_bm25_index()
    try:
        bm25.load()
        logger.info("BM25 index loaded: %d documents", len(bm25.chunk_ids))
    except Exception:
        logger.warning("BM25 index not found on disk — run /ingest first.")

    logger.info("%s ready.", settings.app_name)
    yield

    # ── Shutdown ───────────────────────────────────────────────────────────
    logger.info("Shutting down %s.", settings.app_name)


app = FastAPI(
        title="NASA Handbook QA System",
        description=(
            "Technical QA system for the NASA Systems Engineering Handbook "
            "(SP-2016-6105 Rev2). Powered by Qdrant + LangGraph + GPT-4o."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
PREFIX = "/api/v1"
app.include_router(ingestion.router,prefix=PREFIX)
app.include_router(query.router,prefix=PREFIX)
app.include_router(collection.router,prefix=PREFIX)
app.include_router(health_check.router,prefix=PREFIX)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "DocNav is Live"}