from pydantic import BaseModel
from typing import Literal

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    qdrant: bool
    bm25_ready: bool
    version: str = "0.1.0"