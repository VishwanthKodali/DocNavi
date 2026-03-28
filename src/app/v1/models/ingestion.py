from pydantic import BaseModel
from typing import Literal

class IngestionStatusResponse(BaseModel):
    job_id: str
    status: Literal["pending", "running", "complete", "error"]
    total_pages: int = 0
    total_chunks: int = 0
    total_images: int = 0
    total_tables: int = 0
    error: str = ""