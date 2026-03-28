from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str = Field(...,description="Natural language question")


class CitationModel(BaseModel):
    section_id: str
    page_num: int | str
    confidence: float
    source_type: str = "text"


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[CitationModel] = []
    intent: str = ""
    confidence_score: float = 0.0
    expanded_query: str = ""