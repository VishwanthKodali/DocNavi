from pydantic import BaseModel, Field

class CollectionInfo(BaseModel):
    name: str
    vectors_count: int
    status: str


class CollectionsResponse(BaseModel):
    collections: list[CollectionInfo]