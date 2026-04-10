from pydantic import BaseModel, Field
from typing import Optional

class SearchQuery(BaseModel):
    query: str
    document_id: Optional[str] = None
    top_k: int = Field(default=5)

class EmbedQuery(BaseModel):
    text: str
