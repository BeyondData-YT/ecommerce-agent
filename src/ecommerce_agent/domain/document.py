from pydantic import BaseModel
from typing import List, Optional

class Document(BaseModel):
    id: Optional[int] = None
    content: Optional[str] = None
    embedding: Optional[List[float]] = None
    window_content: Optional[str] = None 
    source: Optional[str] = None 
    text_rank: Optional[float] = None
    semantic_distance: Optional[float] = None
    rrf_score: Optional[float] = None