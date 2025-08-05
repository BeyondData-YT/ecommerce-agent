from pydantic import BaseModel
from typing import Optional, List

class Product(BaseModel):
  id: Optional[int] = None
  code: Optional[str] = None
  name: Optional[str] = None
  description: Optional[str] = None
  embedding: Optional[List[float]] = None
  price: Optional[float] = None
  stock_level: Optional[int] = None
  image_url: Optional[str] = None
  is_active: bool
  text_rank: Optional[float] = None
  semantic_distance: Optional[float] = None
  rrf_score: Optional[float] = None