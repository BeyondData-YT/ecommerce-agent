from pydantic import BaseModel
from typing import Optional, List

class Product(BaseModel):
  id: Optional[int] = None
  code: Optional[str] = None
  name: Optional[str] = None
  description: Optional[str] = None
  embedding: Optional[List[float]] = None
  price: Optional[float] = None
  image_url: Optional[str] = None
  is_active: bool