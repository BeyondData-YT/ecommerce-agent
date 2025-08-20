from pydantic import BaseModel
from typing import Tuple

class MemoryInput(BaseModel):
  memory: dict[str, str]
  namespace_for_memory: Tuple[str, str]