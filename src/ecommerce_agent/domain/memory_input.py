from pydantic import BaseModel

class MemoryInput(BaseModel):
  memory: dict[str, str]