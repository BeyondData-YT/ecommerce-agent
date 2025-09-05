from pydantic import BaseModel, Field

class MemoryInput(BaseModel):
  memory: dict[str, str] = Field(description="The information to store in the memory. Must be a dictionary with the key as the name of the information and the value as the value of the information.")