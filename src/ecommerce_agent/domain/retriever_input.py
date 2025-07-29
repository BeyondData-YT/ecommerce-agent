from pydantic import BaseModel, Field

class RetrieverInput(BaseModel):
  query: str = Field(description="The query to retrieve documents from the database")
  top_k: int = Field(description="The number of documents to retrieve", default=5)