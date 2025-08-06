from pydantic import BaseModel, Field

class RetrieverInput(BaseModel):
  """
  Input schema for the Retriever Services.
  """
  query: str = Field(description="The query to retrieve data from the database")
  top_k: int = Field(description="The maximum number of data to retrieve", default=5)