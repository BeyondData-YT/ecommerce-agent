from pydantic import BaseModel, Field

class TextRetrieverInput(BaseModel):
  """
  Input schema for the Retriever Services.
  """
  query: str = Field(description="The query to retrieve data from the database")
  top_k: int = Field(description="The maximum number of data to retrieve", default=3)
  
class ImageRetrieverInput(BaseModel):
  """
  Input schema for the Retriever Services.
  """
  image_url: str = Field(description="The image url to retrieve data from the database")
  top_k: int = Field(description="The maximum number of data to retrieve", default=3)
  
