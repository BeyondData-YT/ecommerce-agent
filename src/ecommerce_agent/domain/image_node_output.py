from pydantic import BaseModel

class ImageResponse(BaseModel):
  image_url: str
  caption: str
  product_name: str

class ImageResponseList(BaseModel):
  image_responses: list[ImageResponse]