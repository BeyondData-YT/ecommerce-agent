from sentence_transformers import SentenceTransformer
from ecommerce_agent.config import settings
import logging
from typing import Union
import asyncio
import platform

if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class EmbeddingsService:
  """
  Service for generating text embeddings using a pre-trained SentenceTransformer model.
  """
  _instance = None
  _lock = asyncio.Lock()
  def __init__(self):
    """
    Initializes the EmbeddingsService by loading the SentenceTransformer model.
    """
    self.model = None
    
  @classmethod
  async def get_instance(cls):
      """
      Asynchronously gets the single instance of the EmbeddingsService.
      Loads the model if it hasn't been loaded yet.
      """
      async with cls._lock:
        if cls._instance is None:
          cls._instance = cls()
          await cls._instance._load_model()
        return cls._instance
    
  async def _load_model(self):
    """
    Loads the SentenceTransformer model asynchronously.
    """
    logging.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
    self.model = SentenceTransformer(settings.EMBEDDING_MODEL, trust_remote_code=True)
    logging.info(f"Embedding model loaded: {settings.EMBEDDING_MODEL}")
    
  def embed_text(self, text: str) -> list[float]:
    """
    Generates an embedding for a given text string.

    Args:
      text (str): The input text string to embed.

    Returns:
      list[float]: A list of floats representing the embedding vector for the text.
    """
    return self.model.encode(text).tolist()
  
  def embed_image(self, image_url: str) -> list[float]:
    """
    Generates an embedding for a given image.

    Args:
      image_url (str): The input image url to embed.

    Returns:
      list[float]: A list of floats representing the embedding vector for the image.
    """
    return self.model.encode(image_url).tolist()
  
  def embed_documents(self, documents: list[str]) -> list[list[float]]:
    """
    Generates embeddings for a list of text documents.

    Args:
      documents (list[str]): A list of text strings, where each string is a document.

    Returns:
      list[list[float]]: A list of embedding vectors, one for each document.
    """
    return [self.embed_text(doc) for doc in documents]
  
  def embed_images(self, images: list[Union[str, bytes]]) -> list[list[float]]:
    """
    Generates embeddings for a list of images.

    Args:
      images (list[Union[str, bytes]]): A list of images to embed.

    Returns:
      list[list[float]]: A list of embedding vectors, one for each image.
    """
    return [self.embed_image(image) for image in images]
  
  def embed_query(self, query: str) -> list[float]:
    """
    Generates an embedding for a given query string.

    Args:
      query (str): The input query string to embed.

    Returns:
      list[float]: A list of floats representing the embedding vector for the query.
    """
    return self.embed_text(query)
