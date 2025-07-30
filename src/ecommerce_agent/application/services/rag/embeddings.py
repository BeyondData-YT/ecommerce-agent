from sentence_transformers import SentenceTransformer
from ecommerce_agent.config import settings
import logging

class EmbeddingsService:
  """
  Service for generating text embeddings using a pre-trained SentenceTransformer model.
  """
  def __init__(self):
    """
    Initializes the EmbeddingsService by loading the SentenceTransformer model.
    """
    self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
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
  
  def embed_documents(self, documents: list[str]) -> list[list[float]]:
    """
    Generates embeddings for a list of text documents.

    Args:
      documents (list[str]): A list of text strings, where each string is a document.

    Returns:
      list[list[float]]: A list of embedding vectors, one for each document.
    """
    return [self.embed_text(doc) for doc in documents]
  
  def embed_query(self, query: str) -> list[float]:
    """
    Generates an embedding for a given query string.

    Args:
      query (str): The input query string to embed.

    Returns:
      list[float]: A list of floats representing the embedding vector for the query.
    """
    return self.embed_text(query)
