from sentence_transformers import SentenceTransformer
from ecommerce_agent.config import settings
import logging

class EmbeddingsService:
  def __init__(self):
    self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
    logging.info(f"Embedding model loaded: {settings.EMBEDDING_MODEL}")
    
  def embed_text(self, text: str) -> list[float]:
    return self.model.encode(text).tolist()
  
  def embed_documents(self, documents: list[str]) -> list[list[float]]:
    return [self.embed_text(doc) for doc in documents]
  
  def embed_query(self, query: str) -> list[float]:
    return self.embed_text(query)
