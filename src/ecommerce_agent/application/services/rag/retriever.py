from ecommerce_agent.domain.document import Document
from ecommerce_agent.application.services.document_service import DocumentService
from ecommerce_agent.application.services.rag.embeddings import EmbeddingsService
import logging

class RetrieverService:
  def __init__(self):
    self.embeddings_service = EmbeddingsService()
    self.document_service = DocumentService()
    
  def retrieve_similar_documents(self, query: str, top_k: int = 5) -> list[Document]:
    logging.info("Generating query embedding")
    query_embedding = self.embeddings_service.embed_query(query)
    logging.info("Retrieving similar documents")
    return self.document_service.retrieve_similar_documents(query_embedding, top_k)
  
  def retrieve_text_search_documents(self, query: str, top_k: int = 5) -> list[Document]:
    logging.info("Generating query embedding")
    query_embedding = self.embeddings_service.embed_query(query)
    logging.info("Retrieving text search documents")
    return self.document_service.retrieve_text_search_documents(query, top_k)
  
  def retrieve_hybrid_documents(self, query: str, top_k: int = 5) -> list[Document]:
    logging.info("Generating query embedding")
    query_embedding = self.embeddings_service.embed_query(query)
    logging.info("Retrieving hybrid documents")
    return self.document_service.retrieve_hybrid_documents(query_embedding, query, top_k)