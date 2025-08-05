from ecommerce_agent.domain.document import Document
from ecommerce_agent.application.services.document_service import DocumentService
from ecommerce_agent.application.services.rag.embeddings import EmbeddingsService
import logging

class DocumentRetrieverService:
  """
  Service for retrieving documents from the knowledge base using various search strategies.
  It leverages embedding and document services for semantic, text, and hybrid searches.
  """
  def __init__(self):
    """
    Initializes the RetrieverService with instances of EmbeddingsService and DocumentService.
    """
    self.embeddings_service = EmbeddingsService()
    self.document_service = DocumentService()
    
  def retrieve_similar_documents(self, query: str, top_k: int = 5) -> list[Document]:
    """
    Retrieves documents based on semantic similarity to the given query.

    Args:
      query (str): The query string for semantic search.
      top_k (int): The maximum number of similar documents to retrieve. Defaults to 5.

    Returns:
      list[Document]: A list of Document objects semantically similar to the query.
    """
    logging.info(f"Generating embedding for semantic search query: '{query}'.")
    query_embedding = self.embeddings_service.embed_query(query)
    logging.info(f"Retrieving {top_k} similar documents semantically.")
    return self.document_service.retrieve_similar_documents(query_embedding, top_k)
  
  def retrieve_text_search_documents(self, query: str, top_k: int = 5) -> list[Document]:
    """
    Retrieves documents based on text similarity to the given query.

    Args:
      query (str): The query string for text search.
      top_k (int): The maximum number of text-similar documents to retrieve. Defaults to 5.

    Returns:
      list[Document]: A list of Document objects text-similar to the query.
    """
    logging.info(f"Initiating text search for query: '{query}'.")
    logging.info(f"Retrieving {top_k} documents via text search.")
    return self.document_service.retrieve_text_search_documents(query, top_k)
  
  def retrieve_hybrid_documents(self, query: str, top_k: int = 5) -> list[Document]:
    """
    Retrieves documents using a hybrid search approach (semantic + text) and merges results.

    Args:
      query (str): The query string for hybrid search.
      top_k (int): The maximum number of hybrid documents to retrieve. Defaults to 5.

    Returns:
      list[Document]: A list of Document objects from the hybrid search.
    """
    logging.info(f"Generating embedding for hybrid search query: '{query}'.")
    query_embedding = self.embeddings_service.embed_query(query)
    logging.info(f"Retrieving {top_k} hybrid documents.")
    return self.document_service.retrieve_hybrid_documents(query_embedding, query, top_k)