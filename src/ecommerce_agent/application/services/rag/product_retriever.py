from ecommerce_agent.domain.product import Product
from ecommerce_agent.application.services.products_service import ProductsService
from ecommerce_agent.application.services.rag.embeddings import EmbeddingsService
import logging

class ProductRetrieverService:
  """
  Service for retrieving products from the knowledge base using various search strategies.
  It leverages embedding and product services for semantic, text, and hybrid searches.
  """
  def __init__(self):
    """
    Initializes the RetrieverService with instances of EmbeddingsService and ProductsService.
    """
    self.embeddings_service = EmbeddingsService()
    self.products_service = ProductsService()
    
  def retrieve_similar_products(self, query: str, top_k: int = 5) -> list[Product]:
    """
    Retrieves products based on semantic similarity to the given query.

    Args:
      query (str): The query string for semantic search.
      top_k (int): The maximum number of similar products to retrieve. Defaults to 5.

    Returns:
      list[Product]: A list of Product objects semantically similar to the query.
    """
    logging.info(f"Generating embedding for semantic search query: '{query}'.")
    query_embedding = self.embeddings_service.embed_query(query)
    logging.info(f"Retrieving {top_k} similar products semantically.")
    return self.products_service.retrieve_similar_products(query_embedding, top_k)
  
  def retrieve_text_search_products(self, query: str, top_k: int = 5) -> list[Product]:
    """
    Retrieves products based on text similarity to the given query.

    Args:
      query (str): The query string for text search.
      top_k (int): The maximum number of text-similar products to retrieve. Defaults to 5.

    Returns:
      list[Product]: A list of Product objects text-similar to the query.
    """
    logging.info(f"Initiating text search for query: '{query}'.")
    logging.info(f"Retrieving {top_k} products via text search.")
    return self.products_service.retrieve_text_search_products(query, top_k)
  
  def retrieve_hybrid_products(self, query: str, top_k: int = 5) -> list[Product]:
    """
    Retrieves products using a hybrid search approach (semantic + text) and merges results.

    Args:
      query (str): The query string for hybrid search.
      top_k (int): The maximum number of hybrid products to retrieve. Defaults to 5.'
    Returns:
      list[Product]: A list of Product objects from the hybrid search.
    """
    logging.info(f"Generating embedding for hybrid search query: '{query}'.")
    query_embedding = self.embeddings_service.embed_query(query)
    logging.info(f"Retrieving {top_k} hybrid products.")
    return self.products_service.retrieve_hybrid_products(query_embedding, query, top_k)