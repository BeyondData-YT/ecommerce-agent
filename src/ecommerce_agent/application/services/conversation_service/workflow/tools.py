from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from ecommerce_agent.application.services.rag.document_retriever import DocumentRetrieverService
from ecommerce_agent.application.services.rag.product_retriever import ProductRetrieverService
from ecommerce_agent.domain.retriever_input import RetrieverInput
from ecommerce_agent.domain.document import Document
from ecommerce_agent.domain.product import Product
import logging

class DocumentRetrieverTool(BaseTool):
  """
  A tool for retrieving relevant documents from the knowledge base based on a query.

  This tool uses a hybrid search approach (semantic and text-based) to find
  documents that best match the user's query.
  """
  name:str = "document_retriever"
  description:str = "Retrieve documents from the knowledge base"
  args_schema:ArgsSchema = RetrieverInput
  return_direct:bool = True
  
  def _format_docs(self, docs: list[Document]) -> str:
    """
    Formats a list of Document objects into a single string.

    Args:
      docs (list[Document]): A list of Document objects to format.

    Returns:
      str: A single string containing the window content of all documents, separated by double newlines.
    """
    return "\n\n".join([doc.window_content for doc in docs])
  
  def _run(self, query: str, top_k: int = 5) -> str:
    """
    Retrieves documents from the database based on a query.
    
    Args:
      query (str): The query string to retrieve documents.
      top_k (int): The maximum number of documents to retrieve. Defaults to 5.
      
    Returns:
      str: A formatted string containing the content of the retrieved documents.
    """
    logging.info(f"Initiating document retrieval with query: '{query}'.")
    docs = DocumentRetrieverService().retrieve_hybrid_documents(query, top_k)
    logging.info(f"Document retrieval completed. Found {len(docs)} documents.") 
    return self._format_docs(docs)

class ProductRetrieverTool(BaseTool):
  """
  A tool for retrieving relevant products from the knowledge base based on a query.

  This tool uses a hybrid search approach (semantic and text-based) to find
  products that best match the user's query.
  """
  name:str = "product_retriever"
  description:str = "Retrieve products from the database"
  args_schema:ArgsSchema = RetrieverInput
  return_direct:bool = True
  
  def _format_products(self, products: list[Product]) -> str:
    """
    Formats a list of Product objects into a single string.
    """
    return "\n\n".join([(product.name + " - " + product.description) for product in products])

  def _run(self, query: str, top_k: int = 5) -> str:
    """
    Retrieves products from the database based on a query.
    
    Args:
      query (str): The query string to retrieve products.
      top_k (int): The maximum number of products to retrieve. Defaults to 5.
    """
    logging.info(f"Initiating product retrieval with query: '{query}'.")
    products = ProductRetrieverService().retrieve_hybrid_products(query, top_k)
    logging.info(f"Product retrieval completed. Found {len(products)} products.")
    return self._format_products(products)

tools = [DocumentRetrieverTool(), ProductRetrieverTool()]

