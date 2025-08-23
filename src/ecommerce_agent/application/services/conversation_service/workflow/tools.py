from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from langchain_core.runnables import RunnableConfig
from ecommerce_agent.application.services.rag.document_retriever import DocumentRetrieverService
from ecommerce_agent.application.services.rag.product_retriever import ProductRetrieverService  
from ecommerce_agent.application.services.memory import MemoryService
from ecommerce_agent.domain.retriever_input import TextRetrieverInput, ImageRetrieverInput
from ecommerce_agent.domain.document import Document
from ecommerce_agent.domain.product import Product
from ecommerce_agent.domain.memory_input import MemoryInput
import logging
import asyncio

class DocumentRetrieverTool(BaseTool):
  """
  A tool for retrieving relevant documents from the knowledge base based on a query.

  This tool uses a hybrid search approach (semantic and text-based) to find
  documents that best match the user's query.
  """
  name:str = "document_retriever"
  description:str = "Retrieve documents from the knowledge base"
  args_schema:ArgsSchema = TextRetrieverInput
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
  
  def _run(self, query: str, top_k: int = 3) -> str:
    """
    Retrieves documents from the database based on a query.
    """
    return asyncio.run(self._arun(query, top_k))
  
  async def _arun(self, query: str, top_k: int = 3) -> str:
    """
    Retrieves documents from the database based on a query.
    
    Args:
      query (str): The query string to retrieve documents.
      top_k (int): The maximum number of documents to retrieve. Defaults to 3.
      
    Returns:
      str: A formatted string containing the content of the retrieved documents.
    """
    logging.info(f"Initiating document retrieval with query: '{query}'.")
    docs = await DocumentRetrieverService().retrieve_hybrid_documents(query, top_k)
    logging.info(f"Document retrieval completed. Found {len(docs)} documents.") 
    return self._format_docs(docs)

class TextProductRetrieverTool(BaseTool):
  """
  A tool for retrieving relevant products from the knowledge base based on a text query.

  This tool uses a hybrid search approach (semantic and text-based) to find
  products that best match the user's query.
  """
  name:str = "text_product_retriever"
  description:str = "Retrieve products from the database based on a text query"
  args_schema:ArgsSchema = TextRetrieverInput
  return_direct:bool = True
  
  def _format_products(self, products: list[Product]) -> str:
    """
    Formats a list of Product objects into a single string.
    """
    return "\n\n".join([(product.name + " - " + product.description) for product in products])

  def _run(self, query: str, top_k: int = 3) -> str:
    """
    Retrieves products from the database based on a query.
    """
    return asyncio.run(self._arun(query, top_k))

  async def _arun(self, query: str, top_k: int = 3) -> str:
    """
    Retrieves products from the database based on a query.
    
    Args:
      query (str): The query string to retrieve products.
      top_k (int): The maximum number of products to retrieve. Defaults to 3.
    """
    logging.info(f"Initiating product retrieval with query: '{query}'.")
    products = await ProductRetrieverService().retrieve_hybrid_products(query, top_k)
    logging.info(f"Product retrieval completed. Found {len(products)} products.")
    return self._format_products(products)

class ImageProductRetrieverTool(BaseTool):
  """
  A tool for retrieving relevant products from the knowledge base based on an image query.
  """
  name:str = "image_product_retriever"
  description:str = "Retrieve products from the database based on an image query"
  args_schema:ArgsSchema = ImageRetrieverInput
  return_direct:bool = True
  
  def _format_products(self, products: list[Product]) -> str:
    """
    Formats a list of Product objects into a single string.
    """
    return "\n\n".join([(product.name + " -- " + product.description + " -- " + product.image_url) for product in products])
  
  def _run(self, image_url: str, top_k: int = 3) -> str:
    """
    Retrieves products from the database based on an image query.
    """
    return asyncio.run(self._arun(image_url, top_k))

  async def _arun(self, image_url: str, top_k: int = 3) -> str:
    """
    Retrieves products from the database based on an image query.
    """
    logging.info("Initiating product retrieval with image.")
    products = await ProductRetrieverService().retrieve_similar_image_products(image_url, top_k)
    logging.info(f"Product retrieval completed. Found {len(products)} products.")
    return self._format_products(products)

class MemoryTool(BaseTool):
  """
  A tool for storing and retrieving memories.
  """
  name:str = "store_memory"
  description:str = "Store memories in the database"
  args_schema:ArgsSchema = MemoryInput
  return_direct:bool = True
  
  def _run(self, memory: dict[str, str], config: RunnableConfig) -> str:
    """
    Stores memories in the database.
    """
    return asyncio.run(self._arun(memory, config))
  
  async def _arun(self, memory: dict[str, str], config: RunnableConfig) -> str:
    """
    Stores memories in the database.
    """
    await MemoryService().store_memory(memory, config)
    return "Memory stored successfully"

tools = [DocumentRetrieverTool(), TextProductRetrieverTool(), ImageProductRetrieverTool()]
memory_tools = [MemoryTool()]