from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from ecommerce_agent.application.services.rag.retriever import RetrieverService
from ecommerce_agent.domain.retriever_input import RetrieverInput
from ecommerce_agent.domain.document import Document
import logging

class RetrieverTool(BaseTool):
  name:str = "retriever"
  description:str = "Retrieve documents from the knowledge base"
  args_schema:ArgsSchema = RetrieverInput
  return_direct:bool = True
  
  def _format_docs(self, docs: list[Document]) -> str:
    return "\n\n".join([doc.window_content for doc in docs])
  
  def _run(self, query: str, top_k: int = 5) -> list[Document]:
    """
    Retrieve documents from the database
    
    Args:
      query: The query to retrieve documents from the database
      top_k: The number of documents to retrieve
      
    Returns:
      A list of documents
    """
    logging.info(f"Retrieving documents with query: {query}")
    docs = RetrieverService().retrieve_hybrid_documents(query, top_k)
    logging.info("Documents retrieved") 
    logging.info(f"Documents retrieved: {len(docs)}")
    return self._format_docs(docs)


tools = [RetrieverTool()]

