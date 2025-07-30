from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from ecommerce_agent.application.services.rag.retriever import DocumentRetrieverService
from ecommerce_agent.domain.document_retriever_input import DocumentRetrieverInput
from ecommerce_agent.domain.document import Document
import logging

class DocumentRetrieverTool(BaseTool):
  """
  A tool for retrieving relevant documents from the knowledge base based on a query.

  This tool uses a hybrid search approach (semantic and text-based) to find
  documents that best match the user's query.
  """
  name:str = "document_retriever"
  description:str = "Retrieve documents from the knowledge base"
  args_schema:ArgsSchema = DocumentRetrieverInput
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
    logging.info(f"Retrieving documents with query: {query}")
    docs = DocumentRetrieverService().retrieve_hybrid_documents(query, top_k)
    logging.info("Documents retrieved") 
    logging.info(f"Documents retrieved: {len(docs)}")
    return self._format_docs(docs)


tools = [DocumentRetrieverTool()]

