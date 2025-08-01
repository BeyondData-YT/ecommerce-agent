from pathlib import Path
from ecommerce_agent.application.services.extract_service.loader import LoaderService
from ecommerce_agent.application.services.extract_service.splitter import SplitterService
from ecommerce_agent.domain.document import Document
import logging

class ExtractService:
  """
  Service for extracting documents from a given directory.
  It utilizes LoaderService to load documents and SplitterService to split them into chunks.
  """
  def __init__(self):
    self.loader_service = LoaderService()
    self.splitter_service = SplitterService()
    
  def extract_documents(self, directory: Path) -> list[Document]:
    """
    Loads documents from a specified directory and splits them into processed chunks.

    Args:
      directory (Path): The path to the directory containing the documents.

    Returns:
      list[Document]: A list of Document objects, each representing a processed chunk with metadata.
    """
    logging.info(f"Starting document extraction from directory: {directory}")
    documents = self.loader_service.load_documents(directory)
    logging.info(f"Successfully loaded {len(documents)} documents.")
    chunks = self.splitter_service.split_documents(documents)
    logging.info(f"Successfully split documents into {len(chunks)} chunks.")
    return chunks
  
  