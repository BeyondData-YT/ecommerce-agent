from ecommerce_agent.application.services.extract_service.loader import LoaderService
from ecommerce_agent.application.services.extract_service.splitter import SplitterService
from ecommerce_agent.domain.document import Document

class ExtractService:
  def __init__(self):
    self.loader_service = LoaderService()
    self.splitter_service = SplitterService()
    
  def extract_documents(self, directory: str) -> list[Document]:
    documents = self.loader_service.load_documents(directory)
    chunks = self.splitter_service.split_documents(documents)
    return chunks
  
  