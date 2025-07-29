import argparse
from ecommerce_agent.domain.document import Document
from ecommerce_agent.application.services.document_service import DocumentService
from ecommerce_agent.application.services.extract_service.extract import ExtractService
from ecommerce_agent.application.services.rag.embeddings import EmbeddingsService
from ecommerce_agent.config import settings

parser = argparse.ArgumentParser(description='Ingest documents into the database')
parser.add_argument('--directory', type=str, help='Directory to ingest documents from', default=settings.DATA_FAQS_DIR)
args = parser.parse_args()

class IngestDocumentsTable:
  def __init__(self):
    self.document_service = DocumentService()
    self.extract_service = ExtractService()
    self.embeddings_service = EmbeddingsService()
    
  def add_embedding(self, document: Document):
    embedding = self.embeddings_service.embed_text(document.content)
    document.embedding = embedding
    return document

  def ingest_documents_table(self, directory: str):
    documents = self.extract_service.extract_documents(directory)
    for document in documents:
      document = self.add_embedding(document)
      self.document_service.create_document(document)

ingest_documents_table = IngestDocumentsTable()
ingest_documents_table.ingest_documents_table(args.directory)