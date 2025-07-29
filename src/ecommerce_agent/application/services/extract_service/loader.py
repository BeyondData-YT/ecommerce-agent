from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
import logging

class LoaderService:
  def __init__(self):
    pass
  
  def get_txt_loader(self, directory: str):
    loader = DirectoryLoader(
      path=directory,
      glob="**/*.txt",
      recursive=True,
      show_progress=True,
      use_multithreading=True,
      loader_cls=TextLoader,
      loader_kwargs={"autodetect_encoding": True}
    )
    return loader
  
  def get_pdf_loader(self, directory: str):
    loader = DirectoryLoader(
      path=directory,
      glob="**/*.pdf",
      recursive=True,
      show_progress=True,
      use_multithreading=True,
      loader_cls=PyPDFLoader
    )
    return loader
  
  def get_loaders(self, directory: str):
    txt_loader = self.get_txt_loader(directory)
    pdf_loader = self.get_pdf_loader(directory)
    return txt_loader, pdf_loader
    
  def load_documents(self, directory: str):
    documents = []
    txt_loader, pdf_loader = self.get_loaders(directory)
    logging.info("Loading documents")
    documents.extend(txt_loader.load())
    documents.extend(pdf_loader.load())
    logging.info("Documents loaded")
    logging.info(f"Loaded {len(documents)} documents from {directory}")
    return documents
  
  