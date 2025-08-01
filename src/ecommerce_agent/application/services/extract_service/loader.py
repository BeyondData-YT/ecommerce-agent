from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
import logging
from langchain_core.documents import Document

class LoaderService:
  """
  Service for loading documents from a directory.
  """
  def __init__(self):
    """
    Initializes the LoaderService.
    """
    pass
  
  def get_txt_loader(self, directory: Path) -> DirectoryLoader:
    """
    Retrieves a DirectoryLoader configured for loading text files (.txt).

    Args:
      directory (Path): The path to the directory from which to load text files.

    Returns:
      DirectoryLoader: An initialized DirectoryLoader for text files.
    """
    logging.info(f"Configuring TXT loader for directory: {directory}")
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
  
  def get_pdf_loader(self, directory: Path) -> DirectoryLoader:
    """
    Retrieves a DirectoryLoader configured for loading PDF files (.pdf).

    Args:
      directory (Path): The path to the directory from which to load PDF files.

    Returns:
      DirectoryLoader: An initialized DirectoryLoader for PDF files.
    """
    logging.info(f"Configuring PDF loader for directory: {directory}")
    loader = DirectoryLoader(
      path=directory,
      glob="**/*.pdf",
      recursive=True,
      show_progress=True,
      use_multithreading=True,
      loader_cls=PyPDFLoader
    )
    return loader
  
  def get_loaders(self, directory: Path) -> tuple[DirectoryLoader, DirectoryLoader]:
    """
    Retrieves both text and PDF loaders for a given directory.

    Args:
      directory (Path): The path to the directory containing documents.

    Returns:
      tuple[DirectoryLoader, DirectoryLoader]: A tuple containing the text loader and the PDF loader.
    """
    logging.info(f"Getting all document loaders for directory: {directory}")
    txt_loader = self.get_txt_loader(directory)
    pdf_loader = self.get_pdf_loader(directory)
    return txt_loader, pdf_loader
    
  def load_documents(self, directory: Path) -> list[Document]:
    """
    Loads all supported documents (text and PDF) from a specified directory.

    Args:
      directory (Path): The path to the directory from which to load documents.

    Returns:
      list[Document]: A list of loaded LangChain Document objects.
    """
    documents = []
    txt_loader, pdf_loader = self.get_loaders(directory)
    logging.info(f"Loading documents from directory: {directory}")
    txt_documents = txt_loader.load()
    pdf_documents = pdf_loader.load()
    documents.extend(txt_documents)
    documents.extend(pdf_documents)
    logging.info(f"Finished loading. Loaded {len(documents)} documents in total.")
    return documents
  
  