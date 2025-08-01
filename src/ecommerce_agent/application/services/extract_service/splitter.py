from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from ecommerce_agent.domain.document import Document as DocumentDomain
from ecommerce_agent.config import settings
import logging

class SplitterService:
    """
    Service for splitting documents into smaller chunks and adding window context metadata.
    """
    def __init__(self, small_chunk_size: int = None, small_chunk_overlap: int = None, 
                window_size: int = None, window_overlap: int = None):
        """
        Initializes the SplitterService with parameters for small chunks and window context.

        Args:
            small_chunk_size (int, optional): The size of the small chunks for search. Defaults to settings.SMALL_CHUNK_SIZE.
            small_chunk_overlap (int, optional): The overlap between small chunks. Defaults to settings.SMALL_CHUNK_OVERLAP.
            window_size (int, optional): The size of the context window for retrieval. Defaults to settings.WINDOW_SIZE.
            window_overlap (int, optional): The overlap for the context window. Defaults to settings.WINDOW_OVERLAP.
        """
        
        # Sizes for "small chunks" (for searching)
        self.small_chunk_size = small_chunk_size if small_chunk_size is not None else settings.SMALL_CHUNK_SIZE
        self.small_chunk_overlap = small_chunk_overlap if small_chunk_overlap is not None else settings.SMALL_CHUNK_OVERLAP

        # Sizes for context "windows" (for retrieval)
        self.window_size = window_size if window_size is not None else settings.WINDOW_SIZE
        self.window_overlap = window_overlap if window_overlap is not None else settings.WINDOW_OVERLAP
        
        # Common separators for text
        self.separators = ["\n\n", "\n", ". ", "? ", "! ", " "]
        
        # Initialize the splitter for small chunks
        self._set_small_chunk_splitter()
        
        # Initialize an auxiliary splitter to calculate context windows
        self._set_window_splitter()

    def _set_small_chunk_splitter(self) -> None:
        """
        Configures the splitter for small chunks used for search.
        """
        self.small_chunk_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.small_chunk_size, 
            chunk_overlap=self.small_chunk_overlap, 
            separators=self.separators
        )

    def _set_window_splitter(self) -> None:
        """
        Configures an auxiliary splitter to obtain context windows.
        """
        # We use a chunk_overlap of 0 for windows to avoid excessive duplication,
        # since the window itself is already a context around a central point.
        # The chunk_size will be the desired total window size.
        self.window_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.window_size,
            chunk_overlap=self.window_overlap,
            separators=self.separators
        )
    
    def _get_document_text(self, documents: list[Document]) -> str:
        """
        Extracts and concatenates the text content from a list of LangChain Document objects.

        Args:
            documents (list[Document]): A list of LangChain Document objects.

        Returns:
            str: A single string containing the concatenated text content of all documents.
        """
        return "\n\n".join([doc.page_content for doc in documents])
    
    def create_document(self, document: Document) -> DocumentDomain:
        """
        Creates a domain-specific Document object from a LangChain Document object.

        Args:
            document (Document): A LangChain Document object.

        Returns:
            DocumentDomain: A domain-specific Document object with content, window_content, and source.
        """
        return DocumentDomain(
            content=document.page_content,
            window_content=document.metadata.get("window_content"),
            source=document.metadata.get("source")
        )

    def split_documents(self, documents: list[Document]) -> list[DocumentDomain]:
        """
        Splits documents into "small chunks" and adds window context metadata to each chunk.

        Args:
            documents (list[Document]): A list of LangChain Document objects to be split.

        Returns:
            list[DocumentDomain]: A list of domain-specific Document objects, each representing
                                a small chunk with its associated window content and source.
        """
        all_small_chunks = []
        full_text = self._get_document_text(documents)
        logging.info(f"Starting document splitting for {len(documents)} documents.")
        # 1. Generate "small chunks"
        small_chunks = self.small_chunk_splitter.create_documents([full_text], metadatas=[{"source": doc.metadata["source"]} for doc in documents])
        logging.info(f"Generated {len(small_chunks)} small chunks.")
        # 2. For each small chunk, calculate its context window
        for _, small_chunk in enumerate(small_chunks):
            # Get the content of the small chunk
            small_chunk_content = small_chunk.page_content
            
            # Find the index of the small chunk in the full text
            start_index = full_text.find(small_chunk_content)
            
            if start_index != -1:
                end_index = start_index + len(small_chunk_content)

                # Calculate the range for the window backward and forward
                window_start = max(0, start_index - (self.window_size - self.small_chunk_size) // 2)
                window_end = min(len(full_text), end_index + (self.window_size - self.small_chunk_size) // 2)
                
                # Ensure the window is at least the size of the small_chunk
                if (window_end - window_start) < self.small_chunk_size:
                    window_start = max(0, end_index - self.window_size)
                    window_end = min(len(full_text), start_index + self.window_size)

                # Extract the window content
                window_content = full_text[window_start:window_end]

                # Update the metadata of the small chunk
                small_chunk.metadata["window_content"] = window_content
                small_chunk.metadata["original_document_id"] = small_chunk.metadata.get("source") # Or the actual original document ID
                
                all_small_chunks.append(self.create_document(small_chunk))
            else:
                # If the small chunk is not found in the full text, add it as is
                # This can happen if there are transformations in the content
                logging.warning(f"Small chunk content not found in full text. Adding as is. Chunk: {small_chunk.page_content[:50]}...")
                all_small_chunks.append(self.create_document(small_chunk))
        logging.info(f"Document splitting completed. Total small chunks with window context: {len(all_small_chunks)}.")
        return all_small_chunks