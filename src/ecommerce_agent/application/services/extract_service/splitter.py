from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from ecommerce_agent.domain.document import Document as DocumentDomain
from ecommerce_agent.config import settings
import logging

class SplitterService:
    def __init__(self, small_chunk_size: int = None, small_chunk_overlap: int = None, 
                window_size: int = None, window_overlap: int = None):
        
        # Tamaños para los "small chunks" (para búsqueda)
        self.small_chunk_size = small_chunk_size if small_chunk_size is not None else settings.SMALL_CHUNK_SIZE
        self.small_chunk_overlap = small_chunk_overlap if small_chunk_overlap is not None else settings.SMALL_CHUNK_OVERLAP

        # Tamaños para las "ventanas" de contexto (para recuperación)
        self.window_size = window_size if window_size is not None else settings.WINDOW_SIZE
        self.window_overlap = window_overlap if window_overlap is not None else settings.WINDOW_OVERLAP
        
        # Separadores comunes para texto
        self.separators = ["\n\n", "\n", ". ", "? ", "! ", " "]
        
        # Inicializamos el splitter para los small chunks
        self._set_small_chunk_splitter()
        
        # Inicializamos un splitter auxiliar para calcular las ventanas de contexto
        self._set_window_splitter()

    def _set_small_chunk_splitter(self):
        """Configura el splitter para los chunks pequeños de búsqueda."""
        self.small_chunk_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.small_chunk_size, 
            chunk_overlap=self.small_chunk_overlap, 
            separators=self.separators
        )

    def _set_window_splitter(self):
        """Configura un splitter auxiliar para obtener las ventanas de contexto."""
        # Usamos un chunk_overlap de 0 para las ventanas para evitar duplicación excesiva,
        # ya que la ventana en sí ya es un contexto alrededor de un punto central.
        # El chunk_size será el tamaño total de la ventana deseado.
        self.window_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.window_size,
            chunk_overlap=self.window_overlap, # Puedes ajustar esto si quieres solapamiento en las ventanas
            separators=self.separators
        )
    
    def _get_document_text(self, documents: list[Document]) -> str:
        """Extrae y concatena el texto de una lista de documentos."""
        return "\n\n".join([doc.page_content for doc in documents])
      
    def create_document(self, document: Document) -> DocumentDomain:
        return DocumentDomain(
            content=document.page_content,
            window_content=document.metadata.get("window_content"),
            source=document.metadata.get("source")
        )

    def split_documents(self, documents: list[Document]) -> list[DocumentDomain]:
        """
        Divide los documentos en "small chunks" y agrega metadatos de ventana de contexto.
        """
        all_small_chunks = []
        full_text = self._get_document_text(documents)
        logging.info("Splitting documents")
        # 1. Generar los "small chunks"
        small_chunks = self.small_chunk_splitter.create_documents([full_text], metadatas=[{"source": doc.metadata["source"]} for doc in documents])
        logging.info("Small chunks generated")
        # 2. Para cada small chunk, calcular su ventana de contexto
        for i, small_chunk in enumerate(small_chunks):
            # Obtener el contenido del small chunk
            small_chunk_content = small_chunk.page_content
            
            # Encontrar el índice del small chunk en el texto completo
            # Esto puede ser un poco impreciso con solapamientos y separadores complejos.
            # Una forma más robusta sería manejar los índices directamente desde el splitter.
            # Por simplicidad, aquí buscaremos la primera ocurrencia.
            start_index = full_text.find(small_chunk_content)
            
            if start_index != -1:
                end_index = start_index + len(small_chunk_content)

                # Calcular el rango para la ventana hacia atrás y hacia adelante
                window_start = max(0, start_index - (self.window_size - self.small_chunk_size) // 2)
                window_end = min(len(full_text), end_index + (self.window_size - self.small_chunk_size) // 2)
                
                # Asegurarse de que la ventana sea al menos del tamaño del small_chunk
                if (window_end - window_start) < self.small_chunk_size:
                    window_start = max(0, end_index - self.window_size)
                    window_end = min(len(full_text), start_index + self.window_size)

                # Extraer el contenido de la ventana
                window_content = full_text[window_start:window_end]

                # Actualizar los metadatos del small chunk
                small_chunk.metadata["window_content"] = window_content
                small_chunk.metadata["original_document_id"] = small_chunk.metadata.get("source") # O el ID real del documento original
                
                all_small_chunks.append(self.create_document(small_chunk))
            else:
                # Si no se encuentra el small chunk en el texto completo, añadirlo tal cual
                # Esto puede ocurrir si hay transformaciones en el contenido
                all_small_chunks.append(self.create_document(small_chunk))
        logging.info("Small chunks split")

        # print(f"Split {len(documents)} documents into {len(all_small_chunks)} small chunks with window context.")
        # if all_small_chunks:
        #     print(f"Example Small Chunk:\n{all_small_chunks[0].content}")
        #     print(f"Example Window Content (Metadata):\n{all_small_chunks[0].window_content}")
        #     print(all_small_chunks[0].source)
        logging.info(f"Split {len(documents)} documents into {len(all_small_chunks)} small chunks with window context.")
        return all_small_chunks