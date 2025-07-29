from typing import List, Dict
from ecommerce_agent.domain.document import Document
from ecommerce_agent.infrastructure.database.postgresql.postgres_client import db_client
import math # Para la función de Reciprocal Rank Fusion
import logging # Para logging consistente

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentService:
    def __init__(self):
        self.db_client = db_client
    
    def _sanitize_string_for_db(self, text: str) -> str:
        """Removes null bytes from a string before DB insertion."""
        if text is None:
            return None
        return text.replace('\x00', '')
    
    def create_document(self, document: Document):
        sanitized_content = self._sanitize_string_for_db(document.content)
        sanitized_window_content = self._sanitize_string_for_db(document.window_content)
        sanitized_source = self._sanitize_string_for_db(document.source)
        query = """
            INSERT INTO documents (content, embedding, window_content, source)
            VALUES (%s, %s, %s, %s)
            RETURNING id
        """
        embedding_str = f"[{','.join(map(str, document.embedding))}]"
        if document.window_content is None:
            document.window_content = document.content
        if document.content is None:
            return
        try:
            result = self.db_client.execute_query(
                query, 
                (sanitized_content, embedding_str, sanitized_window_content, sanitized_source), 
                fetch_one=True
            )
            if result and 'id' in result:
                document.id = result['id']
                document.content = sanitized_content 
                document.window_content = sanitized_window_content
                document.source = sanitized_source
                logging.info(f"Documento con ID {document.id} creado exitosamente.")
                return document
            else:
                raise ValueError("Error al crear el documento: ID no retornado.")
        except Exception as e:
            logging.error(f"Error al crear el documento: {e}")
            raise ValueError(f"Error al crear el documento: {e}")
    
    # Mantener el método get_document_by_id si lo necesitas, adaptando la recuperación de campos
    def get_document_by_id(self, document_id: int):
        query = """
            SELECT id, content, embedding, window_content, source
            FROM documents
            WHERE id = %s
        """
        try:
            result = self.db_client.execute_query(query, (document_id,), fetch_one=True)
            if result:
                embedding = [float(x) for x in result['embedding'][1:-1].split(',')] if isinstance(result['embedding'], str) else result['embedding']
                return Document(
                    id=result['id'],
                    content=result['content'],
                    embedding=embedding,
                    window_content=result.get('window_content'), # Usar .get para seguridad
                    source=result.get('source')
                )
            else:
                raise ValueError(f"Documento con ID {document_id} no encontrado")
        except Exception as e:
            logging.error(f"Error al obtener el documento con ID {document_id}: {e}")
            raise ValueError(f"Error al obtener el documento: {e}")
            
    # La búsqueda semántica seguirá sobre 'content' y retornará 'window_content' y 'source'
    def retrieve_similar_documents(self, query_embedding: List[float], top_k: int = 5):
        query = """
            SELECT id, content, embedding, window_content, source, embedding <=> %s AS distance
            FROM documents
            ORDER BY distance
            LIMIT %s
        """
        embedding_str = f"[{','.join(map(str, query_embedding))}]"
        try:
            results = self.db_client.execute_query(query, (embedding_str, top_k), fetch_all=True)
            if results:
                documents = []
                for result in results:
                    embedding = [float(x) for x in result['embedding'][1:-1].split(',')] if isinstance(result['embedding'], str) else result['embedding']
                    doc = Document(
                        id=result['id'],
                        content=result['content'],
                        embedding=embedding,
                        window_content=result.get('window_content'),
                        source=result.get('source')
                    )
                    # Adjuntamos la distancia para uso en la fusión
                    setattr(doc, 'semantic_distance', result['distance'])
                    documents.append(doc)
                logging.info(f"Similar documents retrieved: {len(documents)}")
                return documents
            else:
                return []
        except Exception as e:
            logging.error(f"Error al buscar documentos similares: {e}")
            raise ValueError(f"Error al buscar documentos similares: {e}")

    # La búsqueda de texto seguirá sobre 'content_tsv' y retornará 'window_content' y 'source'
    def retrieve_text_search_documents(self, query_text: str, top_k: int = 5):
        query = """
            SELECT id, content, embedding, window_content, source,
                    ts_rank_cd(content_tsv, plainto_tsquery('spanish', %s)) AS rank
            FROM documents
            WHERE content_tsv @@ plainto_tsquery('spanish', %s)
            ORDER BY rank DESC
            LIMIT %s;
        """
        try:
            results = self.db_client.execute_query(query, (query_text, query_text, top_k), fetch_all=True)
            
            if results:
                documents = []
                for result in results:
                    embedding_list = [float(x) for x in result['embedding'][1:-1].split(',')] if isinstance(result['embedding'], str) else result['embedding']
                    doc = Document(
                        id=result['id'], 
                        content=result['content'], 
                        embedding=embedding_list,
                        window_content=result.get('window_content'),
                        source=result.get('source')
                    )
                    # Adjuntamos el rank para uso en la fusión
                    setattr(doc, 'text_rank', result['rank'])
                    documents.append(doc)
                logging.info(f"Text search documents retrieved: {len(documents)}")
                return documents
            return []
        except Exception as e:
            logging.error(f"Error en búsqueda de texto: {e}")
            raise ValueError(f"Error al buscar documentos por texto: {e}")
            
    # La búsqueda híbrida simplemente llamará a las anteriores, que ya retornarán los campos correctos
    def retrieve_hybrid_documents(self, query_embedding: List[float], query_text: str, top_k: int = 5) -> List[Document]:
        """
        Tool para realizar una búsqueda híbrida (semántica + texto) y fusionar los resultados.
        Utiliza Reciprocal Rank Fusion (RRF) para combinar los rankings.
        """
        # Realiza ambas búsquedas
        # Ajusta los límites de cada búsqueda para obtener suficientes candidatos para la fusión
        logging.info("Executing semantic search")
        semantic_results = self.retrieve_similar_documents(query_embedding, top_k=top_k * 2) # Obtén más para la fusión
        logging.info("Executing text search")
        text_results = self.retrieve_text_search_documents(query_text, top_k=top_k * 2) # Obtén más para la fusión
        logging.info("Retrieved hybrid documents")

        # Diccionario para almacenar el score RRF de cada documento
        # { doc_id: score }
        reranked_scores: Dict[int, float] = {}
        
        # Mapeo de ID a Documento para reconstruir los resultados
        documents_by_id: Dict[int, Document] = {}

        # Constante K para RRF (ajustable)
        k = 60 

        # Procesa resultados semánticos
        for i, doc in enumerate(semantic_results):
            doc_id = doc.id
            documents_by_id[doc_id] = doc
            reranked_scores[doc_id] = reranked_scores.get(doc_id, 0.0) + 1.0 / (k + i + 1)

        # Procesa resultados de texto
        for i, doc in enumerate(text_results):
            doc_id = doc.id
            documents_by_id[doc_id] = doc
            reranked_scores[doc_id] = reranked_scores.get(doc_id, 0.0) + 1.0 / (k + i + 1)

        # Ordena los documentos por su score RRF combinado (de mayor a menor)
        sorted_doc_ids = sorted(reranked_scores.keys(), key=lambda doc_id: reranked_scores[doc_id], reverse=True)

        # Reconstruye la lista final de documentos
        hybrid_documents: List[Document] = []
        for doc_id in sorted_doc_ids:
            if len(hybrid_documents) >= top_k:
                break
            hybrid_documents.append(documents_by_id[doc_id])
        
        # Opcional: Puedes añadir el score RRF al objeto Document si lo necesitas para depuración
        for doc in hybrid_documents:
            setattr(doc, 'rrf_score', reranked_scores[doc.id])

        logging.info(f"Hybrid documents retrieved: {len(hybrid_documents)}")
        return hybrid_documents