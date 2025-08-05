from typing import List, Dict, Optional
from ecommerce_agent.config import settings
from ecommerce_agent.domain.document import Document
from ecommerce_agent.application.services.base_service import BaseService
from ecommerce_agent.infrastructure.database.postgresql.postgres_client import db_client, db_transaction
import logging

class DocumentService(BaseService):
    """
    Service class for interacting with the document storage in the PostgreSQL database.
    Handles CRUD operations for documents, including embedding and full-text search capabilities.
    """
    def __init__(self):
        """
        Initializes the DocumentService with a database client.
        """
        super().__init__(db_client) 
        
    def _create_table(self):
        try:
            with db_transaction() as conn:
                cursor = conn.cursor()
                logging.info("Creating documents table...")
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding VECTOR(%s) NOT NULL,
                    window_content TEXT,
                    source TEXT
                );
                """, (settings.EMBEDDING_DIMENSION,))
                conn.commit()
        except Exception as e:
            logging.error(f"Error creating documents table: {e}")
            raise
            
    def _create_index(self):
        try:
            with db_transaction() as conn:
                cursor = conn.cursor()
                logging.info("Creating documents index...")
                cursor.execute("""
                CREATE INDEX IF NOT EXISTS documents_search_idx
                ON documents USING bm25 (id, content)
                WITH (key_field='id');
                """)
            conn.commit()
        except Exception as e:
            logging.error(f"Error creating documents index: {e}")
            raise
        
    def create_document(self, document: Document) -> Optional[Document]:
        """
        Creates a new document record in the database.

        Args:
            document (Document): The Document object containing content, embedding, window_content, and source.

        Returns:
            Optional[Document]: The created Document object with its assigned ID, or None if creation fails.

        Raises:
            ValueError: If an error occurs during database insertion or if the ID is not returned.
        """
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
                logging.info(f"Document with ID {document.id} created successfully.")
                return document
            else:
                raise ValueError("Error creating document: ID not returned.")
        except Exception as e:
            logging.error(f"Error creating document: {e}")
            raise ValueError(f"Error creating document: {e}")
    
    def get_document_by_id(self, document_id: int) -> Document:
        """
        Retrieves a document from the database by its ID.

        Args:
            document_id (int): The unique identifier of the document.

        Returns:
            Document: The retrieved Document object.

        Raises:
            ValueError: If the document is not found or an error occurs during retrieval.
        """
        query = """
            SELECT id, content, embedding, window_content, source
            FROM documents
            WHERE id = %s
        """
        try:
            result = self.db_client.execute_query(query, (document_id,), fetch_one=True)
            if result:
                embedding = [float(x) for x in result['embedding'][1:-1].split(',')] if isinstance(result['embedding'], str) else result['embedding']
                logging.info(f"Document with ID {document_id} retrieved successfully.")
                return Document(
                    id=result['id'],
                    content=result['content'],
                    embedding=embedding,
                    window_content=result.get('window_content'), 
                    source=result.get('source')
                )
            else:
                logging.warning(f"Document with ID {document_id} not found.")
                raise ValueError(f"Document with ID {document_id} not found.")
        except Exception as e:
            logging.error(f"Error retrieving document with ID {document_id}: {e}")
            raise ValueError(f"Error retrieving document: {e}")
            

    def retrieve_similar_documents(self, query_embedding: List[float], top_k: int = 5) -> List[Document]:
        """
        Retrieves documents semantically similar to the given query embedding.

        Args:
            query_embedding (List[float]): The embedding vector of the query.
            top_k (int): The maximum number of similar documents to retrieve. Defaults to 5.

        Returns:
            List[Document]: A list of Document objects ordered by semantic similarity.

        Raises:
            ValueError: If an error occurs during the semantic search.
        """
        query = """
            SELECT id, content, embedding, window_content, source, embedding <=> %s AS distance
            FROM documents
            ORDER BY distance
            LIMIT %s
        """
        embedding_str = f"[{ ','.join(map(str, query_embedding)) }]"
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
                    # Attach distance for fusion use
                    setattr(doc, 'semantic_distance', result['distance'])
                    documents.append(doc)
                logging.info(f"Retrieved {len(documents)} similar documents.")
                return documents
            else:
                logging.info("No similar documents found.")
                return []
        except Exception as e:
            logging.error(f"Error searching for similar documents: {e}")
            raise ValueError(f"Error searching for similar documents: {e}")

    def retrieve_text_search_documents(self, query_text: str, top_k: int = 5) -> List[Document]:
        """
        Retrieves documents matching the given text query using BM25.

        Args:
            query_text (str): The text query string.
            top_k (int): The maximum number of documents to retrieve. Defaults to 5.

        Returns:
            List[Document]: A list of Document objects ordered by text search rank.

        Raises:
            ValueError: If an error occurs during the text search.
        """
        query = """
            SELECT id, content, embedding, window_content, source,
                    paradedb.score(id) AS rank
            FROM documents
            WHERE id @@@ paradedb.with_index('documents_search_idx', paradedb.match('content', %s))
            ORDER BY rank DESC
            LIMIT %s;
        """
        try:
            results = self.db_client.execute_query(query, (query_text, top_k), fetch_all=True)
            
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
                    # Attach rank for fusion use
                    setattr(doc, 'text_rank', result['rank'])
                    documents.append(doc)
                logging.info(f"Retrieved {len(documents)} text search documents.")
                return documents
            return []
        except Exception as e:
            logging.error(f"Error in text search: {e}")
            raise ValueError(f"Error searching documents by text: {e}")
            
    def retrieve_hybrid_documents(self, query_embedding: List[float], query_text: str, top_k: int = 5) -> List[Document]:
        """
        Performs a hybrid search (semantic + full-text) and merges the results using Reciprocal Rank Fusion (RRF).

        Args:
            query_embedding (List[float]): The embedding vector for semantic search.
            query_text (str): The text query string for full-text search.
            top_k (int): The maximum number of combined documents to retrieve. Defaults to 5.

        Returns:
            List[Document]: A list of Document objects representing the top hybrid results.
        """
        logging.info("Executing semantic search for hybrid retrieval.")
        semantic_results = self.retrieve_similar_documents(query_embedding, top_k=top_k * 2) # Get more for fusion
        logging.info("Executing text search for hybrid retrieval.")
        text_results = self.retrieve_text_search_documents(query_text, top_k=top_k * 2) # Get more for fusion
        logging.info("Hybrid document retrieval initiated.")

        reranked_scores: Dict[int, float] = {}
        documents_by_id: Dict[int, Document] = {}
        k = 60

        # Process semantic results
        for i, doc in enumerate(semantic_results):
            doc_id = doc.id
            documents_by_id[doc_id] = doc
            reranked_scores[doc_id] = reranked_scores.get(doc_id, 0.0) + 1.0 / (k + i + 1)

        # Process text results
        for i, doc in enumerate(text_results):
            doc_id = doc.id
            documents_by_id[doc_id] = doc
            reranked_scores[doc_id] = reranked_scores.get(doc_id, 0.0) + 1.0 / (k + i + 1)

        # Sort documents by their combined RRF score (from highest to lowest)
        sorted_doc_ids = sorted(reranked_scores.keys(), key=lambda doc_id: reranked_scores[doc_id], reverse=True)

        # Reconstruct the final list of documents
        hybrid_documents: List[Document] = []
        for doc_id in sorted_doc_ids:
            if len(hybrid_documents) >= top_k:
                break
            hybrid_documents.append(documents_by_id[doc_id])
        
        for doc in hybrid_documents:
            setattr(doc, 'rrf_score', reranked_scores[doc.id])

        logging.info(f"Successfully retrieved {len(hybrid_documents)} hybrid documents.")
        return hybrid_documents