from typing import List, Dict, Optional
from ecommerce_agent.config import settings
from ecommerce_agent.domain.product import Product
from ecommerce_agent.application.services.base_service import BaseService
from ecommerce_agent.infrastructure.database.postgresql.postgres_client import db_client, db_transaction
import logging

class ProductsService(BaseService):
  def __init__(self):
    """
    Initializes the DocumentService with a database client.
    """
    super().__init__(db_client)
    
  def _create_table(self):
    try:
      with db_transaction() as conn:
        cursor = conn.cursor()
        logging.info("Creating products table...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
          id SERIAL PRIMARY KEY,
          code TEXT NOT NULL,
          name TEXT NOT NULL,
          description TEXT NOT NULL,
          embedding VECTOR(%s) NOT NULL,
          price DECIMAL(10, 2) NOT NULL,
          image_url TEXT,
          stock_level INT NOT NULL,
          is_active BOOLEAN NOT NULL
        );
        """, (settings.EMBEDDING_DIMENSION,))
        conn.commit()
    except Exception as e:
      logging.error(f"Error creating products table: {e}")
      raise
      
  def _create_index(self):
    try:
      with db_transaction() as conn:
        cursor = conn.cursor()
        logging.info("Creating products index...")
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS products_search_idx
        ON products USING bm25 (id, name, description)
        WITH (key_field='id');
        """)
      conn.commit()
    except Exception as e:
      logging.error(f"Error creating products index: {e}")
      raise
  
  def create_product(self, product: Product) -> Optional[Product]:
    sanitized_name = self._sanitize_string_for_db(product.name)
    sanitized_code = self._sanitize_string_for_db(product.code)
    sanitized_description = self._sanitize_string_for_db(product.description)
    sanitized_image_url = self._sanitize_string_for_db(product.image_url)
    embedding_str = f"[{','.join(map(str, product.embedding))}]"
    query = """
        INSERT INTO products (code, name, description, embedding, price, image_url, stock_level, is_active)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """
    
    try:
      result = self.db_client.execute_query(
        query, 
        (sanitized_code, sanitized_name, sanitized_description, embedding_str, product.price, sanitized_image_url, product.stock_level, product.is_active), 
        fetch_one=True
      )
      if result and 'id' in result:
        product.id = result['id']
        product.name = sanitized_name
        product.description = sanitized_description
        product.code = sanitized_code
        product.image_url = sanitized_image_url
        product.stock_level = product.stock_level
        product.is_active = product.is_active
        logging.info(f"Product with ID {product.id} created successfully.")
        return product
      else:
        raise ValueError("Error creating product: ID not returned.")
    except Exception as e:
      logging.error(f"Error creating product: {e}")
      raise ValueError(f"Error creating product: {e}")
    
  def get_product_by_id(self, product_id: int) -> Optional[Product]:
    query = """
        SELECT * FROM products WHERE id = %s AND is_active = TRUE
    """
    try:
      result = self.db_client.execute_query(query, (product_id,), fetch_one=True)
      if result:
        embedding = [float(x) for x in result['embedding'][1:-1].split(',')] if isinstance(result['embedding'], str) else result['embedding']
        logging.info(f"Product with ID {product_id} retrieved successfully.")
        return Product(
          id=result['id'],
          code=result['code'],
          name=result['name'],
          description=result['description'],
          embedding=embedding,
          price=result['price'],
          image_url=result['image_url'],
          stock_level=result['stock_level'],
          is_active=result['is_active'])
      else:
        logging.warning(f"Product with ID {product_id} not found.")
        raise ValueError(f"Product with ID {product_id} not found.")
    except Exception as e:
      logging.error(f"Error retrieving product with ID {product_id}: {e}")
      raise ValueError(f"Error retrieving product: {e}")
      
  def get_product_by_code(self, product_code: str) -> Optional[Product]:
    query = """
        SELECT * FROM products WHERE code = %s AND is_active = TRUE
    """
    try:
      result = self.db_client.execute_query(query, (product_code,), fetch_one=True)
      if result:
        embedding = [float(x) for x in result['embedding'][1:-1].split(',')] if isinstance(result['embedding'], str) else result['embedding']
        logging.info(f"Product with code {product_code} retrieved successfully.")
        return Product(
          id=result['id'],
          code=result['code'],
          name=result['name'],
          description=result['description'],
          embedding=embedding,
          price=result['price'],
          image_url=result['image_url'],
          stock_level=result['stock_level'],
          is_active=result['is_active'])
      else:
        logging.warning(f"Product with code {product_code} not found.")
        raise ValueError(f"Product with code {product_code} not found.")
    except Exception as e:
      logging.error(f"Error retrieving product with code {product_code}: {e}")
      raise ValueError(f"Error retrieving product: {e}")
    
  def retrieve_similar_products(self, query_embedding: List[float], top_k: int = 5) -> List[Product]:
    query = """
        SELECT id, code, name, description, embedding, price, image_url, stock_level, is_active, embedding <=> %s AS distance
        FROM products
        WHERE is_active = TRUE
        ORDER BY distance
        LIMIT %s
    """
    embedding_str = f"[{ ','.join(map(str, query_embedding)) }]"
    try:
      results = self.db_client.execute_query(query, (embedding_str, top_k), fetch_all=True)
      if results:
        products = []
        for result in results:
          embedding = [float(x) for x in result['embedding'][1:-1].split(',')] if isinstance(result['embedding'], str) else result['embedding']
          product = Product(
            id=result['id'],
            code=result['code'],
            name=result['name'],
            description=result['description'],
            embedding=embedding,
            price=result['price'],
            image_url=result['image_url'],
            stock_level=result['stock_level'],
            is_active=result['is_active']
          )
          setattr(product, 'semantic_distance', result['distance'])
          products.append(product)
        logging.info(f"Retrieved {len(products)} similar products.")
        return products
      else:
        logging.info("No similar products found.")
        return []
    except Exception as e:
      logging.error(f"Error retrieving similar products: {e}")
      raise ValueError(f"Error retrieving similar products: {e}")
    
  def retrieve_text_search_products(self, query_text: str, top_k: int = 5) -> List[Product]:
    sanitized_query_text = self._sanitize_string_for_db(query_text)
    query = """
        SELECT id, code, name, description, embedding, price, image_url, stock_level, is_active, paradedb.score(id) AS rank
        FROM products
        WHERE id @@@ paradedb.with_index('products_search_idx', paradedb.match('description', %s)) AND is_active = TRUE
        ORDER BY rank DESC
        LIMIT %s 
    """
    try:
      results = self.db_client.execute_query(query, (sanitized_query_text, top_k), fetch_all=True)
      if results:
        products = []
        for result in results:
          embedding = [float(x) for x in result['embedding'][1:-1].split(',')] if isinstance(result['embedding'], str) else result['embedding']
          product = Product(
            id=result['id'],
            code=result['code'],
            name=result['name'],
            description=result['description'],
            embedding=embedding,
            price=result['price'],
            image_url=result['image_url'],
            stock_level=result['stock_level'],
            is_active=result['is_active']
          )
          setattr(product, 'text_rank', result['rank'])
          products.append(product)
        logging.info(f"Retrieved {len(products)} text search products.")
        return products
      else:
        logging.info("No text search products found.")
        return []
    except Exception as e:
      logging.error(f"Error retrieving text search products: {e}")
      raise ValueError(f"Error retrieving text search products: {e}")
    
  def retrieve_hybrid_products(self, query_embedding: List[float], query_text: str, top_k: int = 5) -> List[Product]:
    semantic_results = self.retrieve_similar_products(query_embedding, top_k=top_k * 2)
    text_results = self.retrieve_text_search_products(query_text, top_k=top_k * 2)
    
    reranked_scores: Dict[int, float] = {}
    products_by_id: Dict[int, Product] = {}
    k = 60

    # Process semantic results
    for i, product in enumerate(semantic_results):
        product_id = product.id
        products_by_id[product_id] = product
        reranked_scores[product_id] = reranked_scores.get(product_id, 0.0) + 1.0 / (k + i + 1)

    # Process text results
    for i, product in enumerate(text_results):
        product_id = product.id
        products_by_id[product_id] = product
        reranked_scores[product_id] = reranked_scores.get(product_id, 0.0) + 1.0 / (k + i + 1)

    # Sort documents by their combined RRF score (from highest to lowest)
    sorted_product_ids = sorted(reranked_scores.keys(), key=lambda product_id: reranked_scores[product_id], reverse=True)

    # Reconstruct the final list of documents
    hybrid_products: List[Product] = []
    for product_id in sorted_product_ids:
        if len(hybrid_products) >= top_k:
            break
        hybrid_products.append(products_by_id[product_id])
    
    for product in hybrid_products:
        setattr(product, 'rrf_score', reranked_scores[product.id])

    logging.info(f"Successfully retrieved {len(hybrid_products)} hybrid products.")
    return hybrid_products