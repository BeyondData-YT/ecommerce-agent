from typing import Optional
from ecommerce_agent.infrastructure.database.postgresql.postgres_client import db_transaction
import logging

class BaseService:
  def __init__(self, db_client):
    self.db_client = db_client
  
  def _sanitize_string_for_db(self, text: Optional[str]) -> Optional[str]:
    """
    Removes null bytes from a string before DB insertion to prevent database errors.
    
    Args:
        text (Optional[str]): The input string to sanitize.
        
    Returns:
        Optional[str]: The sanitized string, or None if the input was None.
    """
    if text is None:
      return None
    return text.replace('\x00', '')
  
  def _create_extensions(self):
    try:
      with db_transaction() as conn:
        logging.info("Creating extensions...")
        cursor = conn.cursor()
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_search;") 
        conn.commit()
    except Exception as e:
      logging.error(f"Error creating extensions: {e}")
      raise
      
  def _create_table(self):
    pass
  
  def _create_index(self):
    pass