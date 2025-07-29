import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from ecommerce_agent.config import settings
import logging

class PostgresClient:
  def __init__(self):
    self.conn_params = self._load_conn_params()
    self._connection = None
    
  def _load_conn_params(self):
    return {
      "host": settings.POSTGRES_HOST,
      "port": settings.POSTGRES_PORT,
      "database": settings.POSTGRES_DB,
      "user": settings.POSTGRES_USER,
      "password": settings.POSTGRES_PASSWORD
    }
    
  def _get_connection(self):
    if self._connection is None or self._connection.closed:
      try:
        self._connection = psycopg2.connect(**self.conn_params)
        self._connection.autocommit = False # Controlar transacciones manualmente
        logging.info("Connected to the database")
      except psycopg2.Error as e:
        raise ConnectionError(f"Error al conectar a la base de datos: {e}")
    return self._connection
  
  def close_connection(self):
    if self._connection and not self._connection.closed:
      self._connection.close()
      self._connection = None
      logging.info("Disconnected from the database")
  
  def execute_query(self, query, params=None, fetch_one=False, fetch_all=False):
    connection = self._get_connection()
    cursor = None
    try:
      cursor = connection.cursor(cursor_factory=RealDictCursor)
      cursor.execute(query, params)
      logging.info("Query executed")
      
      if fetch_one:
        result = cursor.fetchone()
      elif fetch_all:
        result = cursor.fetchall()
      else:
        result = None
      
      query_upper = query.strip().upper()
      if query_upper.startswith("INSERT") or \
        query_upper.startswith("UPDATE") or \
        query_upper.startswith("DELETE"):
        connection.commit()

      return result
    
    except psycopg2.Error as e:
      logging.error(f"Error executing query: {e}")
      connection.rollback()
      raise 
    finally:
      if cursor:
        cursor.close()

# Instancia global del cliente de base de datos
db_client = PostgresClient()

@contextmanager
def db_transaction():
  connection = db_client._get_connection()
  try:
    yield connection
    connection.commit()
  except psycopg2.Error as e:
    logging.error(f"Error in database transaction: {e}")
    connection.rollback()
    raise
  finally:
    pass

