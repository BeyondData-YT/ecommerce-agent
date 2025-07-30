import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from ecommerce_agent.config import settings
import logging
from typing import Optional, Union

class PostgresClient:
  """
  A client for interacting with the PostgreSQL database.
  Provides methods for establishing and closing connections, and executing queries.
  """
  def __init__(self):
    """
    Initializes the PostgresClient by loading connection parameters.
    """
    self.conn_params = self._load_conn_params()
    self._connection = None
    
  def _load_conn_params(self) -> dict[str, Union[str, int]]:
    """
    Loads PostgreSQL connection parameters from application settings.

    Returns:
      dict: A dictionary containing connection parameters.
    """
    return {
      "host": settings.POSTGRES_HOST,
      "port": settings.POSTGRES_PORT,
      "database": settings.POSTGRES_DB,
      "user": settings.POSTGRES_USER,
      "password": settings.POSTGRES_PASSWORD
    }
    
  def _get_connection(self) -> psycopg2.extensions.connection:
    """
    Establishes and returns a new PostgreSQL database connection if one does not exist or is closed.

    Returns:
      psycopg2.extensions.connection: An active database connection object.

    Raises:
      ConnectionError: If there is an error connecting to the database.
    """
    if self._connection is None or self._connection.closed:
      try:
        self._connection = psycopg2.connect(**self.conn_params)
        self._connection.autocommit = False # Control transactions manually
        logging.info("Connected to the database")
      except psycopg2.Error as e:
        raise ConnectionError(f"Error connecting to the database: {e}")
    return self._connection
  
  def close_connection(self) -> None:
    """
    Closes the current PostgreSQL database connection if it is open.
    """
    if self._connection and not self._connection.closed:
      self._connection.close()
      self._connection = None
      logging.info("Disconnected from the database")
  
  def execute_query(self, query: str, params: Optional[tuple] = None, fetch_one: bool = False, fetch_all: bool = False) -> Optional[Union[dict, list[dict]]]:
    """
    Executes a SQL query on the PostgreSQL database.

    Args:
      query (str): The SQL query string to execute.
      params (Optional[tuple]): A tuple of parameters to pass to the query. Defaults to None.
      fetch_one (bool): If True, fetches a single row. Defaults to False.
      fetch_all (bool): If True, fetches all rows. Defaults to False.

    Returns:
      Optional[Union[dict, list[dict]]]: The fetched row (as a dictionary), all fetched rows (as a list of dictionaries), or None.

    Raises:
      psycopg2.Error: If an error occurs during query execution.
    """
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

# Global database client instance
db_client = PostgresClient()

@contextmanager
def db_transaction():
  """
  Provides a context manager for database transactions.
  Ensures commit on success and rollback on error.

  Yields:
    psycopg2.extensions.connection: The database connection object.

  Raises:
    psycopg2.Error: If an error occurs within the transaction block.
  """
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

