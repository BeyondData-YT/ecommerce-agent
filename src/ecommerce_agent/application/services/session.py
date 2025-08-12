import uuid
import logging
from datetime import datetime, timezone

from ecommerce_agent.infrastructure.database.postgresql.postgres_client import db_client, db_transaction

class SessionService:
  def __init__(self):
    self.db_client = db_client
    self.db_transaction = db_transaction
    
  def create_table(self):
    try:
      with self.db_transaction() as conn:
        cursor = conn.cursor()
        logging.info("Creating sessions table...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS telegram_sessions  (
          user_id BIGINT PRIMARY KEY,
          current_thread_id TEXT NOT NULL,
          last_message_timestamp TIMESTAMP WITH TIME ZONE NOT NULL
        );
        """)
        conn.commit()
    except Exception as e:
      logging.error(f"Error creating sessions table: {e}")
      raise
      
  def _get_current_session(self, user_id: int) -> str:
    try:
      query = """
        SELECT current_thread_id FROM telegram_sessions WHERE user_id = %s
      """
      result = self.db_client.execute_query(
        query,
        (user_id,),
        fetch_one=True
      )
      return result['current_thread_id'] if result else None
    except Exception as e:
      logging.error(f"Error getting current session: {e}")
      raise
  
  def _upsert_session(self, user_id: int, thread_id: str):
    try:
      query = """
        INSERT INTO telegram_sessions (user_id, current_thread_id, last_message_timestamp)
        VALUES (%s, %s, %s)
        ON CONFLICT (user_id) DO UPDATE SET 
        current_thread_id = EXCLUDED.current_thread_id,
        last_message_timestamp = EXCLUDED.last_message_timestamp
      """
      self.db_client.execute_query(
        query,
        (user_id, thread_id, datetime.now(timezone.utc))
      )
    except Exception as e:
      logging.error(f"Error upserting session: {e}")
      raise
      
  async def get_or_create_session(self, user_id: int) -> str:
    thread_id = self._get_current_session(user_id)
    if not thread_id:
      thread_id = str(uuid.uuid4())
      self._upsert_session(user_id, thread_id)
    else:
      self._upsert_session(user_id, thread_id)
      
    return thread_id
  
  async def start_new_session(self, user_id: int) -> str:
    new_thread_id = str(uuid.uuid4())
    self._upsert_session(user_id, new_thread_id)
    return new_thread_id
