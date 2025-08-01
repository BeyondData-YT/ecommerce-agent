from contextlib import asynccontextmanager
import logging
import asyncio

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from ecommerce_agent.infrastructure.logger import setup_logging

from ecommerce_agent.application.services.conversation_service.generate_response import generate_response, get_streaming_response
from ecommerce_agent.infrastructure.database.postgresql.postgres_client import db_transaction, db_client
from ecommerce_agent.infrastructure.messaging.telegram.telegram_bot_handler import bot_instance, telegram_bot_main
from ecommerce_agent.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager for managing the lifespan of the FastAPI application.
    Initializes database connections, creates necessary tables and functions for document storage,
    and sets up the Telegram bot webhook upon startup. Ensures proper shutdown procedures.

    Args:
        app (FastAPI): The FastAPI application instance.

    Yields:
        None: The execution context within the lifespan.

    Raises:
        Exception: If an error occurs during database initialization.
    """
    setup_logging()
    logging.info("Initializing FastAPI application...")
    try:
      with db_transaction() as conn:
        cursor = conn.cursor()
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_search;") 
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding VECTOR(%s) NOT NULL,
                window_content TEXT,
                source TEXT
            );
        """, (settings.EMBEDDING_DIMENSION,))

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS documents_search_idx
            ON documents USING bm25 (id, content)
            WITH (key_field='id');
        """)

        conn.commit()
        logging.info("Database tables and FTS configurations verified/created successfully for the agent tool.")
        asyncio.create_task(telegram_bot_main(app))
    except Exception as e:
        logging.error(f"Error initializing database at agent startup: {e}")
        raise
    yield
    logging.info("Shutting down FastAPI application...")
    db_client.close_connection()
    logging.info("PostgreSQL connection closed for the agent tool.")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    message: str
    
@app.post("/chat")
async def chat(chat_message: ChatMessage):
  """
  Handles incoming chat messages and generates a response using the conversation agent.

  Args:
    chat_message (ChatMessage): The incoming chat message containing the user's message string.

  Returns:
    dict: A dictionary containing the agent's response.

  Raises:
    HTTPException: If an error occurs during response generation.
  """
  try:
      logging.info(f"Chat message received: {chat_message.message}")
      response, _ = await generate_response(chat_message.message)
      logging.info(f"Response generated: {response}")
      return {"response": response}
  except Exception as e:
      raise HTTPException(status_code=500, detail=str(e))
    
# Telegram Webhook (will handle incoming bot requests)
# This endpoint will be invoked by Telegram when a new message arrives.
@app.post(f"/telegram_webhook/{settings.TELEGRAM_BOT_TOKEN}")
async def telegram_webhook(request: Request):
  """
  Handles incoming Telegram webhook updates.

  Processes text messages from Telegram, generates a response using the conversation agent,
  and sends the response back to the user.

  Args:
    request (Request): The incoming FastAPI request object containing the Telegram update.

  Returns:
    dict: A status dictionary indicating whether the update was processed or ignored.
  """
  update = await request.json()
  logging.info(f"Telegram update received: {update}")
  if "message" in update and "text" in update["message"]:
      text = update["message"]["text"]
      chat_id = update["message"]["chat"]["id"]
      user_id = update["message"]["from"]["id"]
      

      logging.info("Generating response...")
      agent_response_obj, _ = await generate_response(text)
      agent_response_text = str(agent_response_obj)
      logging.info(f"Agent response text: {agent_response_text}")
      # Send the response back to Telegram
      await bot_instance.send_message(chat_id=chat_id, text=agent_response_text)
      logging.info("Response sent to Telegram")
      return {"status": "ok"}
  
  return {"status": "ignored", "message": "No text message received or processed."}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)