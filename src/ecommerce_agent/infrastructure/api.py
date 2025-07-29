from contextlib import asynccontextmanager
import logging
import asyncio

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from ecommerce_agent.application.services.conversation_service.generate_response import generate_response, get_streaming_response
from ecommerce_agent.infrastructure.database.postgresql.postgres_client import db_transaction, db_client
from ecommerce_agent.infrastructure.messaging.telegram.telegram_bot_handler import bot_instance, telegram_bot_main
from ecommerce_agent.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Initializing FastAPI application...")
    try:
      with db_transaction() as conn:
          cursor = conn.cursor()
          cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
          cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;") 
          
          cursor.execute("""
              CREATE TABLE IF NOT EXISTS documents (
                  id SERIAL PRIMARY KEY,
                  content TEXT NOT NULL,
                  embedding VECTOR(1536) NOT NULL,
                  content_tsv TSVECTOR,
                  window_content TEXT,
                  source TEXT
              );
          """)
          
          cursor.execute("""
              CREATE INDEX IF NOT EXISTS documents_content_tsv_idx
              ON documents USING GIN (content_tsv);
          """)

          cursor.execute("""
              CREATE OR REPLACE FUNCTION update_documents_content_tsv() RETURNS TRIGGER AS $$
              BEGIN
                  NEW.content_tsv := to_tsvector('spanish', NEW.content);
                  RETURN NEW;
              END;
              $$ LANGUAGE plpgsql;
          """)

          cursor.execute("""
              DROP TRIGGER IF EXISTS trg_documents_content_tsv ON documents;
          """)
          cursor.execute("""
              CREATE TRIGGER trg_documents_content_tsv
              BEFORE INSERT OR UPDATE ON documents
              FOR EACH ROW EXECUTE FUNCTION update_documents_content_tsv();
          """)
          
          cursor.execute("""
              UPDATE documents SET content_tsv = to_tsvector('spanish', content) WHERE content_tsv IS NULL;
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
  try:
      logging.info(f"Received chat message: {chat_message.message}")
      response, _ = await generate_response(chat_message.message)
      logging.info(f"Response generated: {response}")
      return {"response": response}
  except Exception as e:
      raise HTTPException(status_code=500, detail=str(e))
    
# Webhook de Telegram (manejará las solicitudes entrantes del bot)
# Este endpoint será invocado por Telegram cuando haya un nuevo mensaje.
@app.post(f"/telegram_webhook/{settings.TELEGRAM_BOT_TOKEN}")
async def telegram_webhook(request: Request):
  update = await request.json()
  print(f"Received Telegram update: {update}")
  # La lógica de procesamiento real se delegará al bot_handler
  # Para el MVP, solo procesaremos mensajes de texto simples aquí.
  if "message" in update and "text" in update["message"]:
      text = update["message"]["text"]
      chat_id = update["message"]["chat"]["id"]
      user_id = update["message"]["from"]["id"]
      
      # Simular historial de chat vacío para el MVP.
      # En un sistema real, aquí se recuperaría el historial de una memoria.
      # chat_history_for_agent = []

      logging.info("Generating response...")  
      agent_response_obj, _ = await generate_response(text)
      agent_response_text = str(agent_response_obj)
      logging.info(f"Agent response text: {agent_response_text}")
      # Enviar la respuesta de vuelta a Telegram
      await bot_instance.send_message(chat_id=chat_id, text=agent_response_text)
      logging.info("Response sent to Telegram")
      return {"status": "ok"}
  
  return {"status": "ignored", "message": "No text message"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)