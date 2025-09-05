from contextlib import asynccontextmanager
import logging
import json
from ecommerce_agent.infrastructure.logger import setup_logging
setup_logging()
import asyncio

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessage

from pydantic import BaseModel

from ecommerce_agent.application.services.conversation_service.generate_response import generate_response
from ecommerce_agent.application.services.conversation_service.workflow.nodes import handle_mixed_input
from ecommerce_agent.application.services.speech_service.speech_to_text import SpeechToTextService
from ecommerce_agent.application.services.session import SessionService
from ecommerce_agent.infrastructure.database.postgresql.postgres_client import db_client
from ecommerce_agent.infrastructure.messaging.telegram.telegram_bot_handler import bot_instance, telegram_bot_main
from ecommerce_agent.config import settings

session_service = SessionService()
speech_to_text_service = SpeechToTextService()

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
    logging.info("Initializing FastAPI application...")
    try:
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
  user_id = int(update["message"]["from"]["id"])
  chat_id = update["message"]["chat"]["id"]
  logging.info(f"Telegram update received: {update}")

  if "message" in update and "text" in update["message"]:
      text = update["message"]["text"]

      if text.strip().lower().startswith("/newchat"):
        thread_id = await session_service.start_new_session(user_id)
        await bot_instance.send_message(chat_id=user_id, text=f"New chat session started with thread ID: {thread_id}")
        return {"status": "ok"}
      
      thread_id = await session_service.get_or_create_session(user_id)
      # thread_id = "9"
      logging.info(f"Thread ID: {thread_id} \n User ID: {user_id}")

      logging.info("Generating response...")
      agent_response_obj, _ = await generate_response(text, workflow="conversation", thread_id=thread_id, user_id=user_id)
      agent_response_text = str(agent_response_obj)
      logging.info(f"Agent response text: {agent_response_text}")
      # Send the response back to Telegram
      if agent_response_text:
        await bot_instance.send_message(chat_id=chat_id, text=agent_response_text)
      logging.info("Response sent to Telegram")
      return {"status": "ok"}
  
  elif "message" in update and "photo" in update["message"]:
    photo = update["message"]["photo"]
    caption = update["message"].get("caption", None)
    photo_id = photo[-1]["file_id"]
    file = await bot_instance.get_file(photo_id)
    image_url = file.file_path
    
    thread_id = await session_service.get_or_create_session(user_id)
    # thread_id = "9"
    logging.info(f"Thread ID: {thread_id} \n User ID: {user_id}")
    
    input_message = f"The user has sent a photo with the following caption: {caption}. The image url is: {image_url}" if caption else f"The user has sent a photo. The image url is: {image_url}"
    
    logging.info("Generating response...")
    agent_response_obj, state = await generate_response(input_message, workflow="image", thread_id=thread_id, user_id=user_id)
    
    for message in reversed(state["messages"]):
      if isinstance(message, HumanMessage):
        logging.info(f"Human message received: {message.content}")
        try:
          image_response_list = handle_mixed_input(message.content)
          logging.info(f"Image response list: {image_response_list}")
          for image_response in image_response_list:
            logging.info(f"Sending image response to Telegram: {image_response}")
            await bot_instance.send_photo(chat_id=chat_id, photo=image_response["image_url"], caption=image_response["caption"])
            logging.info(f"Image response sent to Telegram")
          return {"status": "ok"}
        except Exception as e:
          logging.error(f"Error sending image response: {e}")
          return {"status": "error", "message": "Error sending image response"}
      
    if agent_response_obj:
      await bot_instance.send_message(chat_id=chat_id, text=str(agent_response_obj))
    logging.info("Response sent to Telegram")
    return {"status": "ok"}
    
  
  
  elif "message" in update and ("audio" or "voice" in update["message"]):
    media_type = "audio" if "audio" in update["message"] else "voice"
    file_id = update["message"][media_type]["file_id"]
    file = await bot_instance.get_file(file_id)
    file_bytes = await file.download_as_bytearray()
    file_bytes = bytes(file_bytes)
    
    thread_id = await session_service.get_or_create_session(user_id)
    logging.info(f"Thread ID: {thread_id} \n User ID: {user_id}")
    
    transcription = await speech_to_text_service.transcribe(file_bytes)
    logging.info(f"Transcription: {transcription}")
    
    logging.info("Generating response...")
    agent_response_obj, agent_response_state = await generate_response(transcription, workflow="audio", thread_id=thread_id, user_id=user_id)
    
    if agent_response_state["workflow"] == "audio":
      agent_response_audio_buffer = agent_response_state["audio_buffer"]
      await bot_instance.send_audio(chat_id=chat_id, audio=agent_response_audio_buffer)
    else:
      await bot_instance.send_message(chat_id=chat_id, text=str(agent_response_obj))
    logging.info("Response sent to Telegram")
    
    return {"status": "ok"}
    
  return {"status": "ignored", "message": "No text message received or processed."}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)