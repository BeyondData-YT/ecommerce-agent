from telegram import Update, Bot
from telegram.ext import Application, MessageHandler, filters, ContextTypes, CommandHandler
import asyncio
from ecommerce_agent.config import settings 
from fastapi import FastAPI
import logging

# Initializes the Telegram bot
bot_instance = Bot(token=settings.TELEGRAM_BOT_TOKEN)

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Sends a message when the command /start is issued.

    Args:
        update (Update): The Telegram Update object.
        context (ContextTypes.DEFAULT_TYPE): The context object for the bot.
    """
    await update.message.reply_text('Hello! I am your e-commerce assistant. Ask me anything.')

async def echo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Echoes the user message. 
    This handler is used for simple testing in polling mode.

    Args:
        update (Update): The Telegram Update object.
        context (ContextTypes.DEFAULT_TYPE): The context object for the bot.
    """
    await update.message.reply_text(f"Received: {update.message.text}")

async def telegram_bot_main(app_instance: FastAPI) -> None:
    """
    Configures and sets up the Telegram bot webhook for FastAPI integration.
    This function is intended to be called from your FastAPI application's startup event.

    Args:
        app_instance (FastAPI): The FastAPI application instance to which the webhook will be linked.
    """
    application = Application.builder().token(settings.TELEGRAM_BOT_TOKEN).build()

    # Build the webhook URL using the configuration
    webhook_url = f"{settings.WEBHOOK_URL}/telegram_webhook/{settings.TELEGRAM_BOT_TOKEN}"
    
    # Delete old webhooks to avoid conflicts
    await application.bot.delete_webhook()
    
    # Set the new webhook
    await application.bot.set_webhook(url=webhook_url)
    logging.info(f"Telegram webhook configured at: {webhook_url}")

    # In this mode, the Telegram application does not need to "run_polling" or "run_webhook"
    # because FastAPI will handle incoming webhook requests.
    # We only initialize the bot to be able to use `bot_instance` to send messages from FastAPI.


def run_polling_bot() -> None:
    """
    Runs the Telegram bot in polling mode for local testing.
    This function will check for new messages periodically.
    """
    application = Application.builder().token(settings.TELEGRAM_BOT_TOKEN).build()

    # Add handlers for commands and text messages
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo_message))
    
    logging.info("Telegram bot starting in polling mode (for local testing)...")
    # Start the bot and keep it listening for messages
    application.run_polling(poll_interval=1.0) # Check for new messages every 1 second

if __name__ == "__main__":
    run_polling_bot()