from telegram import Update, Bot
from telegram.ext import Application, MessageHandler, filters, ContextTypes, CommandHandler
import asyncio
# Asegúrate de que esta importación sea correcta según tu estructura de carpetas
from ecommerce_agent.config import settings 
from fastapi import FastAPI # Importar FastAPI para el webhook, aunque no se usa en el modo polling
import logging

# Inicializa el bot de Telegram
bot_instance = Bot(token=settings.TELEGRAM_BOT_TOKEN)

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a message when the command /start is issued."""
    await update.message.reply_text('¡Hola! Soy tu asistente de e-commerce. Pregúntame cualquier cosa.')

async def echo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Echoes the user message. 
    This handler is used for simple testing in polling mode.
    """
    await update.message.reply_text(f"Recibí: {update.message.text}")

async def telegram_bot_main(app_instance: FastAPI):
    """
    Configures and sets up the Telegram bot webhook for FastAPI integration.
    This function is intended to be called from your FastAPI application's startup event.
    """
    application = Application.builder().token(settings.TELEGRAM_BOT_TOKEN).build()

    # Construye la URL del webhook usando la configuración
    # Asegúrate de que settings.WEBHOOK_URL esté configurado en tu archivo .env
    # Por ejemplo: WEBHOOK_URL="https://tu-ngrok-url.ngrok-free.app"
    webhook_url = f"{settings.WEBHOOK_URL}/telegram_webhook/{settings.TELEGRAM_BOT_TOKEN}"
    
    # Eliminar webhooks antiguos para evitar conflictos
    await application.bot.delete_webhook()
    
    # Establecer el nuevo webhook
    await application.bot.set_webhook(url=webhook_url)
    logging.info(f"Telegram webhook configured at: {webhook_url}")

    # En este modo, la aplicación de Telegram no necesita "run_polling" ni "run_webhook"
    # porque FastAPI manejará las solicitudes entrantes del webhook.
    # Solo inicializamos el bot para poder usar `bot_instance` para enviar mensajes desde FastAPI.


def run_polling_bot():
    """
    Runs the Telegram bot in polling mode for local testing.
    This function will check for new messages periodically.
    """
    application = Application.builder().token(settings.TELEGRAM_BOT_TOKEN).build()

    # Añadir manejadores para comandos y mensajes de texto
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo_message))
    
    logging.info("Telegram bot starting in polling mode (for local testing)...")
    # Inicia el bot y lo mantiene escuchando mensajes
    application.run_polling(poll_interval=1.0) # Revisa nuevos mensajes cada 1 segundo

if __name__ == "__main__":
    # Este bloque se ejecuta solo cuando el script se corre directamente
    # No se ejecuta cuando se importa como un módulo en FastAPI
    run_polling_bot()