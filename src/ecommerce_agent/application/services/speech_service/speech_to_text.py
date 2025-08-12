import tempfile
import os

from ecommerce_agent.config import settings
from groq import AsyncGroq

class SpeechToTextService:
  def __init__(self):
    self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)
    
  async def transcribe(self, audio_bytes: bytes) -> str:
    if not audio_bytes:
      raise ValueError("Audio bytes are required")
    
    try:
      with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(audio_bytes)
        temp_file_path = temp_file.name
        try:
          with open(temp_file_path, "rb") as audio_file:
            transcription = self.client.audio.transcriptions.create(
              file=audio_file,
              model=settings.GROQ_STT_MODEL,
              language="es",
              response_format="text",
            )
            
          if not transcription:
            raise ValueError("No transcription received from the API")
          
          return transcription
        finally:
          os.unlink(temp_file_path)
    except Exception as e:
      raise ValueError(f"Error transcribing audio: {e}")
    