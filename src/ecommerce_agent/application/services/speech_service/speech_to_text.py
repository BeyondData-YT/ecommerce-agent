from ecommerce_agent.config import settings
from groq import AsyncGroq

class SpeechToTextService:
  def __init__(self):
    self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)
    
  async def transcribe(self, audio_bytes: bytes) -> str:
    if not audio_bytes:
      raise ValueError("Audio bytes are required")
    
    try:
      audio_file_tuple = ("audio.wav", audio_bytes)

      transcription = await self.client.audio.transcriptions.create(
        file=audio_file_tuple,
        model=settings.GROQ_STT_MODEL,
        language="es",
        response_format="text",
      )

      if not transcription:
        raise ValueError("No transcription received from the API")
      
      return transcription
    except Exception as e:
      raise ValueError(f"Error transcribing audio: {e}")
    