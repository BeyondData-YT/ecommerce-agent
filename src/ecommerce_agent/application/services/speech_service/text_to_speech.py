import os

from ecommerce_agent.config import settings

from elevenlabs import ElevenLabs, VoiceSettings

class TextToSpeechService:
  def __init__(self):
    self.client = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)
    
  def synthesize(self, text: str) -> bytes:
    if not text:
      raise ValueError("Text is required")
    
    if len(text) > 3000:
      raise ValueError("Text must be less than 3000 characters")
    
    try:
      audio = self.client.text_to_speech.convert(
        text=text,
        voice_id=settings.ELEVENLABS_VOICE_ID,
        model_id=settings.ELEVENLABS_MODEL_ID,
        voice_settings=VoiceSettings(
          stability=0.5,
          similarity_boost=0.75,
          speed=1.2
        )
      )

      audio_bytes = b"".join(audio)
      if not audio_bytes:
        raise ValueError("No audio bytes received from the API")
      
      return audio_bytes
    except Exception as e:
      raise ValueError(f"Error synthesizing audio: {e}")
