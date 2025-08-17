from langfuse import Langfuse
from ecommerce_agent.config import settings

langfuse_client = Langfuse(
  public_key=settings.LANGFUSE_PUBLIC_KEY,
  secret_key=settings.LANGFUSE_SECRET_KEY,
  host=settings.LANGFUSE_HOST
)

class Prompt:
  def __init__(self, name: str):
    self.name = name
    self.prompt = self._get_prompt()
    
    def __str__(self) -> str:
      return self.prompt
    
    def __repr__(self) -> str:
      return self.__str__()
    
  def _get_prompt(self) -> str:
    prompt = langfuse_client.get_prompt(self.name)
    return prompt.prompt

SYSTEM_PROMPT = Prompt(name="default_prompt")
IMAGE_PROMPT = Prompt(name="image_prompt")
SUMMARY_PROMPT = Prompt(name="summary_prompt")
EXTENDED_SYSTEM_PROMPT = Prompt(name="extended_system_prompt")