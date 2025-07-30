from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq

from ecommerce_agent.application.services.conversation_service.workflow.tools import tools 
from ecommerce_agent.config import settings
from ecommerce_agent.domain.prompt import SYSTEM_PROMPT
import logging

def get_llm(temperature: float = 0.0, model_name:str = settings.GROQ_LLM_MODEL) -> ChatGroq:
  """
  Initializes and returns a ChatGroq language model.

  Args:
    temperature (float): The temperature for the model's output randomness. Defaults to 0.0.
    model_name (str): The name of the Groq model to use. Defaults to settings.GROQ_LLM_MODEL.

  Returns:
    ChatGroq: An initialized ChatGroq language model.
  """
  logging.info(f"Getting LLM with model: {model_name}")
  return ChatGroq(
    model=model_name,
    temperature=temperature,
    api_key=settings.GROQ_API_KEY
  )

def get_response_chain() -> Runnable:
  """
  Creates and returns a LangChain response chain.

  This chain consists of a ChatGroq language model bound with tools and a ChatPromptTemplate.

  Returns:
    Runnable: A LangChain prompt-to-model runnable.
  """
  llm = get_llm()
  logging.info("LLM obtained")
  llm = llm.bind_tools(tools)
  prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT.prompt),
    MessagesPlaceholder(variable_name="messages"),
    
  ],
  template_format='jinja2'
  )
  logging.info("Prompt obtained")
  return prompt | llm





