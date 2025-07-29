from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

from ecommerce_agent.application.services.conversation_service.workflow.tools import tools # TODO: Cambiar a las tools
from ecommerce_agent.config import settings
from ecommerce_agent.domain.prompt import SYSTEM_PROMPT
import logging

def get_llm(temperature: float = 0.0, model_name:str = settings.GROQ_LLM_MODEL):
  logging.info(f"Getting LLM with model: {model_name}")
  return ChatGroq(
    model=model_name,
    temperature=temperature,
    api_key=settings.GROQ_API_KEY
  )

def get_response_chain():
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





