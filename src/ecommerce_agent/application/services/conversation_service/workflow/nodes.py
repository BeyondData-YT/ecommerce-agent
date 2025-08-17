from typing import Any
import re
import json
from langgraph.prebuilt import ToolNode
from langchain_core.messages import ToolMessage, HumanMessage
from ecommerce_agent.application.services.conversation_service.workflow.chains import get_response_chain, get_conversation_summary_chain
from ecommerce_agent.application.services.conversation_service.workflow.tools import tools
from ecommerce_agent.application.services.conversation_service.workflow.state import ConversationState
from ecommerce_agent.application.services.speech_service.text_to_speech import TextToSpeechService
from ecommerce_agent.config import settings
from ecommerce_agent.domain.prompts import IMAGE_PROMPT
import logging

tools_node = ToolNode(tools)

async def conversation_node(state: ConversationState) -> dict[str, Any]:
  """
  Processes the conversation state and generates a response using the response chain.

  Args:
    state (ConversationState): The current conversation state, including messages.

  Returns:
    dict: A dictionary containing the updated messages from the response chain.
  """
  response_chain = get_response_chain()
  summary = state.get("summary", "")
  logging.info("Response chain successfully obtained for conversation node.")
  response = await response_chain.ainvoke(
    {
      "messages": state['messages'],
      "summary": summary
    }
  )
  logging.info("Response chain invoked for conversation node.")
  return {"messages": response}

async def audio_node(state: ConversationState) -> dict[str, Any]:
    """
    Takes the last message from the state and converts it to audio.
    """
    last_message = state['messages'][-1]
    
    if hasattr(last_message, 'content'):
        text_to_synthesize = last_message.content
        
        text_to_speech_service = TextToSpeechService()
        logging.info("Text to speech service successfully obtained for audio node.")
        
        audio_bytes = text_to_speech_service.synthesize(text_to_synthesize)
        logging.info("Audio bytes synthesized for audio node.")
        
        return {"audio_buffer": audio_bytes}

    return {}
  
async def image_node(state: ConversationState) -> dict[str, Any]:
  for message in state['messages']:
    if isinstance(message, ToolMessage):
      if message.name == "image_product_retriever":
        tool_result = message.content
        agent_response = state['messages'][-1].content
        
        input_message = f"The result of the tool call is: {tool_result}. The AI Agent response is: {agent_response}"
        
        response_chain = get_response_chain(system_prompt=IMAGE_PROMPT.prompt, with_structured_output=True)
        logging.info("Response chain successfully obtained for image node.")
        response = await response_chain.ainvoke(
          {
            "messages": [input_message]
          }
        )
        logging.info("Response chain invoked for image node.")

        response_str = str(response)
        logging.info(f"Response string: {response_str}")
        image_urls = re.findall(r"image_url='([^']*)'", response_str)
        captions = re.findall(r"caption='((?:[^'\\]|\\.)*)'", response_str)
        product_names = re.findall(r"product_name='((?:[^'\\]|\\.)*)'", response_str)

        image_responses = [
            {
                "image_url": url,
                "caption": cap,
                "product_name": name
            }
            for url, cap, name in zip(image_urls, captions, product_names)
        ]

        return {"messages": [HumanMessage(content=str(image_responses))]}
  return {}

async def summary_node(state: ConversationState) -> dict[str, Any]:
  summary = state.get("summary", "")
  summary_chain = get_conversation_summary_chain(summary, model_name=settings.GROQ_LLM_MODEL_CONTEXT_SUMMARY)
  logging.info("Summary chain successfully obtained for summary node.")
  summary = await summary_chain.ainvoke(
    {
      "messages": state['messages'],
      "summary": summary
    }
  )
  logging.info("Summary chain invoked for summary node.")
  messages_to_keep = state['messages'][-settings.SUMMARY_MESSAGE_COUNT_TO_KEEP:]
  return {"summary": summary.content, "messages": messages_to_keep}

async def connector_node(state: ConversationState):
  return {}