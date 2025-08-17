from typing import Any
from langgraph.prebuilt import ToolNode
from langchain_core.messages import ToolMessage, RemoveMessage
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
  last_message = state['messages'][-1]
  tool_message = state['messages'][-2]
  if isinstance(tool_message, ToolMessage):
    tool_result = tool_message.content
    
    input_message = f"The result of the tool call is: {tool_result}. The AI Agent response is: {last_message.content}"
    
    response_chain = get_response_chain(system_prompt=IMAGE_PROMPT.prompt, with_structured_output=True)
    logging.info("Response chain successfully obtained for image node.")
    response = await response_chain.ainvoke(
      {
        "messages": [input_message]
      }
    )
    logging.info("Response chain invoked for image node.")
    return {"messages": str(response)}
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
  delete_messages = [
    RemoveMessage(id=message.id)
    for message in state['messages'][:-settings.SUMMARY_MESSAGE_COUNT]
  ]
  return {"summary": summary.content, "messages": delete_messages}

async def connector_node(state: ConversationState):
  return {}