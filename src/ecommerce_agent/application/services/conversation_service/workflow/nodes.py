from typing import Any
import re
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage, SystemMessage
from ecommerce_agent.application.services.conversation_service.workflow.chains import get_response_chain, get_conversation_summary_chain, get_memory_chain
from ecommerce_agent.application.services.conversation_service.workflow.tools import tools, memory_tools
from ecommerce_agent.application.services.conversation_service.workflow.state import ConversationState
from ecommerce_agent.application.services.speech_service.text_to_speech import TextToSpeechService
from ecommerce_agent.application.services.memory import MemoryService
from ecommerce_agent.config import settings
from ecommerce_agent.domain.prompts import IMAGE_PROMPT
from ecommerce_agent.application.services.guardrails.input_guardrail import InputGuardrail
from ecommerce_agent.application.services.guardrails.output_guardrails import OutputGuardrail
import logging

tools_node = ToolNode(tools)
memory_tools_node = ToolNode(memory_tools)

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
  memories = state.get("memories", [])
  logging.info("Response chain successfully obtained for conversation node.")
  try:
    response = await response_chain.ainvoke(
    {
      "messages": state['messages'],
      "summary": summary,
      "memories": memories
    }
    )
    logging.info("Response chain invoked for conversation node.")
    return {"messages": response}
  except Exception as e:
    logging.error(f"Error invoking response chain: {e}")
    return {"messages": [AIMessage(content=f"Error: {e}")]}

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
  memories = state.get("memories", [])
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
            "messages": [input_message],
            "memories": memories
          }
        )
        logging.info("Response chain invoked for image node.")

        response_str = str(response)
        image_responses = handle_mixed_input(response_str)

        return {"messages": [HumanMessage(content=str(image_responses))]}
  return {}

def clean_and_parse_response_string(response_str):
    """
    Cleans and parses a string that contains ImageResponse objects.
      
    Args:
        response_str (str): The string to clean and parse.
        
    Returns:
        list: A list of dictionaries with the parsed data.
    """
    logging.basicConfig(level=logging.INFO)

    cleaned_str = re.sub(r'image_responses=\[|ImageResponse\(|\)\]', '', response_str).strip()

    
    cleaned_str = cleaned_str.replace('\\"', '"').replace("\\'", "'")
    
    
    regex = re.compile(
        r"ImageResponse\(image_url='(.*?)', caption='(.*?)', product_name='(.*?)'\)"
    )
    matches = regex.findall(response_str)
    
    if not matches:
        regex_alt = re.compile(
            r'ImageResponse\(image_url="(.*?)", caption="(.*?)", product_name="(.*?)"\)'
        )
        matches = regex_alt.findall(response_str)

    if not matches:
        logging.error("No matches found in the response string.")
        return []
        
    parsed_list = []
    for match in matches:
        image_url, caption, product_name = match
        parsed_list.append({
            "image_url": image_url,
            "caption": caption,
            "product_name": product_name
        })

    return parsed_list

def handle_mixed_input(input_str):
    """
    Handles inputs that can be either the ImageResponse string or the list of dictionaries.
    """
    if input_str.strip().startswith('['):
        try:
            cleaned_str = input_str.replace("'", '"').replace('"', '\\"').replace('\\\\"', '\\"')
            
            # Try to parse as a list of dictionaries first
            pattern = re.compile(r"\{'image_url': '(.*?)', 'caption': '(.*?)', 'product_name': '(.*?)'\}")
            matches = pattern.findall(input_str)
            
            if not matches:
                # If the simple quotes pattern fails, try double quotes
                pattern = re.compile(r'\{"image_url": "(.*?)", "caption": "(.*?)", "product_name": "(.*?)"\}')
                matches = pattern.findall(input_str)

            if matches:
                result = []
                for match in matches:
                    image_url, caption, product_name = match
                    result.append({
                        "image_url": image_url,
                        "caption": caption.replace('\\"', '"').replace("'", '"'),
                        "product_name": product_name
                    })
                return result
            
        except Exception as e:
            logging.error(f"Error trying to parse as list of dictionaries: {e}")
    
    # If the string doesn't start with '[{' or the first attempt fails, use the ImageResponse parser
    return clean_and_parse_response_string(input_str)

async def summary_node(state: ConversationState) -> dict[str, Any]:
  summary = state.get("summary", "")
  logging.info(f"Number of messages to summarize: {len(state['messages'])}")
  
  summary_chain = get_conversation_summary_chain(summary, model_name=settings.GROQ_LLM_MODEL_CONTEXT_SUMMARY)
  logging.info("Summary chain successfully obtained for summary node.")
  
  try:
    input_data = {
      "messages": state['messages'],
      "summary": summary
    }    
    summary_response = await summary_chain.ainvoke(input_data)
    
    if hasattr(summary_response, 'content'):
      summary_content = summary_response.content
      if not summary_content or summary_content.strip() == "":
        logging.warning("Summary content is empty or whitespace only")
        summary_content = "No summary generated"
    else:
      logging.warning(f"Summary response does not have 'content' attribute: {summary_response}")
      summary_content = str(summary_response) if summary_response else "No summary generated"
    
  except Exception as e:
    logging.error(f"Error invoking summary chain: {e}")
    return {"messages": [AIMessage(content=f"Error: {e}")]}
  
  messages_to_keep = state['messages'][-settings.SUMMARY_MESSAGE_COUNT_TO_KEEP:]
  logging.info(f"Keeping {len(messages_to_keep)} messages out of {len(state['messages'])} total")
  
  result = {"summary": summary_content, "messages": messages_to_keep}
  return result

async def memory_node(state: ConversationState, config: RunnableConfig) -> dict[str, Any]:
  memory_service = MemoryService()
  memories = await memory_service.get_memories(config)
  logging.info(f"Memories obtained for memory node: {memories}")
  memory_chain = get_memory_chain()
  logging.info("Memory chain successfully obtained for memory node.")
  try:
    response = await memory_chain.ainvoke(
      {
        "messages": state['messages'],
        "memories": memories
      }
    )
    logging.info("Memory chain invoked for memory node.")
    return {"messages": response, "memories": memories}
  except Exception as e:
    logging.error(f"Error invoking memory chain: {e}")
    return {"messages": [AIMessage(content=f"Error: {e}")]}

async def input_guardrail_node(state: ConversationState) -> dict[str, Any]:
  input_guardrail = InputGuardrail()
  logging.info("Input guardrail successfully obtained for input guardrail node.")
  last_message = state['messages'][-1].content
  try:
    return {"messages": input_guardrail.validate_input(last_message), "prompt": last_message}
  except ValueError as e:
    logging.error(f"Error validating input: {e}")
    return {"messages" : [SystemMessage(content=f"Text was not valid")]}
  except Exception as e:
    logging.error(f"Error invoking input guardrail: {e}")
    return {"messages": [AIMessage(content=f"Error: {e}")], "prompt": last_message}

async def output_guardrail_node(state: ConversationState) -> dict[str, Any]:
  output_guardrail = OutputGuardrail()
  logging.info("Output guardrail successfully obtained for output guardrail node.")
  prompt = state.get("prompt", "")
  last_message = state['messages'][-1].content
  try:
    return {"messages": output_guardrail.validate_output(prompt, last_message)}
  except Exception as e:
    logging.error(f"Error validating output: {e}")
    return {"messages": [AIMessage(content=f"Error: {e}")]}

async def connector_node(state: ConversationState):
  return {}