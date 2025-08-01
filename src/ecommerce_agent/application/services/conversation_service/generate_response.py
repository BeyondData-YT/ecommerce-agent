from typing import AsyncGenerator, Union, Any
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from ecommerce_agent.application.services.conversation_service.workflow.state import ConversationState
from ecommerce_agent.application.services.conversation_service.workflow.graph import create_graph_workflow
from ecommerce_agent.config import settings
import logging

langfuse_client = Langfuse(
  public_key=settings.LANGFUSE_PUBLIC_KEY,
  secret_key=settings.LANGFUSE_SECRET_KEY,
  host=settings.LANGFUSE_HOST
)

langfuse_handler = CallbackHandler(
  public_key=settings.LANGFUSE_PUBLIC_KEY
)

if langfuse_client.auth_check():
    logging.info("Langfuse client is authenticated and ready!")
else:
    logging.error("Authentication failed. Please check your credentials and host.")

async def generate_response(
  messages: Union[str, list[dict[str, Any]]]
  ) -> tuple[str, ConversationState]:
  """
  Generates a response from the conversation graph.

  Args:
    messages (Union[str, list[dict[str, Any]]]): The input messages, either a single string or a list of message dictionaries.

  Returns:
    tuple[str, ConversationState]: A tuple containing the content of the last message and the complete conversation state.
  """
  graph = create_graph_workflow()
  try:
    graph = graph.compile()
    logging.info("Graph workflow compiled successfully.")
    output_state = await graph.ainvoke(
      input={
        "messages": __format_messages(messages=messages)
      },
      config={
        "configurable": {
          "thread_id": "123"
        },
        "callbacks": [langfuse_handler]
      }
    )
    logging.info("Graph invoked")
    last_message = output_state["messages"][-1]
    return last_message.content, ConversationState(**output_state)
  except Exception as e:
    logging.error(f"Error generating response: {e}")
    raise e
  
async def get_streaming_response(
  messages: Union[str, list[dict[str, Any]]]
) -> AsyncGenerator[str, None]:
  """
  Generates a streaming response from the conversation graph.

  Args:
    messages (Union[str, list[dict[str, Any]]]): The input messages, either a single string or a list of message dictionaries.

  Yields:
    str: Chunks of the AI's response content.
  """
  graph = create_graph_workflow()
  try:
    graph = graph.compile()
    logging.info("Graph workflow compiled successfully for streaming.")
    async for chunk in graph.astream(
      input={
        "messages": __format_messages(messages=messages)
      },
      config={
        "configurable": {
          "thread_id": "123"
        },
        "callbacks": [langfuse_handler]
      },
      stream_mode="messages"
    ):
      logging.info("Streaming response")
      if chunk[1]['langgraph_node'] == 'conversation' and isinstance(chunk[0], AIMessageChunk):
        yield chunk[0].content
  except Exception as e:
    logging.error(f"Error generating streaming response: {e}")
    raise e
  
def __format_messages(messages: Union[str, list[Union[str, dict[str, Any]]]]) -> list[Union[HumanMessage, AIMessage]] :
  """
  Formats various message inputs into a consistent list of HumanMessage or AIMessage objects.

  Args:
    messages (Union[str, list[Union[str, dict[str, Any]]]]): The input messages. Can be a single string, a list of strings, or a list of dictionaries with 'role' and 'content'.

  Returns:
    list[Union[HumanMessage, AIMessage]]: A list of formatted LangChain message objects.

  Raises:
    ValueError: If an invalid message role is encountered in the input dictionaries.
  """
  logging.info(f"Formatting messages: {messages}")
  if isinstance(messages, str):
    return [HumanMessage(content=messages)]
  
  if isinstance(messages, list):
    if not messages:
      logging.warning("No messages provided for formatting.")
      return []
    
    # Check if the first message is a dictionary (for role-based messages)
    if (isinstance(messages[0], dict)
        and 'role' in messages[0]
        and 'content' in messages[0]
      ):
        result = []
        for message in messages:
          if message['role'] == 'user':
            result.append(HumanMessage(content=message['content']))
          elif message['role'] == 'assistant':
            result.append(AIMessage(content=message['content']))
          else:
            logging.error(f"Invalid message role encountered: {message['role']}")
            raise ValueError(f"Invalid message role: {message['role']}")
        return result
      
    # Assume it's a list of strings if not a list of dictionaries
    return [HumanMessage(content=message) for message in messages if isinstance(message, str)]
  
  logging.warning(f"Unsupported message format: {type(messages)}. Returning empty list.")
  return []