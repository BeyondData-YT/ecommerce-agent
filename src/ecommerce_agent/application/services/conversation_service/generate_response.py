from typing import AsyncGenerator, Union, Any
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
from ecommerce_agent.application.services.conversation_service.workflow.state import ConversationState
from ecommerce_agent.application.services.conversation_service.workflow.graph import create_graph_workflow
import logging

async def generate_response(
  messages: str | list[str]
  ) -> tuple[str, list[str]]:
  graph = create_graph_workflow()
  try:
    graph = graph.compile()
    logging.info("Graph compiled")
    output_state = await graph.ainvoke(
      input={
        "messages": __format_messages(messages=messages)
      },
      config={
        "configurable": {
          "thread_id": "123"
        }
      }
    )
    logging.info("Graph invoked")
    last_message = output_state["messages"][-1]
    return last_message.content, ConversationState(**output_state)
  except Exception as e:
    # logging.error(f"Error generating response: {e}")
    raise e
  
async def get_streaming_response(
  messages: str | list[str]
) -> AsyncGenerator[str, None]:
  graph = create_graph_workflow()
  try:
    graph = graph.compile()
    logging.info("Graph compiled")
    async for chunk in graph.astream(
      input={
        "messages": __format_messages(messages=messages)
      },
      config={
        "configurable": {
          "thread_id": "123"
        }
      },
      stream_mode="messages"
    ):
      logging.info("Streaming response")
      if chunk[1]['langgraph_node'] == 'conversation' and isinstance(chunk[0], AIMessageChunk):
        yield chunk[0].content
      # yield chunk
  except Exception as e:
    # logging.error(f"Error generating streaming response: {e}")
    raise e
  
def __format_messages(messages: Union[str, list[dict[str, Any]]]) -> list[Union[HumanMessage, AIMessage]] :
  if isinstance(messages, str):
    return [HumanMessage(content=messages)]
  
  if isinstance(messages, list):
    if not messages:
      return []
    
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
            raise ValueError(f"Invalid message role: {message['role']}")
        return result
      
    return [HumanMessage(content=message) for message in messages]
  
  return []