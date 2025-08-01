from typing import Any
from langgraph.prebuilt import ToolNode
from ecommerce_agent.application.services.conversation_service.workflow.chains import get_response_chain
from ecommerce_agent.application.services.conversation_service.workflow.tools import tools
from ecommerce_agent.application.services.conversation_service.workflow.state import ConversationState
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
  logging.info("Response chain successfully obtained for conversation node.")
  response = await response_chain.ainvoke(
    {
      "messages": state['messages']
    }
  )
  logging.info("Response chain invoked for conversation node.")
  return {"messages": response}
