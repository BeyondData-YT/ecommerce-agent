from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableConfig
from ecommerce_agent.application.services.conversation_service.workflow.chains import get_response_chain
from ecommerce_agent.application.services.conversation_service.workflow.tools import tools
from ecommerce_agent.application.services.conversation_service.workflow.state import ConversationState
import logging

tools_node = ToolNode(tools)

async def conversation_node(state: ConversationState):
  response_chain = get_response_chain()
  logging.info("Response chain obtained")
  response = await response_chain.ainvoke(
    {
      "messages": state['messages']
    }
  )
  logging.info("Response chain invoked")
  return {"messages": response}

async def connector_node(state: ConversationState):
  return {}
