from ecommerce_agent.application.services.conversation_service.workflow.nodes import conversation_node, tools_node
from ecommerce_agent.application.services.conversation_service.workflow.state import ConversationState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition
import logging

def create_graph_workflow() -> StateGraph:
  """
  Creates and configures the LangGraph workflow for the conversation agent.

  This workflow defines the nodes (conversation and tools) and the edges
  that dictate the flow of messages and tool usage within the agent.

  Returns:
    StateGraph: The configured LangGraph workflow.
  """
  logging.info("Creating graph workflow")
  graph = StateGraph(ConversationState)
  
  graph.add_node("conversation", conversation_node)
  graph.add_node("tools", tools_node)
  
  graph.add_conditional_edges(
    "conversation",
    tools_condition
  )
  graph.add_edge("tools", "conversation")
  graph.add_edge(START, "conversation")
  logging.info("Graph workflow created")
  return graph
