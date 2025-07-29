from ecommerce_agent.application.services.conversation_service.workflow.nodes import conversation_node, connector_node, tools_node
from ecommerce_agent.application.services.conversation_service.workflow.state import ConversationState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition
import logging

def create_graph_workflow():
  logging.info("Creating graph workflow")
  graph = StateGraph(ConversationState)
  
  graph.add_node("conversation", conversation_node)
  # graph.add_node("connector", connector_node)
  graph.add_node("tools", tools_node)
  
  graph.add_conditional_edges(
    "conversation",
    tools_condition
  )
  # graph.add_edge("connector", END)
  graph.add_edge("tools", "conversation")
  graph.add_edge(START, "conversation")
  logging.info("Graph workflow created")
  return graph
