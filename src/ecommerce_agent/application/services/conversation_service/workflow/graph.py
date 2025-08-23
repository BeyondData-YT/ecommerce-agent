from ecommerce_agent.application.services.conversation_service.workflow.nodes import conversation_node, tools_node, audio_node, connector_node, image_node, summary_node, memory_node, memory_tools_node
from ecommerce_agent.application.services.conversation_service.workflow.edges import select_workflow, should_summarize
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
  
  graph.add_node("memory_node", memory_node)
  graph.add_node("conversation_node", conversation_node)
  graph.add_node("tools", tools_node)
  graph.add_node("memory_tools", memory_tools_node)
  graph.add_node("audio_node", audio_node)
  graph.add_node("image_node", image_node)
  graph.add_node("connector_node", connector_node)
  graph.add_node("summary_node", summary_node)
  
  graph.add_edge(START, "memory_node")
  graph.add_conditional_edges(
    "memory_node",
    tools_condition,
    {
      "tools": "memory_tools",
      "__end__": "conversation_node"
    }
  )
  graph.add_edge("memory_tools", "memory_node")
  graph.add_conditional_edges(
    "conversation_node",
    tools_condition,
    {
      "tools": "tools",
      "__end__": "connector_node"
    }
  )
  graph.add_edge("tools", "conversation_node")
  graph.add_conditional_edges(
    "connector_node",
    select_workflow
  )
  graph.add_conditional_edges(
    "audio_node",
    should_summarize
  )
  graph.add_conditional_edges(
    "image_node",
    should_summarize
  )
  graph.add_edge("summary_node", END)
  logging.info("Graph workflow created successfully.")
  return graph
