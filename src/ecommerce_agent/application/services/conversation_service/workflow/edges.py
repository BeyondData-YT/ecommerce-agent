from typing import Literal
from ecommerce_agent.application.services.conversation_service.workflow.state import ConversationState
from langgraph.graph import END

def select_workflow(
    state: ConversationState,
) -> Literal[ "audio_node", "__end__"]:
    workflow = state["workflow"]
    
    if workflow == "audio":
        return "audio_node"
    # elif workflow == "image":
    #     return "image_node"
    else:
        return END