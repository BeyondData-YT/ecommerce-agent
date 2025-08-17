from typing import Literal
from ecommerce_agent.application.services.conversation_service.workflow.state import ConversationState
from ecommerce_agent.config import settings

def select_workflow(
    state: ConversationState,
) -> Literal[ "audio_node", "image_node", "summary_node", "__end__"]:
    workflow = state["workflow"]
    
    if workflow == "audio":
        return "audio_node"
    elif workflow == "image":
        return "image_node"
    else:
        if len(state['messages']) > settings.SUMMARY_MESSAGE_COUNT:
            return "summary_node"
        else:
            return "__end__"
    
def should_summarize(state: ConversationState) -> Literal["summary_node", "__end__"]:
    messages = state['messages']
    if len(messages) > settings.SUMMARY_MESSAGE_COUNT:
        return "summary_node"
    else:
        return "__end__"