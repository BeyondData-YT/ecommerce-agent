from langgraph.graph import MessagesState

class ConversationState(MessagesState):
  workflow_state: str
  summary: str
  audio_buffer: bytes
  image_path: str