from langgraph.graph import MessagesState

class ConversationState(MessagesState):
  workflow: str
  summary: str
  audio_buffer: bytes