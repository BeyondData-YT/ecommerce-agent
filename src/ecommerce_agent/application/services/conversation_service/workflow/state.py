from langgraph.graph import MessagesState

class ConversationState(MessagesState):
  workflow: str
  summary: str
  memories: list[str]
  audio_buffer: bytes