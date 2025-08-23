import logging
import uuid
from langchain_core.runnables import RunnableConfig
from langgraph.store.postgres.aio import AsyncPostgresStore
from ecommerce_agent.config import settings

class MemoryService:  
  async def store_memory(self, memory: dict[str, str], config: RunnableConfig):
    async with AsyncPostgresStore.from_conn_string(
      conn_string=settings.POSTGRES_URI
    ) as store:
      memory_id = str(uuid.uuid4())
      namespace = (str(config["configurable"].get("user_id")), "memories")
      await store.aput(
        namespace,
        memory_id,
        memory
      )
      logging.info(f"Memory stored for namespace {namespace} and memory id {memory_id}")
    return f"Memory stored for namespace {namespace} and memory id {memory_id}"
  
  async def get_memories(self, config: RunnableConfig):
    async with AsyncPostgresStore.from_conn_string(
      conn_string=settings.POSTGRES_URI
    ) as store:
      namespace = (str(config["configurable"].get("user_id")), "memories")
      memories = await store.asearch(
        namespace
      )
      return [memory.dict()["value"] for memory in memories]