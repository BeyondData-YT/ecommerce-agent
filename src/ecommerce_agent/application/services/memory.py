import logging
import uuid
from typing import Tuple
from langgraph.store.postgres.aio import AsyncPostgresStore
from ecommerce_agent.config import settings

async def store_memory(namespace_for_memory: Tuple[str, str], memory: dict[str, str]):
  async with AsyncPostgresStore.from_conn_string(
    conn_string=settings.POSTGRES_URI
  ) as store:
    memory_id = str(uuid.uuid4())
    await store.aput(
      namespace_for_memory,
      memory_id,
      memory
    )
    logging.info(f"Memory stored for namespace {namespace_for_memory} and memory id {memory_id}")
    
async def get_memories(namespace_for_memory: Tuple[str, str]):
  async with AsyncPostgresStore.from_conn_string(
    conn_string=settings.POSTGRES_URI
  ) as store:
    memories = await store.asearch(
      namespace_for_memory
    )
    return [memory.dict()["value"] for memory in memories]