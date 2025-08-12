from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from ecommerce_agent.config import settings
import asyncio
import platform

if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class SetupShortLongMemory:
  def __init__(self):
    pass
  async def setup_memory(self):
    async with (
      AsyncPostgresSaver.from_conn_string(
        conn_string=settings.POSTGRES_URI
      ) as checkpointer,
      AsyncPostgresStore.from_conn_string(
        conn_string=settings.POSTGRES_URI
      ) as store
    ):
      await store.setup()
      await checkpointer.setup()
      
setup_short_long_memory = SetupShortLongMemory()
asyncio.run(setup_short_long_memory.setup_memory())
