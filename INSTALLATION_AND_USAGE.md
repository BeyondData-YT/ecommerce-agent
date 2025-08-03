# Installation and Usage

This document provides instructions on how to set up and run the Ecommerce Agent project, including its dependencies and available scripts.

## Dependencies

The project is built with Python 3.11+ and relies on the following key dependencies, as listed in `pyproject.toml`:

*   `fastapi`: For building the web API.
*   `langchain[groq]`: For integrating with Groq LLMs.
*   `langchain-community`: General LangChain utilities.
*   `langchain-text-splitters`: For splitting text documents.
*   `langfuse`: For observability and tracing.
*   `langgraph`: For defining conversational workflows.
*   `psycopg2-binary`: PostgreSQL adapter.
*   `pydantic` & `pydantic-settings`: For data validation and settings management.
*   `pypdf`: For PDF document loading.
*   `python-dotenv`: For managing environment variables.
*   `python-telegram-bot`: For Telegram bot integration.
*   `sentence-transformers`: For generating document embeddings.
*   `uvicorn`: ASGI web server for FastAPI.

You can install these dependencies using `uv`:

```bash
uv sync
```

## Environment Variables

Create a `.env` file in the root directory of the project and populate it with the variables as in `env_example.env`:

## Running the Project

The project can be run using Docker Compose for a full-stack setup.

### Using Docker Compose

Docker Compose sets up all necessary services, including PostgreSQL (ParadeDB for vector search), Langfuse, Minio, and Redis.

To start the services:

```bash
docker-compose up -d
```

This command will:

* Start the `db` service (PostgreSQL with ParadeDB and pg_search extensions).
* Start the `langfuse-worker` and `langfuse-web` services (Langfuse for observability).
* Start `clickhouse`, `minio`, and `redis` services (dependencies for Langfuse).
* The FastAPI application within the `ecommerce_agent` will connect to these services.

### Telegram Bot Integration

If you have configured the `TELEGRAM_BOT_TOKEN` and `WEBHOOK_URL` in your `.env` file, the FastAPI application will automatically set up the Telegram webhook on startup.

Alternatively, for local testing without a public webhook, you can run the Telegram bot in polling mode:

```bash
uv run  src/ecommerce_agent/infrastructure/messaging/telegram/telegram_bot_handler.py
```

## Scripts

### `ingest_documents_table.py`

This script is used to load documents from a specified directory, split them into chunks, generate embeddings, and ingest them into the PostgreSQL database.

**Usage:**

```bash
uv run scripts/ingest_documents_table.py --directory <path_to_document_directory>
```

* `<path_to_document_directory>`: The directory containing the documents (e.g., `.txt`, `.pdf`) you wish to ingest. If not provided, it defaults to `data/faqs/`.

**Example:**

To ingest documents from the default `data/faqs/` directory:

```bash
uv run scripts/ingest_documents_table.py
```