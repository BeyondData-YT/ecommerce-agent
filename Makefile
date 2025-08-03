ifeq (,$(wildcard .env))
$(error .env file is missing at .env. Please create one based on .env.example)
endif

# Infrastructure
infrastructure-build:
	docker compose build

infrastructure-up:
	docker compose up --build -d

infrastructure-stop:
	docker compose stop

# Pipelines
ingest-documents-table:
	uv run .\scripts\ingest_documents_table.py --directory '.\data\faqs\'

