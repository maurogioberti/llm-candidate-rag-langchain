# Candidate RAG (LangChain + FastAPI)

POC para indexar CVs (JSON) y consultar via RAG.

## Quickstart
```bash
# 1) instalar
python -m venv .venv && . .venv/bin/activate
pip install -e .

# 2) configurar
cp .env.example .env

# 3) indexar (lee /data/input/*.json)
python -m src.ingest.build_index

# 4) API
uvicorn src.api.main:app --reload --port 8080
