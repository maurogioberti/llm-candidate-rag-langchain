# Candidate RAG (LangChain + FastAPI)

Proof of Concept to index CVs (JSON) and query them using RAG.

## Quickstart

```bash
# 1) Create virtual environment
python -m venv .venv

# 2) Activate it
.\.venv\Scripts\Activate.ps1   # PowerShell (Windows)
# or: source .venv/bin/activate  # Linux/Mac

# 3) Install dependencies
pip install -e .

# 4) Configure environment variables
cp .env.example .env

# 5) Build the index (reads /data/input/*.json)
python -m src.ingest.build_index

# 6) Run the API
uvicorn src.api.main:app --reload --port 8080

# 7) Health check
curl http://localhost:8080/health

## API Docs
OpenAPI (via Swagger UI): http://localhost:8080/docs