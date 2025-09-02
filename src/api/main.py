from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.ingest.build_index import build_index
from src.core.application.agent import build_chain

APP_TITLE = "Candidate RAG (LangChain)"
ROUTE_HEALTH = "/health"
ROUTE_INDEX = "/index"
ROUTE_CHAT = "/chat"
STATUS_OK = "ok"
PAYLOAD_INPUT = "input"
FIELD_ANSWER = "answer"
FIELD_CONTEXT = "context"
ERROR_PREFIX = "LLM/Index error: "

app = FastAPI(title=APP_TITLE)

class ChatRequest(BaseModel):
    question: str
    filters: dict | None = None

@app.get(ROUTE_HEALTH)
def health():
    return {"status": STATUS_OK}

@app.post(ROUTE_INDEX)
def index():
    info = build_index()
    return {"indexed": info}

@app.post(ROUTE_CHAT)
def chat(req: ChatRequest):
    try:
        chain = build_chain()
        result = chain.invoke({PAYLOAD_INPUT: req.question})
        return {
            FIELD_ANSWER: result.get(FIELD_ANSWER, ""),
            "sources": [getattr(d, "metadata", {}) for d in result.get(FIELD_CONTEXT, [])]
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"{ERROR_PREFIX}{e}")
