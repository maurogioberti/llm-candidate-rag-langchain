from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.ingest.build_index import build_index
from src.core.application.agent import build_chain

app = FastAPI(title="Candidate RAG (LangChain)")

class ChatRequest(BaseModel):
    question: str
    filters: dict | None = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/index")
def index():
    info = build_index()
    return {"indexed": info}

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        chain = build_chain()
        result = chain.invoke({"input": req.question})
        
        return {
            "answer": result.get("answer", ""),
            "sources": [getattr(d, "metadata", {}) for d in result.get("context", [])]
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM/Index error: {e}")
