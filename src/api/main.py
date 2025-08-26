from fastapi import FastAPI
from pydantic import BaseModel
from ingest.build_index import build_index
from chat.agent import build_chain

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
    chain = build_chain()
    result = chain({"query": req.question})
    return {
        "answer": result["result"],
        "sources": [d.metadata for d in result["source_documents"]]
    }
