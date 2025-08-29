from pathlib import Path
from typing import List
import json
from langchain_core.documents import Document

def load_instruction_pairs(path: str | Path = "data/instructions/embedings.jsonl") -> List[Document]:
    p = Path(path)
    if not p.exists():
        return []
    docs: List[Document] = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        q = rec.get("query")
        pos = rec.get("positive")
        neg = rec.get("negative")
        if q:
            docs.append(Document(page_content=q,  metadata={"type": "query",    "pair_id": i}))
        if pos:
            docs.append(Document(page_content=pos, metadata={"type": "positive", "pair_id": i, "query": q}))
        if neg:
            docs.append(Document(page_content=neg, metadata={"type": "negative", "pair_id": i, "query": q}))
    return docs
