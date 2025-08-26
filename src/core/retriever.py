import os
from pathlib import Path
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

VECTORS_DIR = Path(os.getenv("DATA_DIR", "./data")) / "vectors"
VECTORS_DIR.mkdir(parents=True, exist_ok=True)

def chroma_from_documents(docs: List[Document], embeddings):
    return Chroma.from_documents(
        docs, embeddings, persist_directory=str(VECTORS_DIR / "chroma")
    )

def chroma_persistent(embeddings):
    return Chroma(
        persist_directory=str(VECTORS_DIR / "chroma"),
        embedding_function=embeddings
    )

def build_metadata_filter(prepared: bool | None = None, english_min: str | None = None) -> Dict[str, Any] | None:
    meta: Dict[str, Any] = {}
    if prepared is not None:
        meta["prepared"] = prepared
    if english_min:
        lvl_map = {"A1":1,"A2":2,"B1":3,"B2":4,"C1":5,"C2":6}
        meta["english_level_num_min"] = lvl_map.get(english_min.upper(), 0)
    return meta or None
