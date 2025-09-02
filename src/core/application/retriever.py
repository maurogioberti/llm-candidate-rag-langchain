import os
from pathlib import Path
from typing import List, Dict, Any
from langchain_chroma import Chroma
from langchain_core.documents import Document

ENV_DATA_DIR = "DATA_DIR"
DEFAULT_DATA_DIR = "./data"
VECTORS_SUBDIR = "vectors"
CHROMA_SUBDIR = "chroma"
ENGLISH_LEVEL_MAP = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}
META_PREPARED = "prepared"
META_ENGLISH_LEVEL_MIN = "english_level_num_min"

BASE_VECTORS_DIR = Path(os.getenv(ENV_DATA_DIR, DEFAULT_DATA_DIR)) / VECTORS_SUBDIR
BASE_VECTORS_DIR.mkdir(parents=True, exist_ok=True)

def chroma_from_documents(docs: List[Document], embeddings):
    return Chroma.from_documents(docs, embeddings, persist_directory=str(BASE_VECTORS_DIR / CHROMA_SUBDIR))

def chroma_persistent(embeddings):
    return Chroma(persist_directory=str(BASE_VECTORS_DIR / CHROMA_SUBDIR), embedding_function=embeddings)

def build_metadata_filter(prepared: bool | None = None, english_min: str | None = None) -> Dict[str, Any] | None:
    metadata: Dict[str, Any] = {}
    if prepared is not None:
        metadata[META_PREPARED] = prepared
    if english_min:
        metadata[META_ENGLISH_LEVEL_MIN] = ENGLISH_LEVEL_MAP.get(english_min.upper(), 0)
    return metadata or None
