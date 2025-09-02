import os
import pytest
from pathlib import Path
from src.ingest.build_index import build_index
from src.core.application.embedding_client import load_embeddings
from src.core.application.retriever import chroma_persistent

LLM_INSTRUCTION_FILE_ENV = "LLM_INSTRUCTION_FILE"
DEFAULT_LLM_INSTRUCTION_FILE = "data/instructions/llm.jsonl"


def test_llm_instructions_are_ingested():
    llm_path = Path(os.getenv(LLM_INSTRUCTION_FILE_ENV, DEFAULT_LLM_INSTRUCTION_FILE))
    if not llm_path.exists():
        pytest.skip("llm.jsonl not present")

    build_index()

    embeddings = load_embeddings()
    store = chroma_persistent(embeddings)

    collection = getattr(store, "_collection", None)
    assert collection is not None

    result = collection.get(where={"type": "llm_instruction"}, include=["metadatas", "documents"]) or {}
    ids = result.get("ids", [])
    assert len(ids) > 0
