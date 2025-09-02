import os
import json
from pathlib import Path
from typing import List
from langchain_core.documents import Document

from src.core.domain.candidate import CandidateRecord
from src.core.application.embedding_client import load_embeddings
from src.core.application.retriever import chroma_from_documents
from src.core.infrastructure.embeddings import load_instruction_pairs

DEFAULT_DATA_DIR = "./data"
INPUT_SUBDIR = "input"
DEFAULT_EMBEDDING_INSTRUCTION_FILE = "data/instructions/embedings.jsonl"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 60
ENGLISH_LEVEL_MAP = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}

DATA_DIR = Path(os.getenv("DATA_DIR", DEFAULT_DATA_DIR))
INPUT_DIR = DATA_DIR / INPUT_SUBDIR

__all__ = ["to_documents", "load_candidate_records", "build_index", "build_index_from_records"]

def _english_to_num(level: str) -> int:
    return ENGLISH_LEVEL_MAP.get((level or "UNK").upper(), 0)

def load_candidate_records() -> list:
    return _load_candidate_records_from_dir(INPUT_DIR)

def _load_candidate_records_from_dir(input_dir: Path) -> list:
    candidate_records = []
    for file_path in sorted(input_dir.glob("*.json")):
        data = json.loads(file_path.read_text(encoding="utf-8"))
        candidate_id = data.get("GeneralInfo", {}).get("CandidateId") or file_path.stem
        candidate_records.append(CandidateRecord(
            candidate_id=candidate_id,
            raw=data,
            summary=data.get("Summary", ""),
            skills=data.get("SkillMatrix", []) or [],
            languages=data.get("Languages", []) or [],
            scores=data.get("Scores", {}) or {}
        ))
    return candidate_records

def to_documents(records: list) -> list:
    docs = []
    for record in records:
        docs.extend(_candidate_to_documents(record))
    return _split_documents(docs)

def _candidate_to_documents(candidate: CandidateRecord) -> list:
    from langchain_core.documents import Document
    documents = []
    for block in candidate.to_text_blocks():
        documents.append(Document(
            page_content=block,
            metadata={
                "candidate_id": candidate.candidate_id,
                "prepared": candidate.prepared,
                "english_level": candidate.english_level,
                "english_level_num": _english_to_num(candidate.english_level),
            }
        ))
    return documents

def _split_documents(documents: list) -> list:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except Exception:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(documents)

def build_index() -> dict:
    records = load_candidate_records()
    docs = to_documents(records)

    instr_path = Path(os.getenv("EMBEDING_INSTRUCTION_FILE", DEFAULT_EMBEDDING_INSTRUCTION_FILE))
    if instr_path.exists():
        docs += _load_and_split_instruction_docs(instr_path)

    emb = load_embeddings()
    chroma_from_documents(docs, emb)
    return {"candidates": len(records), "chunks": len(docs)}

def _load_and_split_instruction_docs(instr_path: Path) -> list:
    from langchain_core.documents import Document
    pairs = load_instruction_pairs(instr_path)
    extra_docs = [Document(page_content=text, metadata=meta) for (text, meta) in pairs]
    return _split_documents(extra_docs)

def build_index_from_records(records: list):
    emb = load_embeddings()
    docs = to_documents(records)
    return chroma_from_documents(docs, emb)