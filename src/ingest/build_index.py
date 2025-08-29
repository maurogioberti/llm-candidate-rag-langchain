import os
import json
from pathlib import Path
from typing import List
from langchain_core.documents import Document

from core.domain import CandidateRecord
from core.embedding_client import load_embeddings
from core.retriever import chroma_from_documents
from infra.embeddings import load_instruction_pairs  # <- JSONL loader

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
INPUT_DIR = DATA_DIR / "input"

__all__ = ["to_documents", "load_candidate_records", "build_index", "build_index_from_records"]


def _english_to_num(level: str) -> int:
    m = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}
    return m.get((level or "UNK").upper(), 0)


def load_candidate_records() -> List[CandidateRecord]:
    recs: List[CandidateRecord] = []
    for fp in sorted(INPUT_DIR.glob("*.json")):
        data = json.loads(fp.read_text(encoding="utf-8"))
        cid = data.get("GeneralInfo", {}).get("CandidateId") or fp.stem
        recs.append(CandidateRecord(
            candidate_id=cid,
            raw=data,
            summary=data.get("Summary", ""),
            skills=data.get("SkillMatrix", []) or [],
            languages=data.get("Languages", []) or [],
            scores=data.get("Scores", {}) or {}
        ))
    return recs


def to_documents(records: List[CandidateRecord]) -> List[Document]:
    docs: List[Document] = []
    for r in records:
        for block in r.to_text_blocks():
            docs.append(Document(
                page_content=block,
                metadata={
                    "candidate_id": r.candidate_id,
                    "prepared": r.prepared,
                    "english_level": r.english_level,
                    "english_level_num": _english_to_num(r.english_level),
                }
            ))
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except Exception:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
    return splitter.split_documents(docs)


def build_index() -> dict:
    records = load_candidate_records()
    docs = to_documents(records)

    instr_path = Path(os.getenv("EMBEDING_INSTRUCTION_FILE", "data/instructions/embedings.jsonl"))
    if instr_path.exists():
        instr_docs = load_instruction_pairs(instr_path)
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except Exception:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
        docs += splitter.split_documents(instr_docs)

    emb = load_embeddings()
    vs = chroma_from_documents(docs, emb)
    vs.persist()
    return {"candidates": len(records), "chunks": len(docs)}


def build_index_from_records(records: List[CandidateRecord]):
    emb = load_embeddings()
    docs = to_documents(records)
    return chroma_from_documents(docs, emb)