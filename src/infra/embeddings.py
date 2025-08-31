from pathlib import Path
from typing import List, Tuple, Dict, Any
import json

def load_instruction_pairs(path: str | Path = "data/instructions/embedings.jsonl") -> List[Tuple[str, Dict[str, Any]]]:
    p = Path(path)
    if not p.exists():
        return []
    out: List[Tuple[str, Dict[str, Any]]] = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        q = rec.get("query")
        pos = rec.get("positive")
        neg = rec.get("negative")
        if q:
            out.append((q,  {"type": "query",    "pair_id": i}))
        if pos:
            out.append((pos, {"type": "positive", "pair_id": i, "query": q}))
        if neg:
            out.append((neg, {"type": "negative", "pair_id": i, "query": q}))
    return out
