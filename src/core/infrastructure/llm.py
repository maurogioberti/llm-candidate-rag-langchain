from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import json

def load_llm_instruction_records(path: str | Path = "data/instructions/llm.jsonl") -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    out: List[Dict[str, Any]] = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if "instruction" in rec and "input" in rec and "output" in rec:
            rec["_row_id"] = i
            out.append(rec)
    return out