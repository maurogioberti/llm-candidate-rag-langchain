from typing import List, Dict, Any

def rank_by_score(docs: List[Any]) -> List[Dict[str, Any]]:
    """Ranks candidates using metadata with a simple heuristic."""
    def key(d):
        meta = getattr(d, "metadata", {}) or {}
        return (int(meta.get("prepared", False)), int(meta.get("english_level_num", 0)))
    ranked = sorted(docs, key=key, reverse=True)
    out = []
    for d in ranked:
        meta = getattr(d, "metadata", {}) or {}
        out.append({
            "candidate_id": meta.get("candidate_id"),
            "english": meta.get("english_level"),
            "why": "prepared+english heuristic"
        })
    return out
