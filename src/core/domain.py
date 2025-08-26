from __future__ import annotations
from pydantic import BaseModel
from typing import Any, Dict, List
import json

class CandidateRecord(BaseModel):
    candidate_id: str
    raw: Dict[str, Any]
    summary: str
    skills: List[Dict[str, Any]] = []
    languages: List[Dict[str, str]] = []
    scores: Dict[str, Any] = {}

    @property
    def prepared(self) -> bool:
        score = self.scores.get("GeneralScore", 0) or 0
        seniority = (self.raw.get("GeneralInfo", {}) or {}).get("Seniority", "").lower()
        return score >= 60 or seniority in {"mid", "senior", "staff"}

    @property
    def english_level(self) -> str:
        lvl_map = {"a1":1,"a2":2,"b1":3,"b2":4,"c1":5,"c2":6}
        best = 0
        for l in self.languages or []:
            if str(l.get("Language","")).lower() in {"english","inglés","en"}:
                lv = lvl_map.get(str(l.get("Proficiency","")).lower(), 0)
                best = max(best, lv)
        inv = {v:k.upper() for k,v in lvl_map.items()}
        return inv.get(best,"UNKNOWN")

    def to_text_blocks(self) -> List[str]:
        blocks = []
        blocks.append(f"[Candidate] {self.candidate_id}\nSummary:\n{self.summary}")
        if self.skills:
            skills = ", ".join([s.get("SkillName","") for s in self.skills])
            blocks.append(f"Skills: {skills}")
        if self.languages:
            langs = ", ".join([f"{l.get('Language')}:{l.get('Proficiency')}" for l in self.languages])
            blocks.append(f"Languages: {langs}")
        blocks.append(json.dumps(self.raw, ensure_ascii=False))
        return blocks
