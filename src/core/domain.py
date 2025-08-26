from __future__ import annotations
from pydantic import BaseModel
from typing import Any, Dict, List
import json, re

class CandidateRecord(BaseModel):
    candidate_id: str
    raw: Dict[str, Any]
    summary: str
    skills: List[Dict[str, Any]] = []
    languages: List[Dict[str, str]] = []
    scores: Dict[str, Any] = {}

    @property
    def prepared(self) -> bool:
        score = (self.scores or {}).get("GeneralScore") or 0
        seniority = ((self.raw.get("GeneralInfo") or {}).get("SeniorityLevel") or "").lower()
        return score >= 60 or any(k in seniority for k in ("mid", "senior", "lead", "staff", "principal"))

    @property
    def english_level(self) -> str:
        lvl_map = {"a1":1,"a2":2,"b1":3,"b2":4,"c1":5,"c2":6}
        texts: List[str] = []

        for l in self.languages or []:
            if str(l.get("Language","")).lower() in {"english", "inglés", "en"}:
                texts.append(str(l.get("Proficiency","")))

        gi = self.raw.get("GeneralInfo") or {}
        if gi.get("EnglishLevel"):
            texts.append(str(gi["EnglishLevel"]))

        if self.raw.get("CleanedResumeText"):
            texts.append(str(self.raw["CleanedResumeText"]))

        best = 0
        for t in texts:
            s = t.lower()
            m = re.search(r"\b([abc][12])\b", s)   # C1, B2, etc.
            if m:
                best = max(best, lvl_map[m.group(1)])
            if "advanced" in s: best = max(best, 5)   # ≈ C1
            if "upper" in s:    best = max(best, 4)   # ≈ B2
            if "intermediate" in s and best < 4: best = max(best, 3)

        inv = {v:k.upper() for k,v in lvl_map.items()}
        return inv.get(best, "UNKNOWN")

    def to_text_blocks(self) -> List[str]:
        blocks: List[str] = []
        gi = self.raw.get("GeneralInfo", {})
        
        titles = " ".join([str(gi.get("TitleDetected","")), str(gi.get("TitlePredicted",""))]).lower()
        title_hint = ""
        if any(k in titles for k in ("backend", ".net", "c#", "asp.net")):
            title_hint = "[HINT] backend-dotnet"
        elif "frontend" in titles or "react" in titles:
            title_hint = "[HINT] frontend"

        text = (self.raw.get("CleanedResumeText") or "") + " " + self.summary
        kws = set()
        if re.search(r"\b\.?net\b", text, re.IGNORECASE): kws.add(".NET")
        if re.search(r"\bc#\b", text, re.IGNORECASE):     kws.add("C#")
        if re.search(r"\basp\.?net\b", text, re.IGNORECASE): kws.add("ASP.NET")
        if re.search(r"\bentity framework\b", text, re.IGNORECASE): kws.add("Entity Framework")
        if re.search(r"\bsql server\b", text, re.IGNORECASE): kws.add("SQL Server")
        if re.search(r"\bazure\b", text, re.IGNORECASE):  kws.add("Azure")

        blocks.append(f"[Candidate] {self.candidate_id} {title_hint}\nSummary:\n{self.summary}")
        if self.skills:
            skills = ", ".join([s.get("SkillName","") for s in self.skills])
            blocks.append(f"Skills: {skills}")
        if kws:
            blocks.append("DerivedKeywords: " + ", ".join(sorted(kws)))

        blocks.append(json.dumps(self.raw, ensure_ascii=False))
        return blocks