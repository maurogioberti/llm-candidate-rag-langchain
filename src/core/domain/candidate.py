from __future__ import annotations
from pydantic import BaseModel
from typing import Any, Dict, List
import json, re

ENGLISH_LEVEL_MAP = {"a1": 1, "a2": 2, "b1": 3, "b2": 4, "c1": 5, "c2": 6}
ENGLISH_LEVEL_INV = {v: k.upper() for k, v in ENGLISH_LEVEL_MAP.items()}
ENGLISH_LANGS = {"english", "inglés", "en"}
SENIORITY_KEYWORDS = ("mid", "senior", "lead", "staff", "principal")
PREPARED_SCORE_THRESHOLD = 60
BACKEND_HINT_KEYWORDS = ("backend", ".net", "c#", "asp.net")
FRONTEND_HINT_KEYWORDS = ("frontend", "react")
KEYWORD_PATTERNS = [
    (r"\b\.?net\b", ".NET"),
    (r"\bc#\b", "C#"),
    (r"\basp\.?net\b", "ASP.NET"),
    (r"\bentity framework\b", "Entity Framework"),
    (r"\bsql server\b", "SQL Server"),
    (r"\bazure\b", "Azure"),
]

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
        return score >= PREPARED_SCORE_THRESHOLD or any(k in seniority for k in SENIORITY_KEYWORDS)

    @property
    def english_level(self) -> str:
        return self._extract_english_level()

    def _extract_english_level(self) -> str:
        texts: List[str] = []
        for lang in self.languages or []:
            if str(lang.get("Language", "")).lower() in ENGLISH_LANGS:
                texts.append(str(lang.get("Proficiency", "")))

        gi = self.raw.get("GeneralInfo") or {}
        if gi.get("EnglishLevel"):
            texts.append(str(gi["EnglishLevel"]))

        if self.raw.get("CleanedResumeText"):
            texts.append(str(self.raw["CleanedResumeText"]))

        best = 0
        for t in texts:
            s = t.lower()
            m = re.search(r"\b([abc][12])\b", s)
            if m:
                best = max(best, ENGLISH_LEVEL_MAP[m.group(1)])
            if "advanced" in s:
                best = max(best, ENGLISH_LEVEL_MAP["c1"])
            if "upper" in s:
                best = max(best, ENGLISH_LEVEL_MAP["b2"])
            if "intermediate" in s and best < ENGLISH_LEVEL_MAP["b2"]:
                best = max(best, ENGLISH_LEVEL_MAP["b1"])

        return ENGLISH_LEVEL_INV.get(best, "UNKNOWN")

    def to_text_blocks(self) -> List[str]:
        blocks: List[str] = []
        gi = self.raw.get("GeneralInfo", {})
        title_hint = self._derive_title_hint(gi)
        derived_keywords = self._derive_keywords()

        blocks.append(f"[Candidate] {self.candidate_id} {title_hint}\nSummary:\n{self.summary}")
        if self.skills:
            skills = ", ".join([s.get("SkillName", "") for s in self.skills])
            blocks.append(f"Skills: {skills}")
        if derived_keywords:
            blocks.append("DerivedKeywords: " + ", ".join(sorted(derived_keywords)))

        blocks.append(json.dumps(self.raw, ensure_ascii=False))
        return blocks

    def _derive_title_hint(self, general_info: Dict[str, Any]) -> str:
        titles = " ".join([str(general_info.get("TitleDetected", "")), str(general_info.get("TitlePredicted", ""))]).lower()
        if any(k in titles for k in BACKEND_HINT_KEYWORDS):
            return "[HINT] backend-dotnet"
        if any(k in titles for k in FRONTEND_HINT_KEYWORDS):
            return "[HINT] frontend"
        return ""

    def _derive_keywords(self) -> set:
        text = (self.raw.get("CleanedResumeText") or "") + " " + self.summary
        kws = set()
        for pattern, keyword in KEYWORD_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                kws.add(keyword)
        return kws