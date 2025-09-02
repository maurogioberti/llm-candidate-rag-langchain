from __future__ import annotations
from pydantic import BaseModel
from typing import Any, Dict, List
import json, re

ENGLISH_LEVEL_MAP = {
    "A1": 1,
    "A2": 2,
    "B1": 3,
    "B2": 4,
    "C1": 5,
    "C2": 6
}
ENGLISH_LEVEL_INV = {v: k for k, v in ENGLISH_LEVEL_MAP.items()}
ENGLISH_LANGUAGE_NAMES = {"english", "inglés", "en"}
SENIORITY_LEVEL_KEYWORDS = ("mid", "senior", "lead", "staff", "principal")
MIN_PREPARED_SCORE = 60
BACKEND_TITLE_KEYWORDS = ("backend", ".net", "c#", "asp.net")
FRONTEND_TITLE_KEYWORDS = ("frontend", "react")
REGEX_ENGLISH_LEVEL = r"\b([ABC][12])\b"
REGEX_DOTNET = r"\b\.net\b"
REGEX_CSHARP = r"\bc#\b"
REGEX_ASPNET = r"\basp\.net\b"
REGEX_ENTITY_FRAMEWORK = r"\bentity framework\b"
REGEX_SQL_SERVER = r"\bsql server\b"
REGEX_AZURE = r"\bazure\b"
KEYWORD_PATTERNS = [
    (REGEX_DOTNET, ".NET"),
    (REGEX_CSHARP, "C#"),
    (REGEX_ASPNET, "ASP.NET"),
    (REGEX_ENTITY_FRAMEWORK, "Entity Framework"),
    (REGEX_SQL_SERVER, "SQL Server"),
    (REGEX_AZURE, "Azure"),
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
        general_score = (self.scores or {}).get("GeneralScore") or 0
        seniority_level = ((self.raw.get("GeneralInfo") or {}).get("SeniorityLevel") or "").lower()
        return general_score >= MIN_PREPARED_SCORE or any(keyword in seniority_level for keyword in SENIORITY_LEVEL_KEYWORDS)

    @property
    def english_level(self) -> str:
        return self._get_english_level()

    def _get_english_level(self) -> str:
        english_level_sources: List[str] = []
        for language_record in self.languages or []:
            language_name = str(language_record.get("Language", "")).lower()
            if language_name in ENGLISH_LANGUAGE_NAMES:
                english_level_sources.append(str(language_record.get("Proficiency", "")))

        general_info = self.raw.get("GeneralInfo") or {}
        if general_info.get("EnglishLevel"):
            english_level_sources.append(str(general_info["EnglishLevel"]))

        if self.raw.get("CleanedResumeText"):
            english_level_sources.append(str(self.raw["CleanedResumeText"]))

        best_level = 0
        for source_text in english_level_sources:
            normalized_text = source_text.lower()
            match = re.search(REGEX_ENGLISH_LEVEL, normalized_text)
            if match:
                best_level = max(best_level, ENGLISH_LEVEL_MAP[match.group(1).upper()])
            if "advanced" in normalized_text:
                best_level = max(best_level, ENGLISH_LEVEL_MAP["C1"])
            if "upper" in normalized_text:
                best_level = max(best_level, ENGLISH_LEVEL_MAP["B2"])
            if "intermediate" in normalized_text and best_level < ENGLISH_LEVEL_MAP["B2"]:
                best_level = max(best_level, ENGLISH_LEVEL_MAP["B1"])

        return ENGLISH_LEVEL_INV.get(best_level, "UNKNOWN")

    def to_text_blocks(self) -> List[str]:
        text_blocks: List[str] = []
        general_info = self.raw.get("GeneralInfo", {})
        title_hint = self._get_title_hint(general_info)
        derived_keywords = self._get_derived_keywords()

        candidate_summary = f"[Candidate] {self.candidate_id} {title_hint}\nSummary:\n{self.summary}"
        text_blocks.append(candidate_summary)
        if self.skills:
            skills_summary = ", ".join([skill.get("SkillName", "") for skill in self.skills])
            text_blocks.append(f"Skills: {skills_summary}")
        if derived_keywords:
            text_blocks.append(f"DerivedKeywords: {', '.join(sorted(derived_keywords))}")

        text_blocks.append(json.dumps(self.raw, ensure_ascii=False))
        return text_blocks

    def _get_title_hint(self, general_info: Dict[str, Any]) -> str:
        detected_title = str(general_info.get("TitleDetected", ""))
        predicted_title = str(general_info.get("TitlePredicted", ""))
        combined_titles = f"{detected_title} {predicted_title}".lower()
        if any(keyword in combined_titles for keyword in BACKEND_TITLE_KEYWORDS):
            return "[HINT] backend-dotnet"
        if any(keyword in combined_titles for keyword in FRONTEND_TITLE_KEYWORDS):
            return "[HINT] frontend"
        return ""

    def _get_derived_keywords(self) -> set:
        resume_text = (self.raw.get("CleanedResumeText") or "")
        candidate_summary = self.summary
        combined_text = f"{resume_text} {candidate_summary}"
        derived_keywords = set()
        for regex_pattern, keyword_name in KEYWORD_PATTERNS:
            if re.search(regex_pattern, combined_text, re.IGNORECASE):
                derived_keywords.add(keyword_name)
        return derived_keywords