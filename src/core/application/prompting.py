from pathlib import Path

PRIMARY_PROMPTS_DIR = Path("data") / "prompts"
FALLBACK_PROMPTS_DIR = Path("prompts")
PROMPT_ENCODING = "utf-8"

def load_prompt(name: str) -> str:
    primary_path = PRIMARY_PROMPTS_DIR / name
    if primary_path.exists():
        return primary_path.read_text(encoding=PROMPT_ENCODING)
    fallback_path = FALLBACK_PROMPTS_DIR / name
    return fallback_path.read_text(encoding=PROMPT_ENCODING)
