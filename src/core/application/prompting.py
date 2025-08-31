from pathlib import Path

PROMPTS_DIR = Path("prompts")

def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")
