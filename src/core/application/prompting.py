from pathlib import Path

PROMPTS_DIR = Path("data") / "prompts"
PROMPT_ENCODING = "utf-8"

def load_prompt(name: str) -> str:
    filepath = PROMPTS_DIR / name
    if filepath.exists():
        return filepath.read_text(encoding=PROMPT_ENCODING)
