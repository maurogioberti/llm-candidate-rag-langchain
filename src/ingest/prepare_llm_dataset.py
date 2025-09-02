import os, json
from pathlib import Path
from src.core.infrastructure.llm import load_llm_instruction_records

ENV_LLM_INSTRUCTION_FILE = "LLM_INSTRUCTION_FILE"
ENV_LLM_FT_EXPORT_DIR = "LLM_FT_EXPORT_DIR"
DEFAULT_LLM_INSTRUCTION_FILE = "data/instructions/llm.jsonl"
DEFAULT_FT_EXPORT_DIR = "data/finetune_exports"
OPENAI_EXPORT_FILENAME = "openai_chat.jsonl"
INSTRUCT_EXPORT_FILENAME = "instruct_generic.jsonl"
FILE_ENCODING = "utf-8"
ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

def main():
    source_path = os.getenv(ENV_LLM_INSTRUCTION_FILE, DEFAULT_LLM_INSTRUCTION_FILE)
    export_dir = Path(os.getenv(ENV_LLM_FT_EXPORT_DIR, DEFAULT_FT_EXPORT_DIR))
    export_dir.mkdir(parents=True, exist_ok=True)

    records = load_llm_instruction_records(source_path)
    if not records:
        print(f"[WARN] No records loaded from {source_path}")
        return

    openai_path = export_dir / OPENAI_EXPORT_FILENAME
    with open(openai_path, "w", encoding=FILE_ENCODING) as f:
        for r in records:
            f.write(f"{json.dumps({
                "messages": [
                    {"role": ROLE_SYSTEM, "content": r["instruction"]},
                    {"role": ROLE_USER, "content": json.dumps(r["input"], ensure_ascii=False)},
                    {"role": ROLE_ASSISTANT, "content": json.dumps(r["output"], ensure_ascii=False)}
                ]
            }, ensure_ascii=False)}\n")

    instruct_path = export_dir / INSTRUCT_EXPORT_FILENAME
    with open(instruct_path, "w", encoding=FILE_ENCODING) as f:
        for r in records:
            f.write(f"{json.dumps({
                "instruction": r["instruction"],
                "input": r["input"],
                "output": r["output"]
            }, ensure_ascii=False)}\n")

    print(f"[OK] Loaded {len(records)} records")
    print(f"[OK] Wrote: {openai_path}")
    print(f"[OK] Wrote: {instruct_path}")

if __name__ == "__main__":
    main()
