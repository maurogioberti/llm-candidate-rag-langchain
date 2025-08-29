#TODO: Get rid of folder scripts and relocate in better place
import os, json
from pathlib import Path
from infra.llm import load_llm_instruction_records

def main():
    src = os.getenv("LLM_INSTRUCTION_FILE", "data/instructions/llm.jsonl")
    out_dir = Path(os.getenv("LLM_FT_EXPORT_DIR", "data/finetune_exports"))
    out_dir.mkdir(parents=True, exist_ok=True)

    recs = load_llm_instruction_records(src)
    if not recs:
        print(f"[WARN] No records loaded from {src}")
        return

    openai_path = out_dir / "openai_chat.jsonl"
    with open(openai_path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps({
                "messages": [
                    {"role": "system", "content": r["instruction"]},
                    {"role": "user", "content": json.dumps(r["input"], ensure_ascii=False)},
                    {"role": "assistant", "content": json.dumps(r["output"], ensure_ascii=False)}
                ]
            }, ensure_ascii=False) + "\n")

    instruct_path = out_dir / "instruct_generic.jsonl"
    with open(instruct_path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps({
                "instruction": r["instruction"],
                "input": r["input"],
                "output": r["output"]
            }, ensure_ascii=False) + "\n")

    print(f"[OK] Loaded {len(recs)} records")
    print(f"[OK] Wrote: {openai_path}")
    print(f"[OK] Wrote: {instruct_path}")
    print("\nNext steps (OpenAI):")
    print("  1) export OPENAI_API_KEY=... ")
    print(f"  2) openai files create -f {openai_path} -p fine-tune")
    print("  3) openai fine_tuning.jobs.create -t <file_id> -m gpt-4o-mini")

if __name__ == "__main__":
    main()
