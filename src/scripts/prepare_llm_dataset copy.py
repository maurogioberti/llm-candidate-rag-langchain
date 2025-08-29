#TODO: Get rid of folder scripts and relocate in better place ... also check if we really need this smoke test of the dataset
import os, json, random
from infra.llm import load_llm_instruction_records

def main():
    path = os.getenv("LLM_INSTRUCTION_FILE", "data/instructions/llm.jsonl")
    recs = load_llm_instruction_records(path)
    print(f"[INFO] loaded {len(recs)} records from {path}")
    if not recs:
        return
    sample = random.choice(recs)
    print("[SAMPLE] instruction:", sample["instruction"])
    print("[SAMPLE] input.question:", sample["input"].get("question"))
    print("[SAMPLE] output keys:", list(sample["output"].keys()))

if __name__ == "__main__":
    main()