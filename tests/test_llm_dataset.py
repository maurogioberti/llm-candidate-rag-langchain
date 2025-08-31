import os
from src.infra.llm import load_llm_instruction_records

def test_llm_dataset_loads():
    path = os.getenv("LLM_INSTRUCTION_FILE", "data/instructions/llm.jsonl")
    recs = load_llm_instruction_records(path)
    
    if os.path.exists(path):
        assert len(recs) > 0
        assert all("instruction" in r and "input" in r and "output" in r for r in recs)
