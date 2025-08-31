import os, pytest
from src.core.application.agent import build_chain

@pytest.mark.skipif(not os.path.exists("data/vectors/chroma"),
                    reason="index not built")
def test_chat_smoke():
    chain = build_chain()
    out = chain({"query": "¿Who has better English?"})
    assert "result" in out and isinstance(out["result"], str)
