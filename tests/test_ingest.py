from src.ingest.build_index import build_index

def test_build_index_runs():
    info = build_index()
    assert "candidates" in info and "chunks" in info