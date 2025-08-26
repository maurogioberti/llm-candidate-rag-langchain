import os
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_embeddings():
    model_name = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    normalize = os.getenv("EMB_NORMALIZE", "true").lower() == "true"
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": normalize}
    )