import os
from langchain_huggingface import HuggingFaceEmbeddings

ENV_EMB_MODEL = "EMB_MODEL"
ENV_EMB_NORMALIZE = "EMB_NORMALIZE"
DEFAULT_EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DEVICE = "cpu"
DEFAULT_NORMALIZE = "true"

def load_embeddings():
    model_name = os.getenv(ENV_EMB_MODEL, DEFAULT_EMB_MODEL)
    normalize = os.getenv(ENV_EMB_NORMALIZE, DEFAULT_NORMALIZE).lower() == "true"
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": DEFAULT_DEVICE},
        encode_kwargs={"normalize_embeddings": normalize}
    )