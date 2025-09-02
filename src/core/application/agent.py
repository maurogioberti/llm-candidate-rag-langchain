import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv(dotenv_path=".env", override=True)

__all__ = ["build_chain", "build_index"]

ENV_LLM_PROVIDER = "LLM_PROVIDER"
ENV_LLM_MODEL = "LLM_MODEL"
ENV_OLLAMA_BASE_URL = "OLLAMA_BASE_URL"
ENV_OPENAI_BASE_URL = "OPENAI_BASE_URL"
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_ENABLE_GPT5_MINI = "ENABLE_GPT5_MINI_PREVIEW"
LLM_PROVIDER_OLLAMA = "ollama"
LLM_PROVIDER_OPENAI = "openai"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OPENAI_MODEL = "gpt-5-mini"
PROMPT_SYSTEM_FILE = "chat_system.txt"
PROMPT_HUMAN_FILE = "chat_human.txt"
RETRIEVER_TOP_K = 6
TEMPERATURE_ZERO = 0
ENV_RETRIEVAL_TYPES = "RETRIEVAL_TYPES"
DEFAULT_RETRIEVAL_TYPES = "candidate"

def _load_llm():
    provider = os.getenv(ENV_LLM_PROVIDER)
    model_name = os.getenv(ENV_LLM_MODEL)
    if not provider:
        raise RuntimeError("Missing LLM_PROVIDER in .env")
    provider = provider.lower()
    if provider == LLM_PROVIDER_OLLAMA:
        base_url = os.getenv(ENV_OLLAMA_BASE_URL, DEFAULT_OLLAMA_BASE_URL)
        return ChatOllama(model=model_name, base_url=base_url, temperature=TEMPERATURE_ZERO)
    if provider == LLM_PROVIDER_OPENAI:
        base_url = os.getenv(ENV_OPENAI_BASE_URL)
        api_key = os.getenv(ENV_OPENAI_API_KEY)
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY in .env for provider=openai")
        enable_preview = os.getenv(ENV_ENABLE_GPT5_MINI, "true").lower() == "true"
        effective_model = (DEFAULT_OPENAI_MODEL if enable_preview else model_name) or DEFAULT_OPENAI_MODEL
        return ChatOpenAI(model=effective_model, base_url=base_url, api_key=api_key, temperature=TEMPERATURE_ZERO)
    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

def build_chain():
    from src.core.application.embedding_client import load_embeddings
    from src.core.application.retriever import chroma_persistent
    from src.core.application.prompting import load_prompt

    embeddings = load_embeddings()
    vector_store = chroma_persistent(embeddings)
    types = os.getenv(ENV_RETRIEVAL_TYPES, DEFAULT_RETRIEVAL_TYPES)
    metadata_filter = {"type": {"$in": [t.strip() for t in types.split(",") if t.strip()]}}
    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K, "filter": metadata_filter})
    system_prompt = load_prompt(PROMPT_SYSTEM_FILE)
    human_prompt = load_prompt(PROMPT_HUMAN_FILE)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
    ])
    llm = _load_llm()
    doc_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, doc_chain)

def build_index():
    return build_chain()
