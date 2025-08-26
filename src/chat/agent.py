import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv(dotenv_path=".env", override=True)

__all__ = ["build_chain", "build_index"]


def _load_llm():
    prov = os.getenv("LLM_PROVIDER")
    model = os.getenv("LLM_MODEL")
    if not prov or not model:
        raise RuntimeError("Missing LLM_PROVIDER or LLM_MODEL in .env")
    prov = prov.lower()
    if prov == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        print(f"[LLM] provider=ollama model={model} base={base_url}")
        return ChatOllama(model=model, base_url=base_url, temperature=0)
    if prov == "openai":
        base_url = os.getenv("OPENAI_BASE_URL")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY in .env for provider=openai")
        print(f"[LLM] provider=openai model={model} base={base_url or 'default'}")
        return ChatOpenAI(model=model, base_url=base_url, api_key=api_key, temperature=0)
    raise ValueError(f"Unsupported LLM_PROVIDER: {prov}")


def build_chain():
    from core.embedding_client import load_embeddings
    from core.retriever import chroma_persistent
    from core.prompting import load_prompt

    emb = load_embeddings()
    store = chroma_persistent(emb)
    retriever = store.as_retriever(search_kwargs={"k": 6})
    system = load_prompt("chat_system.txt")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        (
            "human",
            'If the question mentions backend or .NET/C#, treat the retrieval focus terms as: '
            '"\\.NET, C#, ASP.NET, Entity Framework, SQL Server, Azure".\n'
            "Context:\n{context}\n\nQuestion:\n{input}\n\n"
            "Answer (cite CandidateId and section):"
        ),
    ])
    llm = _load_llm()
    doc_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, doc_chain)


def build_index():
    return build_chain()
