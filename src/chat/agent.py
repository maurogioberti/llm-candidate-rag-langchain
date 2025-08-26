import os
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from core.embedding_client import load_embeddings
from core.retriever import chroma_persistent
from core.prompting import load_prompt

def _load_llm():
    prov = os.getenv("LLM_PROVIDER", "ollama").lower()
    model = os.getenv("LLM_MODEL", "llama3.1:8b-instruct")
    if prov == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model, base_url=base_url, temperature=0)
    return ChatOpenAI(
        model=model,
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0
    )

def build_chain():
    emb = load_embeddings()
    store = chroma_persistent(emb)
    retriever = store.as_retriever(search_kwargs={"k": 6})
    system = load_prompt("chat_system.txt")
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            system +
            "\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer (cite candidate_id and section):"
        ),
    )
    llm = _load_llm()
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return chain
