"""
rag.py
Se encarga de:
1. Buscar en Chroma los fragmentos más relevantes para una pregunta
2. Armar un prompt con esos fragmentos como contexto
3. Pedirle a un modelo (via Groq) que genere una respuesta basada en ese contexto
"""

import os
from dotenv import load_dotenv
from groq import Groq
from ingest import get_chroma_collection
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def search(query: str, n_results: int = 4):
    """Busca los fragmentos más relevantes semánticamente para la consulta."""
    collection = get_chroma_collection()
    results = collection.query(query_texts=[query], n_results=n_results)

    fragments = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        fragments.append({"text": doc, "source": meta["source"]})
    return fragments


def build_prompt(query: str, fragments: list[dict]) -> str:
    context = "\n\n".join(
        f"[Fuente: {f['source']}]\n{f['text']}" for f in fragments
    )
    return f"""Respondé la siguiente pregunta usando ÚNICAMENTE la información del contexto.
Si el contexto no tiene la respuesta, decí explícitamente que no la encontraste en los documentos.

Contexto:
{context}

Pregunta: {query}

Respuesta:"""


def answer_question(query: str, n_results: int = 4) -> dict:
    """Pipeline completo de RAG: busca fragmentos y genera una respuesta."""
    fragments = search(query, n_results=n_results)

    if not fragments:
        return {"answer": "No hay documentos cargados todavía.", "sources": []}

    prompt = build_prompt(query, fragments)

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response.choices[0].message.content
    return {"answer": answer, "sources": fragments}