"""
rag.py
Se encarga de:
1. Buscar en Chroma los fragmentos más relevantes para una pregunta
2. Armar un prompt con esos fragmentos como contexto
3. Pedirle a un modelo (via Groq) que genere una respuesta basada en ese contexto
4. Medir la latencia de cada etapa (búsqueda, primer token y respuesta completa)
"""

import os
import time
from dotenv import load_dotenv
from groq import Groq
from ingest import get_chroma_collection

# load_dotenv() tiene que ir ANTES de leer las variables de entorno,
# si no, los valores del .env se ignoran.
load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Parámetros comunes para todas las llamadas al modelo.
# gpt-oss es un modelo de razonamiento: con esfuerzo "low" piensa menos antes
# de responder, lo que reduce bastante el tiempo hasta el primer token.
# (Si se cambia a un modelo que no sea de razonamiento, sacar "extra_body").
GENERATION_PARAMS = {
    "model": MODEL,
    "max_tokens": 1024,
    "extra_body": {"reasoning_effort": "low"},
}


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
    return f"""Respondé la pregunta usando ÚNICAMENTE la información del contexto.

Tené en cuenta que la pregunta puede usar palabras distintas a las del documento
(por ejemplo, "sueldo" en vez de "salario", o "arreglar" en vez de "reparar").
Si el contexto contiene la información aunque esté dicha con otras palabras, respondé.

Respondé de forma breve y directa, mencionando el dato concreto.
Solo si el contexto realmente no contiene la respuesta, respondé exactamente:
"No encontré esa información en los documentos."

Contexto:
{context}

Pregunta: {query}

Respuesta:"""


def answer_question(query: str, n_results: int = 4) -> dict:
    """Pipeline completo de RAG sin streaming (útil para tests y evaluación)."""
    fragments = search(query, n_results=n_results)

    if not fragments:
        return {"answer": "No hay documentos cargados todavía.", "sources": []}

    prompt = build_prompt(query, fragments)

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        **GENERATION_PARAMS,
    )

    answer = response.choices[0].message.content
    return {"answer": answer, "sources": fragments}


def answer_question_stream(query: str, n_results: int = 4):
    """
    Pipeline de RAG con streaming y medición de latencia.

    Devuelve una tupla (fragments, stream, metrics):
    - fragments: los fragmentos recuperados de Chroma
    - stream: generador que va entregando la respuesta de a pedacitos
      (None si no hay documentos cargados)
    - metrics: diccionario con los tiempos en segundos. "search_s" está
      disponible enseguida; "first_token_s" y "generation_s" se completan
      a medida que se consume el stream.
    """
    start = time.perf_counter()
    fragments = search(query, n_results=n_results)
    metrics = {"search_s": time.perf_counter() - start}

    if not fragments:
        return fragments, None, metrics

    prompt = build_prompt(query, fragments)

    def generate():
        gen_start = time.perf_counter()
        stream = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **GENERATION_PARAMS,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            text = chunk.choices[0].delta.content
            if text:
                # Tiempo hasta el primer token: lo que el usuario percibe como "espera"
                if "first_token_s" not in metrics:
                    metrics["first_token_s"] = time.perf_counter() - gen_start
                yield text
        metrics["generation_s"] = time.perf_counter() - gen_start

    return fragments, generate(), metrics