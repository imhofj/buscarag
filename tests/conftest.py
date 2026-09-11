"""
Configuración compartida por todos los tests.

Los tests NO usan internet ni descargan modelos: el modelo de embeddings y el
cliente de Groq se reemplazan por versiones falsas, así corren en segundos.
"""

import os
import uuid

import chromadb
import pytest
from chromadb import EmbeddingFunction
from chromadb.config import Settings

# rag.py crea el cliente de Groq al importarse y exige una API key.
# Si no hay una real configurada, usamos una falsa (nunca se llama a la API).
os.environ.setdefault("GROQ_API_KEY", "clave-falsa-para-tests")

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_EJEMPLO = os.path.join(RAIZ, "ejemplos", "politica_cobranzas_ceibo.pdf")


class EmbedderFalso(EmbeddingFunction):
    """Devuelve vectores fijos: alcanza para probar la lógica sin descargar modelos."""

    def __init__(self):
        pass

    def __call__(self, input):
        return [[float(len(texto)), 1.0, 0.0] for texto in input]


@pytest.fixture
def coleccion_en_memoria(monkeypatch):
    """
    Reemplaza la colección de Chroma de la app por una en memoria,
    para que los tests no toquen la carpeta chroma_db.
    """
    import ingest

    cliente = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    coleccion = cliente.create_collection(
        name=f"test-{uuid.uuid4().hex[:8]}", embedding_function=EmbedderFalso()
    )
    monkeypatch.setattr(ingest, "get_chroma_collection", lambda: coleccion)
    return coleccion
