"""
ingest.py
Se encarga de:
1. Extraer texto de archivos PDF/TXT
2. Trocear el texto en fragmentos (chunks) manejables
3. Generar embeddings de cada fragmento
4. Guardarlos en una base de datos vectorial (ChromaDB)
"""

import os
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "documentos"
CHUNK_SIZE = 500       # caracteres por fragmento
CHUNK_OVERLAP = 50     # superposición entre fragmentos, para no cortar ideas
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")


def extract_text(filepath: str) -> str:
    """Extrae texto de un PDF o TXT."""
    if filepath.lower().endswith(".pdf"):
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

    text = " ".join(text.split())
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Divide el texto en fragmentos con superposición."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def get_chroma_collection():
    """Crea o recupera la colección de Chroma, usando sentence-transformers como embedder."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedder,
    )
    return collection


def ingest_file(filepath: str) -> int:
    """
    Procesa un archivo completo: extrae texto, lo trocea,
    y lo guarda en la base vectorial. Devuelve la cantidad de chunks agregados.
    """
    filename = os.path.basename(filepath)
    text = extract_text(filepath)
    chunks = chunk_text(text)

    # Si no se pudo extraer texto (por ejemplo, un PDF escaneado), no hay nada
    # que guardar. Chroma no acepta listas vacías, así que cortamos acá.
    if not chunks:
        return 0

    collection = get_chroma_collection()

    ids = [f"{filename}-{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=metadatas,
    )
    return len(chunks)


def reset_database():
    """Borra todos los documentos indexados (sin tocar archivos en disco)."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # si la colección no existía, no hay nada que borrar


if __name__ == "__main__":
    # Prueba rápida: ingesta todos los archivos de la carpeta data/
    data_dir = "data"
    for fname in os.listdir(data_dir):
        if fname.lower().endswith((".pdf", ".txt")):
            n = ingest_file(os.path.join(data_dir, fname))
            print(f"Ingerido '{fname}': {n} fragmentos")