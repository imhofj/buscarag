"""
app.py
Interfaz visual con Streamlit:
- Subir documentos (PDF/TXT) para ingestarlos
- Hacer preguntas en lenguaje natural
- Ver la respuesta generada en streaming junto con las fuentes que la sustentan
- Ver cuánto tarda cada etapa (búsqueda, primer token y respuesta completa)
"""

# Streamlit Cloud trae una versión vieja de sqlite3 que ChromaDB no acepta.
# En Linux la reemplazamos por pysqlite3; en Windows no se instala y se ignora.
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
import tempfile
import streamlit as st
from ingest import ingest_file, get_chroma_collection, reset_database
from rag import answer_question_stream

st.set_page_config(page_title="Buscador Semántico", page_icon="🔍")
st.title("🔍 Buscador Semántico de Documentos")
st.caption("Subí tus documentos y hacé preguntas sobre su contenido en lenguaje natural.")

# --- Barra lateral: estado de la base + reset ---
with st.sidebar:
    st.subheader("Base de documentos")
    try:
        collection = get_chroma_collection()
        st.write(f"Fragmentos indexados: **{collection.count()}**")
    except Exception:
        st.write("Fragmentos indexados: 0")

    if st.button("🗑️ Borrar todo y empezar de cero"):
        reset_database()
        st.success("Base borrada. Subí documentos de nuevo.")
        st.rerun()

# --- Sección 1: subir documentos ---
st.header("1. Subir documentos")
uploaded_files = st.file_uploader(
    "Subí uno o más PDFs o TXT", type=["pdf", "txt"], accept_multiple_files=True
)

if uploaded_files and st.button("Procesar documentos"):
    with st.spinner("Extrayendo texto y generando embeddings..."):
        total_chunks = 0
        for file in uploaded_files:
            # Usamos una carpeta temporal pero conservando el NOMBRE REAL del
            # archivo, para que quede bien identificado como "fuente" en Chroma
            # (antes se usaba un nombre random tipo tmpXk29fa.pdf).
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = os.path.join(tmp_dir, file.name)
                with open(tmp_path, "wb") as f:
                    f.write(file.read())

                n = ingest_file(tmp_path)
                total_chunks += n

                if n == 0:
                    st.warning(
                        f"'{file.name}' se procesó pero no se le pudo extraer texto "
                        f"(¿es un PDF escaneado / imagen sin texto seleccionable?)."
                    )

        st.success(f"Se procesaron {len(uploaded_files)} archivo(s) — {total_chunks} fragmentos indexados.")

# --- Sección 2: preguntar ---
st.header("2. Hacé una pregunta")
query = st.text_input("¿Qué querés saber sobre tus documentos?")

if query:
    with st.spinner("Buscando fragmentos relevantes..."):
        fragments, stream, metrics = answer_question_stream(query)

    if stream is None:
        st.info("No hay documentos cargados todavía. Subí alguno en el paso 1.")
    else:
        st.subheader("Respuesta")
        # write_stream va mostrando el texto a medida que llega del modelo
        st.write_stream(stream)

        # Las métricas de generación se completan recién cuando terminó el stream
        st.caption(
            f"⏱️ Búsqueda: {metrics['search_s'] * 1000:.0f} ms · "
            f"Primer token: {metrics.get('first_token_s', 0) * 1000:.0f} ms · "
            f"Respuesta completa: {metrics.get('generation_s', 0):.2f} s"
        )

        st.subheader("Fuentes")
        for i, source in enumerate(fragments, start=1):
            with st.expander(f"Fragmento {i} — {source['source']}"):
                st.write(source["text"])