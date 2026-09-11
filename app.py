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

# Documento de ejemplo incluido en el repo, para probar la app con un clic
CARPETA_APP = os.path.dirname(os.path.abspath(__file__))
PDF_EJEMPLO = os.path.join(CARPETA_APP, "ejemplos", "politica_cobranzas_ceibo.pdf")
NOMBRE_EJEMPLO = os.path.basename(PDF_EJEMPLO)
# Preguntas parafraseadas que el sistema respondió bien en la evaluación (eval/)
PREGUNTAS_EJEMPLO = [
    "¿Qué tiene que hacer el operador si la persona empieza a insultar?",
    "¿En qué casos el bot tiene que pasarle la llamada a una persona?",
    "Si alguien se quedó sin trabajo, ¿se le puede dar un respiro en las llamadas?",
]


def ejemplo_ya_cargado() -> bool:
    """Indica si el documento de ejemplo ya está indexado en la base."""
    try:
        resultado = get_chroma_collection().get(where={"source": NOMBRE_EJEMPLO}, limit=1)
        return len(resultado["ids"]) > 0
    except Exception:
        return False


def usar_pregunta(pregunta: str):
    """Callback de los botones de preguntas de ejemplo: completa el campo de texto."""
    st.session_state["pregunta"] = pregunta

st.set_page_config(page_title="Buscador Semántico", page_icon="🔍")
st.title("🔍 Buscador Semántico de Documentos")
st.caption("Subí tus documentos y hacé preguntas sobre su contenido en lenguaje natural.")

# --- Sección 1: subir documentos ---
st.header("1. Subir documentos")

if st.button("📄 Probar con un documento de ejemplo"):
    if ejemplo_ya_cargado():
        st.info("El documento de ejemplo ya está cargado. Probá con alguna de las preguntas de abajo.")
    else:
        with st.spinner("Cargando la política de cobranzas de ejemplo..."):
            n = ingest_file(PDF_EJEMPLO)
        st.success(f"Documento de ejemplo cargado: {n} fragmentos indexados.")
st.caption(
    "El ejemplo es la política de cobranzas de Financiera Ceibo, una empresa ficticia. "
    "También podés subir tus propios archivos:"
)

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

if ejemplo_ya_cargado():
    st.write("Preguntas de ejemplo:")
    columnas = st.columns(len(PREGUNTAS_EJEMPLO))
    for columna, pregunta in zip(columnas, PREGUNTAS_EJEMPLO):
        columna.button(pregunta, on_click=usar_pregunta, args=(pregunta,), use_container_width=True)

query = st.text_input("¿Qué querés saber sobre tus documentos?", key="pregunta")

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

# --- Barra lateral: estado de la base + reset ---
# Va al final del script para que el contador ya incluya los documentos
# cargados en esta misma ejecución (Streamlit la muestra igual a la izquierda).
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