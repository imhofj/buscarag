"""
eval/evaluar.py
Evalúa la calidad de la búsqueda semántica de Buscarag usando un documento
de prueba y un set de preguntas cuya respuesta se conoce de antemano.

Uso (desde la raíz del proyecto):
    python eval/evaluar.py
    python eval/evaluar.py --modelo all-MiniLM-L6-v2   # probar con otro modelo
    python eval/evaluar.py --generacion      # además evalúa las respuestas del LLM

Métricas de búsqueda:
- Acierto@1: el fragmento correcto quedó en primer lugar.
- Acierto@k: el fragmento correcto está entre los k recuperados (los que ve el LLM).
- MRR: promedio de 1/posición del fragmento correcto (1.0 = siempre primero).
- Latencia: tiempo de generar el embedding de la pregunta y consultar Chroma.

Métricas de generación (con --generacion):
- Respondió: en preguntas con respuesta, el modelo respondió en vez de decir que no sabe.
- Se abstuvo: en preguntas sin respuesta, el modelo dijo que no la encontró en vez de inventar.
- Tiempo hasta el primer token y tiempo de respuesta completa.

La evaluación usa una base de Chroma EN MEMORIA, así que no toca la base de la app.
"""

import argparse
import json
import os
import statistics
import sys
import time
import unicodedata

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Permite importar los módulos de la raíz del proyecto (ingest.py, rag.py)
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from ingest import extract_text, chunk_text, EMBEDDING_MODEL  # noqa: E402

CARPETA_EVAL = os.path.dirname(os.path.abspath(__file__))
DOCUMENTO_DEFAULT = os.path.join(RAIZ, "ejemplos", "politica_cobranzas_ceibo.pdf")
PREGUNTAS_DEFAULT = os.path.join(CARPETA_EVAL, "preguntas.json")
MODELO_DEFAULT = EMBEDDING_MODEL  # el mismo modelo que usa la app

# El prompt le pide al modelo que, si no encuentra la respuesta, conteste
# "No encontré esa información en los documentos.". Por eso solo miramos el
# COMIENZO de la respuesta: así no confundimos respuestas válidas del estilo
# "No se especifica X, pero el documento indica Y" con una abstención.
INICIOS_ABSTENCION = (
    "no encontre", "no lo encontre", "no la encontre",
    "no se encontr", "no hay informacion",
)


def normalizar(texto: str) -> str:
    """Minúsculas, sin tildes y con espacios colapsados, para comparar textos."""
    texto = unicodedata.normalize("NFD", texto.lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return " ".join(texto.split())


def crear_embedder(modelo: str):
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=modelo)


def indexar_documento(ruta: str, modelo: str):
    """Extrae, trocea e indexa el documento en una colección de Chroma en memoria."""
    texto = extract_text(ruta)
    chunks = chunk_text(texto)

    cliente = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    coleccion = cliente.get_or_create_collection(
        name="evaluacion", embedding_function=crear_embedder(modelo)
    )
    coleccion.add(
        documents=chunks,
        ids=[f"chunk-{i}" for i in range(len(chunks))],
        metadatas=[{"source": os.path.basename(ruta), "chunk_index": i} for i in range(len(chunks))],
    )
    return coleccion, chunks


def validar_preguntas(preguntas: list[dict], chunks: list[str]) -> list[dict]:
    """Descarta preguntas cuya frase esperada no aparece en ningún fragmento (test mal armado)."""
    chunks_norm = [normalizar(c) for c in chunks]
    validas = []
    for p in preguntas:
        frase = p.get("debe_contener")
        if frase and not any(normalizar(frase) in c for c in chunks_norm):
            print(f"  [AVISO] La frase esperada no está en ningún fragmento, se omite: {p['pregunta']}")
            continue
        validas.append(p)
    return validas


def evaluar_busqueda(coleccion, preguntas: list[dict], k: int) -> list[dict]:
    """Corre cada pregunta contra Chroma y registra en qué posición aparece el fragmento correcto."""
    coleccion.query(query_texts=["pregunta de calentamiento"], n_results=1)  # carga el modelo

    resultados = []
    for p in preguntas:
        inicio = time.perf_counter()
        res = coleccion.query(query_texts=[p["pregunta"]], n_results=k)
        latencia = time.perf_counter() - inicio

        documentos = res["documents"][0]
        posicion = None
        if p.get("debe_contener"):
            frase = normalizar(p["debe_contener"])
            posicion = next(
                (i + 1 for i, doc in enumerate(documentos) if frase in normalizar(doc)), None
            )

        resultados.append({**p, "posicion": posicion, "latencia_s": latencia, "fragmentos": documentos})
    return resultados


def evaluar_generacion(resultados: list[dict]) -> None:
    """Le pide al LLM que responda cada pregunta con los fragmentos recuperados."""
    from rag import build_prompt, client, GENERATION_PARAMS  # se importa acá porque necesita la API key

    for r in resultados:
        fragmentos = [{"text": t, "source": "documento"} for t in r["fragmentos"]]
        prompt = build_prompt(r["pregunta"], fragmentos)
        try:
            inicio = time.perf_counter()
            primer_token = None
            partes = []
            stream = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}], stream=True, **GENERATION_PARAMS
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                texto = chunk.choices[0].delta.content
                if texto:
                    if primer_token is None:
                        primer_token = time.perf_counter() - inicio
                    partes.append(texto)
            respuesta = "".join(partes)
            r["respuesta"] = respuesta
            r["primer_token_s"] = primer_token
            r["generacion_s"] = time.perf_counter() - inicio
            inicio_respuesta = normalizar(respuesta).lstrip(' "\'*>¿¡“”«')
            r["se_abstuvo"] = inicio_respuesta.startswith(INICIOS_ABSTENCION)
        except Exception as e:  # por ejemplo, límite de requests del plan gratuito
            r["error"] = str(e)
        time.sleep(1)  # para no pasarse del límite de requests por minuto de Groq


def porcentaje(parte: int, total: int) -> str:
    return f"{parte}/{total} ({100 * parte / total:.0f}%)" if total else "-"


def imprimir_reporte(resultados: list[dict], k: int, modelo: str, con_generacion: bool) -> None:
    con_respuesta = [r for r in resultados if r.get("debe_contener")]

    print("\nDetalle por pregunta")
    print("-" * 80)
    for r in resultados:
        if r.get("debe_contener"):
            estado = f"posición {r['posicion']}" if r["posicion"] else f"NO está en top {k}"
        else:
            estado = "sin respuesta en el doc"
        linea = f"[{r['tipo']:<13}] {estado:<22} {r['latencia_s'] * 1000:>5.0f} ms  {r['pregunta']}"
        print(linea)
        if con_generacion and "error" in r:
            print(f"    ERROR del LLM: {r['error'][:120]}")
        elif con_generacion:
            esperado = not r.get("debe_contener")
            if r["se_abstuvo"] != esperado:
                print(f"    OJO respuesta: {r['respuesta'][:150]!r}")

    print("\n" + "=" * 80)
    print(f"RESUMEN  |  modelo de embeddings: {modelo}  |  k = {k}")
    print("=" * 80)
    for tipo, etiqueta in [("literal", "literales"), ("parafraseada", "parafraseadas")]:
        grupo = [r for r in con_respuesta if r["tipo"] == tipo]
        if grupo:
            aciertos = sum(1 for r in grupo if r["posicion"])
            print(f"Acierto@{k} en preguntas {etiqueta}: {porcentaje(aciertos, len(grupo))}")

    total = len(con_respuesta)
    acierto_1 = sum(1 for r in con_respuesta if r["posicion"] == 1)
    acierto_k = sum(1 for r in con_respuesta if r["posicion"])
    mrr = sum(1 / r["posicion"] for r in con_respuesta if r["posicion"]) / total if total else 0
    latencias = [r["latencia_s"] * 1000 for r in resultados]

    print(f"Acierto@1 total:  {porcentaje(acierto_1, total)}")
    print(f"Acierto@{k} total:  {porcentaje(acierto_k, total)}")
    print(f"MRR:              {mrr:.2f}")
    print(f"Latencia de búsqueda: mediana {statistics.median(latencias):.0f} ms, "
          f"máxima {max(latencias):.0f} ms")

    if con_generacion:
        ok = [r for r in resultados if "error" not in r]
        con_resp = [r for r in ok if r.get("debe_contener")]
        sin_resp = [r for r in ok if not r.get("debe_contener")]
        respondio = sum(1 for r in con_resp if not r["se_abstuvo"])
        se_abstuvo = sum(1 for r in sin_resp if r["se_abstuvo"])
        primeros = [r["primer_token_s"] * 1000 for r in ok if r.get("primer_token_s")]
        completas = [r["generacion_s"] for r in ok]

        print(f"\nRespondió cuando había respuesta:     {porcentaje(respondio, len(con_resp))}")
        print(f"Se abstuvo cuando no había respuesta: {porcentaje(se_abstuvo, len(sin_resp))}")
        if primeros:
            print(f"Tiempo hasta el primer token: mediana {statistics.median(primeros):.0f} ms")
        if completas:
            print(f"Respuesta completa: mediana {statistics.median(completas):.2f} s")
        errores = len(resultados) - len(ok)
        if errores:
            print(f"Preguntas con error del LLM (no contadas): {errores}")


def main():
    parser = argparse.ArgumentParser(description="Evaluación de Buscarag")
    parser.add_argument("--documento", default=DOCUMENTO_DEFAULT)
    parser.add_argument("--preguntas", default=PREGUNTAS_DEFAULT)
    parser.add_argument("--modelo", default=MODELO_DEFAULT, help="modelo de sentence-transformers")
    parser.add_argument("--k", type=int, default=4, help="fragmentos recuperados (la app usa 4)")
    parser.add_argument("--generacion", action="store_true", help="evaluar también las respuestas del LLM")
    args = parser.parse_args()

    # Evita errores con tildes en algunas terminales de Windows (Git Bash, por ejemplo)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    with open(args.preguntas, encoding="utf-8") as f:
        preguntas = json.load(f)

    print(f"Indexando {os.path.basename(args.documento)} con {args.modelo}...")
    coleccion, chunks = indexar_documento(args.documento, args.modelo)
    print(f"  {len(chunks)} fragmentos indexados")

    preguntas = validar_preguntas(preguntas, chunks)
    print(f"Evaluando {len(preguntas)} preguntas...")
    resultados = evaluar_busqueda(coleccion, preguntas, args.k)

    if args.generacion:
        print("Generando respuestas con el LLM (puede tardar un minuto)...")
        evaluar_generacion(resultados)

    imprimir_reporte(resultados, args.k, args.modelo, args.generacion)


if __name__ == "__main__":
    main()