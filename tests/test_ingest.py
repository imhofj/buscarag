"""Tests de extracción de texto, chunking e ingesta (ingest.py)."""

from pypdf import PdfWriter

from ingest import extract_text, chunk_text, ingest_file, CHUNK_SIZE, CHUNK_OVERLAP
from conftest import PDF_EJEMPLO


# ---------- extract_text ----------

def test_extract_text_txt_normaliza_espacios_y_saltos_de_linea(tmp_path):
    archivo = tmp_path / "notas.txt"
    archivo.write_text("Hola   mundo\n\ncon    espacios\tde más", encoding="utf-8")

    assert extract_text(str(archivo)) == "Hola mundo con espacios de más"


def test_extract_text_pdf_de_ejemplo_conserva_tildes_y_contenido():
    texto = extract_text(PDF_EJEMPLO)

    assert "Financiera Ceibo" in texto
    assert "El plazo máximo de un acuerdo es de 12 cuotas" in texto
    assert "  " not in texto  # no quedan espacios dobles


# ---------- chunk_text ----------

def test_chunk_text_ningun_fragmento_supera_el_tamanio():
    texto = "palabra " * 1000

    chunks = chunk_text(texto)

    assert len(chunks) > 1
    assert all(len(c) <= CHUNK_SIZE for c in chunks)


def test_chunk_text_fragmentos_consecutivos_se_superponen():
    texto = "abcdefghij" * 200  # sin espacios, para que strip() no altere los bordes

    chunks = chunk_text(texto)

    for anterior, siguiente in zip(chunks, chunks[1:]):
        assert anterior[-CHUNK_OVERLAP:] == siguiente[:CHUNK_OVERLAP]


def test_chunk_text_no_pierde_texto():
    texto = "abcdefghij" * 200

    chunks = chunk_text(texto)
    reconstruido = chunks[0] + "".join(c[CHUNK_OVERLAP:] for c in chunks[1:])

    assert reconstruido == texto


def test_chunk_text_texto_corto_devuelve_un_solo_fragmento():
    assert chunk_text("Un texto corto.") == ["Un texto corto."]


def test_chunk_text_texto_vacio_no_devuelve_fragmentos():
    assert chunk_text("") == []


# ---------- ingest_file ----------

def test_ingest_file_guarda_fragmentos_con_el_nombre_real_del_archivo(tmp_path, coleccion_en_memoria):
    archivo = tmp_path / "contrato.txt"
    archivo.write_text("Cláusula de prueba. " * 100, encoding="utf-8")

    cantidad = ingest_file(str(archivo))

    guardado = coleccion_en_memoria.get()
    assert cantidad > 1
    assert coleccion_en_memoria.count() == cantidad
    assert all(m["source"] == "contrato.txt" for m in guardado["metadatas"])
    assert len(set(guardado["ids"])) == cantidad  # ids únicos


def test_ingest_file_pdf_sin_texto_devuelve_cero_sin_romper(tmp_path, coleccion_en_memoria):
    # Simula un PDF escaneado: tiene una página, pero ningún texto seleccionable
    pdf = tmp_path / "escaneado.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    with open(pdf, "wb") as f:
        writer.write(f)

    assert ingest_file(str(pdf)) == 0
    assert coleccion_en_memoria.count() == 0
