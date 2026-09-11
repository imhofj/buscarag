"""Tests del armado del prompt y del pipeline con streaming (rag.py)."""

from types import SimpleNamespace

import rag


FRAGMENTOS = [
    {"text": "El plazo máximo es de 12 cuotas.", "source": "politica.pdf"},
    {"text": "La quita máxima es del 30%.", "source": "politica.pdf"},
]


def chunk_falso(texto):
    """Imita un pedacito de respuesta en streaming de la API de Groq."""
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=texto))])


class ClienteFalso:
    """Imita al cliente de Groq: devuelve una respuesta fija y guarda lo que recibió."""

    def __init__(self, pedazos):
        self.pedazos = pedazos
        self.ultimo_pedido = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.ultimo_pedido = kwargs
        return iter(self.pedazos)


# ---------- build_prompt ----------

def test_build_prompt_incluye_pregunta_fragmentos_y_fuentes():
    prompt = rag.build_prompt("¿Cuántas cuotas como máximo?", FRAGMENTOS)

    assert "¿Cuántas cuotas como máximo?" in prompt
    assert "El plazo máximo es de 12 cuotas." in prompt
    assert "La quita máxima es del 30%." in prompt
    assert "[Fuente: politica.pdf]" in prompt


def test_build_prompt_indica_la_frase_exacta_para_cuando_no_hay_respuesta():
    prompt = rag.build_prompt("pregunta", FRAGMENTOS)

    assert "No encontré esa información en los documentos." in prompt


# ---------- answer_question_stream ----------

def test_stream_sin_documentos_no_llama_al_modelo(monkeypatch):
    cliente = ClienteFalso([])
    monkeypatch.setattr(rag, "search", lambda query, n_results=4: [])
    monkeypatch.setattr(rag, "client", cliente)

    fragmentos, stream, metricas = rag.answer_question_stream("¿algo?")

    assert fragmentos == []
    assert stream is None
    assert "search_s" in metricas
    assert cliente.ultimo_pedido is None


def test_stream_devuelve_la_respuesta_completa_y_las_metricas(monkeypatch):
    pedazos = [
        chunk_falso("El plazo "),
        chunk_falso(None),                 # pedazo sin texto (pasa con modelos de razonamiento)
        SimpleNamespace(choices=[]),       # pedazo final sin choices
        chunk_falso("es de 12 cuotas."),
    ]
    cliente = ClienteFalso(pedazos)
    monkeypatch.setattr(rag, "search", lambda query, n_results=4: FRAGMENTOS)
    monkeypatch.setattr(rag, "client", cliente)

    fragmentos, stream, metricas = rag.answer_question_stream("¿Cuántas cuotas?")
    respuesta = "".join(stream)

    assert respuesta == "El plazo es de 12 cuotas."
    assert fragmentos == FRAGMENTOS
    assert cliente.ultimo_pedido["stream"] is True
    assert "12 cuotas" in cliente.ultimo_pedido["messages"][0]["content"]
    assert 0 <= metricas["first_token_s"] <= metricas["generation_s"]
