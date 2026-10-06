import json

import pytest

from evaluate import (
    cargar_golden_set,
    conclusiones,
    documentos_relevantes,
    evaluar,
    mrr_at_k,
    precision_at_k,
    precision_maxima,
    recall_at_k,
)
from rag_system import RAGSystem

RECUPERADOS = ["a.md", "b.md", "a.md", "c.md", "d.md"]


@pytest.mark.parametrize(("relevantes", "esperado"), [
    ({"a.md"}, 1.0),
    ({"z.md"}, 0.0),
    ({"a.md", "z.md"}, 0.5),
])
def test_recall_at_k(relevantes, esperado):
    assert recall_at_k(RECUPERADOS, relevantes, k=5) == esperado


def test_recall_respeta_el_corte_k():
    assert recall_at_k(RECUPERADOS, {"d.md"}, k=4) == 0.0


def test_precision_at_k_cuenta_chunks_relevantes():
    assert precision_at_k(RECUPERADOS, {"a.md"}, k=5) == pytest.approx(2 / 5)
    assert precision_at_k(RECUPERADOS, {"a.md", "b.md"}, k=5) == pytest.approx(3 / 5)


def test_precision_divide_por_k_aunque_haya_menos_resultados():
    assert precision_at_k(["a.md"], {"a.md"}, k=5) == pytest.approx(1 / 5)


def test_mrr_at_k():
    assert mrr_at_k(RECUPERADOS, {"a.md"}, k=5) == 1.0
    assert mrr_at_k(RECUPERADOS, {"c.md"}, k=5) == pytest.approx(1 / 4)
    assert mrr_at_k(RECUPERADOS, {"z.md"}, k=5) == 0.0


def test_precision_maxima_limitada_por_chunks_disponibles():
    assert precision_maxima({"a.md": 2, "b.md": 9}, {"a.md"}, k=5) == pytest.approx(2 / 5)
    assert precision_maxima({"a.md": 2, "b.md": 9}, {"b.md"}, k=5) == 1.0


def test_documentos_relevantes_admite_lista_opcional():
    assert documentos_relevantes({"documento_id_esperado": "a.md"}) == {"a.md"}
    assert documentos_relevantes({"documento_id_esperado": "a.md",
                                  "documentos_relevantes": ["a.md", "b.md"]}) == {"a.md", "b.md"}


def test_cargar_golden_set_valida_campos(tmp_path):
    ruta = tmp_path / "golden.json"
    ruta.write_text(json.dumps([{"pregunta": "¿?"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="inválido"):
        cargar_golden_set(ruta)


def _reporte(modo, recall, precision, mrr):
    return {"modo": modo, "k": 5, "recall_promedio": recall, "precision_promedio": precision,
            "precision_max_promedio": 0.5, "mrr_promedio": mrr,
            "detalle": [{"recall": 1.0, "mrr": 1.0}, {"recall": 0.0, "mrr": 0.0}]}


def test_conclusiones_resume_cada_modo_y_elige_el_mejor():
    lineas = conclusiones([_reporte("bm25", 0.5, 0.2, 0.5), _reporte("hibrido", 0.5, 0.25, 0.5)])
    assert "1/2 preguntas" in lineas[0]
    assert "40% del máximo" in lineas[0]
    assert lineas[-1].endswith("hibrido")  # empate en recall y MRR: desempata la precisión


def test_evaluar_end_to_end_con_dobles(retriever_fijo, corpus):
    rag = RAGSystem(retriever_fijo, corpus, k=5)
    golden = [
        {"pregunta": "OAuth2PasswordBearer tokenUrl", "documento_id_esperado": "autenticacion_oauth2_jwt.md"},
        {"pregunta": "422 Unprocessable Entity", "documento_id_esperado": "parametros_de_ruta_y_query.md"},
    ]
    reporte = evaluar(rag, golden, modo="bm25")

    assert reporte["modo"] == "bm25" and reporte["k"] == 5
    assert reporte["recall_promedio"] == 1.0
    assert reporte["mrr_promedio"] == 1.0
    assert reporte["precision_max_promedio"] == pytest.approx(1 / 5)  # 1 chunk por documento
    assert [d["recuperados"][0] for d in reporte["detalle"]] == [
        "autenticacion_oauth2_jwt.md", "parametros_de_ruta_y_query.md"]
