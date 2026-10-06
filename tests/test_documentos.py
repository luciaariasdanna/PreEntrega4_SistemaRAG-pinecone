import json

import pytest
import tiktoken

import config
from documentos import (
    cargar_documentos,
    cargar_json,
    cargar_markdown,
    dividir_en_chunks,
    generar_chunk_id,
    parsear_front_matter,
)

TIPOS_METADATA_PINECONE = (str, int, float, bool)


def test_parsear_front_matter_extrae_listas_y_strings():
    texto = "---\ntitulo: Mi doc\netiquetas: [a, b , c]\n---\n# Cuerpo\n"
    meta, cuerpo = parsear_front_matter(texto)
    assert meta == {"titulo": "Mi doc", "etiquetas": ["a", "b", "c"]}
    assert cuerpo == "# Cuerpo\n"


def test_parsear_front_matter_sin_bloque_devuelve_texto_intacto():
    assert parsear_front_matter("# Solo cuerpo") == ({}, "# Solo cuerpo")


def test_cargar_markdown_usa_front_matter(tmp_path):
    ruta = tmp_path / "guia.md"
    ruta.write_text("---\ntitulo: Guía\netiquetas: [x]\n---\nContenido", encoding="utf-8")
    [doc] = cargar_markdown(ruta, "fundamentos")
    assert doc.page_content == "Contenido"
    assert doc.metadata == {"source": "guia.md", "categoria": "fundamentos", "tipo": "markdown",
                            "titulo": "Guía", "etiquetas": ["x"], "page": 1}


def test_cargar_json_una_pagina_por_entrada_y_etiquetas_combinadas(tmp_path):
    ruta = tmp_path / "faq.json"
    ruta.write_text(json.dumps({
        "titulo": "FAQ", "etiquetas": ["faq"],
        "entradas": [{"pregunta": "¿A?", "respuesta": "B", "etiquetas": ["t1"]},
                     {"pregunta": "¿C?", "respuesta": "D"}],
    }), encoding="utf-8")
    docs = cargar_json(ruta, "faq")
    assert [d.metadata["page"] for d in docs] == [1, 2]
    assert docs[0].metadata["etiquetas"] == ["faq", "t1"]
    assert docs[1].page_content == "Pregunta: ¿C?\nRespuesta: D"


def test_generar_chunk_id_es_ascii_y_determinista():
    chunk_id = generar_chunk_id("Guía de Despliegue.pdf", 2, 0)
    assert chunk_id == generar_chunk_id("Guía de Despliegue.pdf", 2, 0)
    assert chunk_id.isascii()
    assert chunk_id.endswith("-p2-c0")


# --- Sobre el dataset real de data/ -----------------------------------------

@pytest.fixture(scope="module")
def documentos_reales():
    return cargar_documentos(config.DATA_DIR)


@pytest.fixture(scope="module")
def chunks_reales(documentos_reales):
    return dividir_en_chunks(documentos_reales)


def test_dataset_incluye_los_tres_formatos(documentos_reales):
    assert {d.metadata["tipo"] for d in documentos_reales} == {"markdown", "json", "pdf"}


def test_pdf_conserva_numero_de_pagina(documentos_reales):
    paginas = [d.metadata["page"] for d in documentos_reales if d.metadata["tipo"] == "pdf"]
    assert paginas == [1, 2, 3]


def test_categoria_sale_de_la_subcarpeta(documentos_reales):
    categorias = {d.metadata["source"]: d.metadata["categoria"] for d in documentos_reales}
    assert categorias["autenticacion_oauth2_jwt.md"] == "seguridad"
    assert categorias["preguntas_frecuentes.json"] == "faq"


def test_chunk_ids_unicos(chunks_reales):
    ids = [c.metadata["chunk_id"] for c in chunks_reales]
    assert len(ids) == len(set(ids))
    assert all(c.id == c.metadata["chunk_id"] for c in chunks_reales)


def test_chunks_respetan_el_tamanio_en_tokens(chunks_reales):
    encoding = tiktoken.get_encoding("cl100k_base")
    assert max(len(encoding.encode(c.page_content)) for c in chunks_reales) <= config.CHUNK_SIZE


def test_metadata_compatible_con_pinecone(chunks_reales):
    """Pinecone solo acepta str, números, bool y listas de str (no None ni dicts)."""
    for chunk in chunks_reales:
        for clave, valor in chunk.metadata.items():
            if isinstance(valor, list):
                assert all(isinstance(v, str) for v in valor), clave
            else:
                assert isinstance(valor, TIPOS_METADATA_PINECONE), (clave, valor)


def test_golden_set_apunta_a_documentos_existentes(documentos_reales):
    fuentes = {d.metadata["source"] for d in documentos_reales}
    golden = json.loads(config.GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    assert len(golden) >= 5
    for caso in golden:
        assert caso["documento_id_esperado"] in fuentes
        assert set(caso.get("documentos_relevantes", [])) <= fuentes
