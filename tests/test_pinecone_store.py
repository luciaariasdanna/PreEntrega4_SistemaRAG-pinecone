from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from pinecone_store import (
    CLAVE_TEXTO,
    DimensionMismatchError,
    PineconeRetriever,
    a_documento,
    asegurar_indice,
    cargar_corpus,
    preparar_vectores,
    subir_chunks,
)


class ClienteFalso:
    def __init__(self, indices: dict[str, int] | None = None):
        self.indices = dict(indices or {})
        self.creados = []

    def has_index(self, nombre):
        return nombre in self.indices

    def describe_index(self, nombre):
        return SimpleNamespace(dimension=self.indices[nombre])

    def create_index(self, name, dimension, metric, spec):
        self.indices[name] = dimension
        self.creados.append({"name": name, "dimension": dimension, "metric": metric, "spec": spec})

    def Index(self, nombre):
        return f"index:{nombre}"


def test_asegurar_indice_crea_serverless_si_no_existe():
    pc = ClienteFalso()
    index, creado = asegurar_indice(pc, nombre="demo", dimension=384, metrica="cosine",
                                    cloud="aws", region="us-east-1")
    assert creado is True
    assert index == "index:demo"
    [spec] = pc.creados
    assert spec["dimension"] == 384 and spec["metric"] == "cosine"
    assert spec["spec"].cloud == "aws" and spec["spec"].region == "us-east-1"


def test_asegurar_indice_reutiliza_si_la_dimension_coincide():
    pc = ClienteFalso({"demo": 384})
    _, creado = asegurar_indice(pc, nombre="demo", dimension=384)
    assert creado is False
    assert pc.creados == []


def test_asegurar_indice_detecta_mismatch_de_dimensiones():
    pc = ClienteFalso({"demo": 1536})
    with pytest.raises(DimensionMismatchError, match="1536"):
        asegurar_indice(pc, nombre="demo", dimension=384)


def test_preparar_vectores_guarda_el_texto_en_metadata(corpus):
    [registro] = preparar_vectores(corpus[:1], [[0.1, 0.2]])
    assert registro["id"] == "cors-c0"
    assert registro["values"] == [0.1, 0.2]
    assert registro["metadata"][CLAVE_TEXTO] == corpus[0].page_content
    assert registro["metadata"]["source"] == "middleware_y_cors.md"
    assert CLAVE_TEXTO not in corpus[0].metadata  # no muta el documento original


def test_a_documento_separa_texto_y_agrega_score():
    doc = a_documento("id-1", {CLAVE_TEXTO: "hola", "source": "a.md"}, score=0.5)
    assert doc == Document(id="id-1", page_content="hola",
                           metadata={"source": "a.md", "chunk_id": "id-1", "score": 0.5})


def test_subida_y_corpus_ida_y_vuelta(indice_falso, embeddings_falsos, corpus):
    total = subir_chunks(indice_falso, corpus, embeddings_falsos, namespace="ns", tamanio_lote=4)
    assert total == len(corpus)

    recuperado = cargar_corpus(indice_falso, namespace="ns")
    assert sorted(d.id for d in corpus) == [d.id for d in recuperado]
    por_id = {d.id: d for d in recuperado}
    for original in corpus:
        assert por_id[original.id].page_content == original.page_content
        assert por_id[original.id].metadata["source"] == original.metadata["source"]


def test_reingestar_no_duplica_vectores(indice_falso, embeddings_falsos, corpus):
    subir_chunks(indice_falso, corpus, embeddings_falsos, namespace="ns")
    subir_chunks(indice_falso, corpus, embeddings_falsos, namespace="ns")
    assert len(indice_falso.vectores["ns"]) == len(corpus)


def test_pinecone_retriever_consulta_namespace_y_filtro(indice_falso, embeddings_falsos, corpus):
    subir_chunks(indice_falso, corpus, embeddings_falsos, namespace="ns")
    retriever = PineconeRetriever(index=indice_falso, embeddings=embeddings_falsos, namespace="ns", k=3,
                                  filtro={"categoria": {"$eq": "avanzado"}})
    docs = retriever.invoke("cors")

    assert len(docs) == 3
    assert all(d.page_content for d in docs)
    assert indice_falso.ultima_query["namespace"] == "ns"
    assert indice_falso.ultima_query["top_k"] == 3
    assert indice_falso.ultima_query["filter"] == {"categoria": {"$eq": "avanzado"}}
    assert len(indice_falso.ultima_query["vector"]) == 8
