"""Capa de acceso a Pinecone usando el SDK nativo.

Usamos el SDK oficial (`pinecone`) en lugar de `langchain-pinecone` porque este último
todavía no soporta Python 3.14 (fija numpy<2). La consigna admite ambas opciones, y
envolver el SDK en un `BaseRetriever` propio deja explícito qué se envía y qué vuelve.

Clave del diseño: el texto del chunk se guarda en `metadata["text"]` junto al vector.
Una sola consulta a Pinecone devuelve vector + texto + fuente, sin base relacional.
"""

from typing import Any, Iterable

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

import config

CLAVE_TEXTO = "text"
TAMANIO_LOTE = 100  # vectores por upsert (Pinecone recomienda lotes de <= 1000 / 2 MB)


class DimensionMismatchError(RuntimeError):
    """El índice existe pero con una dimensión distinta a la del modelo de embeddings."""


def obtener_cliente():
    from pinecone import Pinecone

    return Pinecone(api_key=config.requerir_variable("PINECONE_API_KEY"))


def asegurar_indice(pc, nombre: str = config.INDEX_NAME, dimension: int = config.EMBEDDING_DIM,
                    metrica: str = config.METRICA, cloud: str = config.PINECONE_CLOUD,
                    region: str = config.PINECONE_REGION) -> tuple[Any, bool]:
    """Crea el índice Serverless si no existe y valida la dimensión si ya existe.

    Devuelve (index, creado).
    """
    if pc.has_index(nombre):
        dimension_actual = pc.describe_index(nombre).dimension
        if dimension_actual != dimension:
            raise DimensionMismatchError(
                f"El índice '{nombre}' tiene dimensión {dimension_actual}, pero el modelo "
                f"'{config.EMBEDDING_MODEL}' genera vectores de {dimension}. Usá otro "
                f"INDEX_NAME o borrá el índice desde la consola de Pinecone."
            )
        return pc.Index(nombre), False

    from pinecone import ServerlessSpec

    pc.create_index(
        name=nombre,
        dimension=dimension,
        metric=metrica,
        spec=ServerlessSpec(cloud=cloud, region=region),
    )  # bloquea hasta que el índice está listo
    return pc.Index(nombre), True


def abrir_indice_existente(pc, nombre: str = config.INDEX_NAME):
    if not pc.has_index(nombre):
        raise RuntimeError(
            f"El índice '{nombre}' no existe. Corré primero `python setup_index.py` "
            f"y `python ingest.py`."
        )
    return pc.Index(nombre)


# --- Escritura --------------------------------------------------------------

def preparar_vectores(chunks: list[Document], vectores: list[list[float]]) -> list[dict]:
    """Arma los registros de upsert: id determinista + valores + metadata con el texto."""
    registros = []
    for chunk, valores in zip(chunks, vectores, strict=True):
        metadata = dict(chunk.metadata)
        metadata[CLAVE_TEXTO] = chunk.page_content
        registros.append({"id": chunk.metadata["chunk_id"], "values": valores, "metadata": metadata})
    return registros


def _lotes(elementos: list, tamanio: int) -> Iterable[list]:
    for inicio in range(0, len(elementos), tamanio):
        yield elementos[inicio:inicio + tamanio]


def subir_chunks(index, chunks: list[Document], embeddings: Embeddings,
                 namespace: str = config.NAMESPACE, tamanio_lote: int = TAMANIO_LOTE) -> int:
    """Genera embeddings por lotes y hace upsert en el namespace. Devuelve el total subido."""
    total = 0
    for lote in _lotes(chunks, tamanio_lote):
        vectores = embeddings.embed_documents([c.page_content for c in lote])
        index.upsert(vectors=preparar_vectores(lote, vectores), namespace=namespace)
        total += len(lote)
    return total


def vaciar_namespace(index, namespace: str = config.NAMESPACE) -> bool:
    """Borra todos los vectores del namespace. Devuelve False si el namespace no existía."""
    from pinecone.exceptions import NotFoundException

    try:
        index.delete(delete_all=True, namespace=namespace)
        return True
    except NotFoundException:
        return False


# --- Lectura ----------------------------------------------------------------

def a_documento(vector_id: str, metadata: dict | None, score: float | None = None) -> Document:
    """Reconstruye un Document de LangChain a partir de lo que devuelve Pinecone."""
    metadata = dict(metadata or {})
    texto = metadata.pop(CLAVE_TEXTO, "")
    metadata.setdefault("chunk_id", vector_id)
    if score is not None:
        metadata["score"] = float(score)
    return Document(id=vector_id, page_content=texto, metadata=metadata)


def cargar_corpus(index, namespace: str = config.NAMESPACE) -> list[Document]:
    """Descarga todos los chunks del namespace (texto incluido, gracias a la metadata).

    Lo usa el BM25: así el índice léxico se construye desde la MISMA fuente de verdad
    que el vectorial, sin depender de tener los archivos originales en disco.
    """
    documentos = []
    # index.list() pagina solo y devuelve un ListResponse por página (máx. 100 IDs).
    for pagina in index.list(namespace=namespace):
        ids = [item.id for item in pagina.vectors if item.id]
        if not ids:
            continue
        respuesta = index.fetch(ids=ids, namespace=namespace)
        for vector_id, vector in respuesta.vectors.items():
            documentos.append(a_documento(vector_id, vector.metadata))
    return sorted(documentos, key=lambda d: d.id)


class PineconeRetriever(BaseRetriever):
    """Retriever semántico: embebe la consulta y busca los k vecinos más cercanos."""

    index: Any
    embeddings: Embeddings
    namespace: str = config.NAMESPACE
    k: int = config.CANDIDATOS_POR_RETRIEVER
    filtro: dict | None = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        respuesta = self.index.query(
            vector=self.embeddings.embed_query(query),
            top_k=self.k,
            namespace=self.namespace,
            include_metadata=True,
            filter=self.filtro,
        )
        return [a_documento(m.id, m.metadata, m.score) for m in respuesta.matches]
