"""Recuperador híbrido: BM25 (léxico) + Pinecone (semántico) fusionados con un EnsembleRetriever.

¿Por qué híbrido? La búsqueda vectorial entiende parafraseos ("probar endpoints sin
levantar el servidor" ≈ TestClient) pero es débil con identificadores exactos
(`OAuth2PasswordBearer`, `CORSMiddleware`, `422`). BM25 es exactamente al revés.
El EnsembleRetriever combina ambos rankings con Reciprocal Rank Fusion (RRF).
"""

import re
import unicodedata

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

import config

MODOS = ("hibrido", "bm25", "vectorial")

STOPWORDS_ES = frozenset("""
a al algo algun alguna algunas alguno algunos ante antes asi aun cada como con contra
cual cuales cuando de del desde donde dos e el ella ellas ellos en entre era es esa esas
ese eso esos esta estan estas este esto estos fue fueron ha hay hace hacer la las le les
lo los mas me mi mis muy ni no nos o otra otro para pero poco por porque puede que quien
se sea segun ser si sin sobre son su sus tambien tan te tiene tienen todo todos tu un una
uno unos y ya yo
""".split())


def tokenizar(texto: str) -> list[str]:
    """Preprocesamiento para BM25 en castellano.

    El BM25Retriever por defecto hace `texto.split()`: "¿Qué" y "qué" quedarían como
    tokens distintos y "FastAPI?" no coincidiría con "FastAPI". Acá pasamos a minúsculas,
    quitamos tildes, separamos por caracteres no alfanuméricos (conservando `_` para
    identificadores como `status_code`) y descartamos stopwords.
    """
    sin_tildes = unicodedata.normalize("NFKD", texto.lower())
    sin_tildes = "".join(c for c in sin_tildes if not unicodedata.combining(c))
    return [t for t in re.findall(r"[a-z0-9_]+", sin_tildes) if t not in STOPWORDS_ES]


class BM25SinCeros(BM25Retriever):
    """BM25Retriever que descarta los documentos con puntaje 0.

    El original siempre devuelve k documentos, aunque no compartan ni un término con la
    consulta. En un EnsembleRetriever eso es ruido: RRF le da puntos a esos documentos solo
    por su posición en la lista.
    """

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        puntajes = self.vectorizer.get_scores(self.preprocess_func(query))
        orden = sorted(range(len(puntajes)), key=lambda i: puntajes[i], reverse=True)
        return [self.docs[i] for i in orden[: self.k] if puntajes[i] > 0]


def crear_retriever_bm25(corpus: list[Document], k: int = config.CANDIDATOS_POR_RETRIEVER) -> BM25Retriever:
    return BM25SinCeros.from_documents(corpus, k=k, preprocess_func=tokenizar)


class RAGSystem:
    """Encapsula el EnsembleRetriever y devuelve el top-k combinando BM25 + Pinecone."""

    def __init__(self, retriever_vectorial: BaseRetriever, corpus: list[Document],
                 k: int = config.TOP_K,
                 candidatos: int = config.CANDIDATOS_POR_RETRIEVER,
                 pesos: tuple[float, float] = (config.PESO_BM25, config.PESO_VECTORIAL)):
        if not corpus:
            raise ValueError("El corpus está vacío: corré `python ingest.py` antes de consultar.")
        self.retriever_vectorial = retriever_vectorial
        self.corpus = corpus
        self.k = k
        self.candidatos = candidatos
        self.pesos = list(pesos)
        self._bm25_por_categoria: dict[str | None, BM25Retriever] = {}

    @classmethod
    def desde_pinecone(cls, **kwargs) -> "RAGSystem":
        """Conecta al índice existente y arma el BM25 con el corpus guardado en Pinecone."""
        from pinecone_store import PineconeRetriever, abrir_indice_existente, cargar_corpus, obtener_cliente

        index = abrir_indice_existente(obtener_cliente())
        retriever_vectorial = PineconeRetriever(
            index=index,
            embeddings=config.obtener_embeddings(),
            namespace=config.NAMESPACE,
            k=kwargs.get("candidatos", config.CANDIDATOS_POR_RETRIEVER),
        )
        return cls(retriever_vectorial, cargar_corpus(index, config.NAMESPACE), **kwargs)

    @property
    def categorias(self) -> list[str]:
        return sorted({d.metadata.get("categoria", "general") for d in self.corpus})

    # --- Construcción de retrievers ------------------------------------------

    def _bm25(self, categoria: str | None) -> BM25Retriever:
        if categoria not in self._bm25_por_categoria:
            corpus = [d for d in self.corpus if categoria is None or d.metadata.get("categoria") == categoria]
            if not corpus:
                raise ValueError(f"No hay documentos con categoria='{categoria}'. Disponibles: {self.categorias}")
            self._bm25_por_categoria[categoria] = crear_retriever_bm25(corpus, self.candidatos)
        return self._bm25_por_categoria[categoria]

    def _vectorial(self, categoria: str | None) -> BaseRetriever:
        if categoria is None:
            return self.retriever_vectorial
        # El filtro por metadata se resuelve del lado de Pinecone (no post-filtrado).
        return self.retriever_vectorial.model_copy(update={"filtro": {"categoria": {"$eq": categoria}}})

    def retriever(self, modo: str = "hibrido", categoria: str | None = None) -> BaseRetriever:
        if modo == "bm25":
            return self._bm25(categoria)
        if modo == "vectorial":
            return self._vectorial(categoria)
        if modo == "hibrido":
            return EnsembleRetriever(
                retrievers=[self._bm25(categoria), self._vectorial(categoria)],
                weights=self.pesos,
                id_key="chunk_id",  # deduplica por ID de chunk, no por texto
            )
        raise ValueError(f"Modo '{modo}' inválido. Opciones: {MODOS}")

    # --- API pública ----------------------------------------------------------

    def buscar(self, consulta: str, modo: str = "hibrido", categoria: str | None = None) -> list[Document]:
        """Devuelve los top-k Documents para la consulta."""
        return self.retriever(modo, categoria).invoke(consulta)[: self.k]

    def obtener_top_k(self, consulta: str, modo: str = "hibrido", categoria: str | None = None) -> list[dict]:
        """Igual que `buscar`, pero en un formato plano y fácil de imprimir."""
        return [
            {
                "contenido": d.page_content,
                "fuente": d.metadata.get("source", "desconocida"),
                "pagina": d.metadata.get("page"),
                "categoria": d.metadata.get("categoria", "desconocida"),
                "etiquetas": d.metadata.get("etiquetas", []),
                "chunk_id": d.metadata.get("chunk_id"),
            }
            for d in self.buscar(consulta, modo, categoria)
        ]
