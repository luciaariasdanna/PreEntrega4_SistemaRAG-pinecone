"""Fixtures compartidas. Ningún test usa red: Pinecone y los embeddings se reemplazan
por dobles en memoria, así la suite corre sin API keys."""

from types import SimpleNamespace

import pytest
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.retrievers import BaseRetriever


def crear_chunk(chunk_id: str, texto: str, source: str, categoria: str = "general", page: int = 1) -> Document:
    return Document(
        id=chunk_id,
        page_content=texto,
        metadata={"source": source, "categoria": categoria, "page": page,
                  "chunk_id": chunk_id, "etiquetas": []},
    )


@pytest.fixture
def corpus() -> list[Document]:
    return [
        crear_chunk("cors-c0", "CORSMiddleware permite configurar allow_origins y allow_credentials.",
                    "middleware_y_cors.md", "avanzado"),
        crear_chunk("oauth-c0", "OAuth2PasswordBearer recibe tokenUrl y devuelve el token Bearer.",
                    "autenticacion_oauth2_jwt.md", "seguridad"),
        crear_chunk("test-c0", "TestClient permite probar endpoints sin levantar el servidor.",
                    "testing_con_testclient.md", "operaciones"),
        crear_chunk("deps-c0", "Depends inyecta dependencias; con yield se cierra la sesión de base de datos.",
                    "inyeccion_de_dependencias.md", "avanzado"),
        crear_chunk("bg-c0", "BackgroundTasks ejecuta tareas después de enviar la respuesta, como un email.",
                    "tareas_en_segundo_plano.md", "avanzado"),
        crear_chunk("val-c0", "Si falla la validación FastAPI responde 422 Unprocessable Entity.",
                    "parametros_de_ruta_y_query.md", "fundamentos"),
    ]


class RetrieverFijo(BaseRetriever):
    """Retriever vectorial falso: devuelve siempre los mismos documentos, en orden.
    Si se le asigna `filtro` (como hace RAGSystem), filtra por categoría."""

    documentos: list[Document]
    filtro: dict | None = None

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        if not self.filtro:
            return list(self.documentos)
        categoria = self.filtro["categoria"]["$eq"]
        return [d for d in self.documentos if d.metadata["categoria"] == categoria]


@pytest.fixture
def retriever_fijo(corpus) -> RetrieverFijo:
    # Orden "semántico" arbitrario: el de testing primero.
    return RetrieverFijo(documentos=[corpus[2], corpus[4], corpus[0]])


class IndiceFalso:
    """Doble de `pinecone.Index` en memoria con la misma interfaz que usamos."""

    def __init__(self):
        self.vectores: dict[str, dict[str, dict]] = {}
        self.ultima_query: dict | None = None

    def upsert(self, vectors, namespace):
        espacio = self.vectores.setdefault(namespace, {})
        for v in vectors:
            espacio[v["id"]] = v

    def list(self, namespace):
        # Igual que pinecone>=10: un ListResponse por página con `vectors: list[ListItem]`.
        ids = sorted(self.vectores.get(namespace, {}))
        for inicio in range(0, len(ids), 2):  # páginas chicas para ejercitar la paginación
            yield SimpleNamespace(vectors=[SimpleNamespace(id=i) for i in ids[inicio:inicio + 2]])

    def fetch(self, ids, namespace):
        espacio = self.vectores[namespace]
        return SimpleNamespace(vectors={
            i: SimpleNamespace(id=i, metadata=espacio[i]["metadata"]) for i in ids
        })

    def query(self, vector, top_k, namespace, include_metadata, filter):
        self.ultima_query = {"vector": vector, "top_k": top_k, "namespace": namespace, "filter": filter}
        espacio = list(self.vectores.get(namespace, {}).values())[:top_k]
        return SimpleNamespace(matches=[
            SimpleNamespace(id=v["id"], score=0.9 - 0.1 * i, metadata=dict(v["metadata"]))
            for i, v in enumerate(espacio)
        ])


@pytest.fixture
def indice_falso() -> IndiceFalso:
    return IndiceFalso()


@pytest.fixture
def embeddings_falsos() -> DeterministicFakeEmbedding:
    return DeterministicFakeEmbedding(size=8)
