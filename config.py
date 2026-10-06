"""Configuración central del proyecto.

Todas las constantes (modelo de embeddings, dimensión, índice, namespace, chunking)
viven acá para que los scripts no hardcodeen valores por su cuenta. En particular,
la dimensión del índice se DERIVA del modelo de embeddings elegido: así evitamos el
clásico "mismatch de dimensiones" (subir vectores de 1536 a un índice de 384).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

RAIZ = Path(__file__).resolve().parent
DATA_DIR = RAIZ / "data"
GOLDEN_SET_PATH = RAIZ / "golden_set.json"

# --- Embeddings -------------------------------------------------------------
# proveedor -> (modelo, dimensión)
MODELOS_EMBEDDING = {
    # Por defecto. API key gratuita en https://aistudio.google.com/apikey. Multilingüe.
    # gemini-embedding-001 genera 3072 dims, pero admite reducirlas (Matryoshka):
    # 768 da casi la misma calidad con 4 veces menos almacenamiento en Pinecone.
    "google": ("gemini-embedding-001", 768),
    # Alternativa local y gratis, sin API key (requiere requirements-huggingface.txt).
    "huggingface": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 384),
    # Requiere OPENAI_API_KEY (es el modelo que sugiere la consigna).
    "openai": ("text-embedding-3-small", 1536),
}

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "google").strip().lower()
if EMBEDDING_PROVIDER not in MODELOS_EMBEDDING:
    raise ValueError(
        f"EMBEDDING_PROVIDER='{EMBEDDING_PROVIDER}' no es válido. "
        f"Opciones: {', '.join(MODELOS_EMBEDDING)}"
    )
EMBEDDING_MODEL, EMBEDDING_DIM = MODELOS_EMBEDDING[EMBEDDING_PROVIDER]

# --- Pinecone ---------------------------------------------------------------
INDEX_NAME = os.getenv("INDEX_NAME", "fastapi-docs-rag")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "fastapi-docs")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
METRICA = "cosine"

# --- Chunking ---------------------------------------------------------------
CHUNK_SIZE = 600  # tokens: dentro del rango 500-800 sugerido
CHUNK_OVERLAP = 100

# --- Recuperación -----------------------------------------------------------
TOP_K = 5
# Cada retriever trae más candidatos que el top-k final para que la fusión (RRF)
# tenga material para reordenar.
CANDIDATOS_POR_RETRIEVER = 10
PESO_BM25 = float(os.getenv("PESO_BM25", "0.5"))
PESO_VECTORIAL = 1.0 - PESO_BM25


def requerir_variable(nombre: str) -> str:
    """Devuelve una variable de entorno obligatoria o corta con un mensaje claro."""
    valor = os.getenv(nombre, "").strip()
    if not valor:
        raise RuntimeError(
            f"Falta la variable de entorno {nombre}. Copiá .env.example a .env y completala."
        )
    return valor


def obtener_embeddings():
    """Instancia el modelo de embeddings configurado (import perezoso: cada proveedor
    trae dependencias pesadas que no hace falta cargar si no se usan)."""
    if EMBEDDING_PROVIDER == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        if not os.getenv("GEMINI_API_KEY", "").strip():
            requerir_variable("GOOGLE_API_KEY")
        # task_type: la librería ya usa RETRIEVAL_DOCUMENT al indexar y RETRIEVAL_QUERY
        # al consultar, que es lo que Google recomienda para búsqueda.
        return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, output_dimensionality=EMBEDDING_DIM)

    if EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        requerir_variable("OPENAI_API_KEY")
        return OpenAIEmbeddings(model=EMBEDDING_MODEL)

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
