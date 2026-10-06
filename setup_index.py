"""Paso 1: verifica si el índice de Pinecone existe y, si no, lo crea en modo Serverless.

Uso:
    python setup_index.py
"""

import config
from pinecone_store import asegurar_indice, obtener_cliente


def main() -> None:
    print(f"🔧 Proveedor de embeddings: {config.EMBEDDING_PROVIDER} ({config.EMBEDDING_MODEL})")
    print(f"📐 Dimensión requerida:     {config.EMBEDDING_DIM}")

    index, creado = asegurar_indice(obtener_cliente())
    if creado:
        print(f"🆕 Índice '{config.INDEX_NAME}' creado (Serverless {config.PINECONE_CLOUD}/"
              f"{config.PINECONE_REGION}, métrica {config.METRICA}).")
    else:
        print(f"♻️  El índice '{config.INDEX_NAME}' ya existe y su dimensión coincide: no se vuelve a crear.")

    print("📊 Estado del índice:")
    print(index.describe_index_stats())


if __name__ == "__main__":
    main()
