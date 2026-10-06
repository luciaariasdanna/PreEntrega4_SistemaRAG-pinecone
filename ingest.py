"""Paso 2: pipeline de ingesta. Carga data/, divide en chunks, genera embeddings y los
sube a Pinecone con el texto y los metadatos (fuente, página, categoría, etiquetas).

Uso:
    python ingest.py              # ingesta incremental (upsert por ID determinista)
    python ingest.py --limpiar    # vacía el namespace antes de subir (borra chunks huérfanos)
    python ingest.py --dry-run    # solo carga y chunkea; no necesita API keys
"""

import argparse
from collections import Counter

import config
from documentos import cargar_documentos, dividir_en_chunks
from pinecone_store import asegurar_indice, obtener_cliente, subir_chunks, vaciar_namespace


def resumir_chunks(chunks) -> None:
    print(f"✂️  Chunks generados: {len(chunks)} (chunk_size={config.CHUNK_SIZE} tokens, "
          f"overlap={config.CHUNK_OVERLAP})")
    por_fuente = Counter(c.metadata["source"] for c in chunks)
    for fuente, cantidad in sorted(por_fuente.items()):
        print(f"   - {fuente}: {cantidad} chunk(s)")
    print(f"🔎 Metadata de ejemplo: {chunks[0].metadata}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limpiar", action="store_true", help="vacía el namespace antes de subir")
    parser.add_argument("--dry-run", action="store_true", help="no sube nada a Pinecone")
    args = parser.parse_args()

    documentos = cargar_documentos(config.DATA_DIR)
    print(f"📄 Documentos/páginas cargados: {len(documentos)} desde {config.DATA_DIR}")
    chunks = dividir_en_chunks(documentos)
    resumir_chunks(chunks)

    if args.dry_run:
        print("🧪 --dry-run: no se sube nada a Pinecone.")
        return

    index, creado = asegurar_indice(obtener_cliente())
    print(f"{'🆕 Índice creado' if creado else '♻️  Índice existente'}: '{config.INDEX_NAME}'")

    if args.limpiar and vaciar_namespace(index, config.NAMESPACE):
        print(f"🧹 Namespace '{config.NAMESPACE}' vaciado.")

    print(f"🧠 Generando embeddings con {config.EMBEDDING_MODEL}...")
    total = subir_chunks(index, chunks, config.obtener_embeddings(), config.NAMESPACE)
    print(f"📦 {total} vectores subidos al namespace '{config.NAMESPACE}'.")
    # Las estadísticas de un índice Serverless pueden tardar unos segundos en reflejar el upsert.
    print("📊 Estado del índice:", index.describe_index_stats())


if __name__ == "__main__":
    main()
