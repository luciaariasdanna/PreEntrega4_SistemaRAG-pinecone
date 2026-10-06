"""Consulta interactiva del recuperador híbrido.

Uso:
    python consultar.py                                   # modo interactivo
    python consultar.py "¿Cómo configuro CORS?"           # una sola consulta
    python consultar.py "tokens JWT" --categoria seguridad --modo bm25
"""

import argparse

from rag_system import MODOS, RAGSystem


def mostrar(rag: RAGSystem, consulta: str, modo: str, categoria: str | None) -> None:
    resultados = rag.obtener_top_k(consulta, modo=modo, categoria=categoria)
    if not resultados:
        print("⚠️  No se recuperó ningún fragmento.\n")
        return
    print(f"\n📎 Top {len(resultados)} fragmentos ({modo}):")
    for i, r in enumerate(resultados, start=1):
        extracto = " ".join(r["contenido"].split())
        print(f"\n  {i}. [{r['fuente']} | pág. {r['pagina']} | {r['categoria']}] etiquetas={r['etiquetas']}")
        print(f"     {extracto[:220]}{'...' if len(extracto) > 220 else ''}")
    print("-" * 90)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("consulta", nargs="?")
    parser.add_argument("--modo", choices=MODOS, default="hibrido")
    parser.add_argument("--categoria", help="filtra por metadata (ej: seguridad, fundamentos)")
    args = parser.parse_args()

    rag = RAGSystem.desde_pinecone()
    if args.consulta:
        mostrar(rag, args.consulta, args.modo, args.categoria)
        return

    print("💬 Modo interactivo — RAG híbrido (BM25 + Pinecone). Escribí 'salir' para terminar.")
    print(f"   Categorías disponibles: {', '.join(rag.categorias)}\n")
    while True:
        consulta = input("🧑 Vos: ").strip()
        if consulta.lower() in ("", "salir", "exit", "quit"):
            print("👋 Listo, terminamos la sesión.")
            break
        mostrar(rag, consulta, args.modo, args.categoria)


if __name__ == "__main__":
    main()
