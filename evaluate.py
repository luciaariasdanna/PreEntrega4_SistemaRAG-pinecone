"""Paso 3: evaluación del recuperador con un Golden Set.

Métricas (por pregunta, luego promediadas):
    Recall@k     fracción de documentos relevantes que aparecen en el top-k.
                 Con un único documento esperado vale 0 o 1 ("¿está el correcto en el top-5?").
    Precision@k  fracción de los k chunks recuperados que provienen de un documento relevante.
    MRR@k        1 / posición del primer chunk relevante (0 si no aparece). Premia que el
                 documento correcto quede ARRIBA, cosa que Recall@k no distingue.

Nota sobre Precision@k: la unidad que se recupera es el chunk, pero el golden set marca
documentos. Si el documento relevante solo tiene 2 chunks, la Precision@5 máxima posible es
2/5 = 40%. Por eso el reporte muestra también ese techo ("P@k máx").

Uso:
    python evaluate.py                    # modo híbrido
    python evaluate.py --modo todos       # compara bm25 vs vectorial vs híbrido
    python evaluate.py --k 3 --salida resultados.json
"""

import argparse
import json
from pathlib import Path

import config
from rag_system import MODOS, RAGSystem


# --- Golden set -------------------------------------------------------------

def cargar_golden_set(ruta: Path = config.GOLDEN_SET_PATH) -> list[dict]:
    casos = json.loads(Path(ruta).read_text(encoding="utf-8"))
    for caso in casos:
        if "pregunta" not in caso or "documento_id_esperado" not in caso:
            raise ValueError(f"Caso inválido en el golden set: {caso}")
    return casos


def documentos_relevantes(caso: dict) -> set[str]:
    """Por defecto solo el documento esperado; `documentos_relevantes` permite marcar más de uno."""
    return set(caso.get("documentos_relevantes") or [caso["documento_id_esperado"]])


# --- Métricas ---------------------------------------------------------------

def recall_at_k(recuperados: list[str], relevantes: set[str], k: int) -> float:
    if not relevantes:
        return 0.0
    return len(relevantes & set(recuperados[:k])) / len(relevantes)


def precision_at_k(recuperados: list[str], relevantes: set[str], k: int) -> float:
    """Se divide por k (no por la cantidad recuperada): devolver menos de k resultados
    no debe inflar la precisión."""
    if k <= 0:
        return 0.0
    return sum(1 for fuente in recuperados[:k] if fuente in relevantes) / k


def mrr_at_k(recuperados: list[str], relevantes: set[str], k: int) -> float:
    for posicion, fuente in enumerate(recuperados[:k], start=1):
        if fuente in relevantes:
            return 1.0 / posicion
    return 0.0


def precision_maxima(chunks_por_fuente: dict[str, int], relevantes: set[str], k: int) -> float:
    """Techo de Precision@k: cuántos chunks relevantes existen en todo el corpus."""
    disponibles = sum(chunks_por_fuente.get(fuente, 0) for fuente in relevantes)
    return min(k, disponibles) / k if k > 0 else 0.0


# --- Evaluación -------------------------------------------------------------

def evaluar(rag: RAGSystem, golden_set: list[dict], modo: str = "hibrido") -> dict:
    k = rag.k
    chunks_por_fuente: dict[str, int] = {}
    for doc in rag.corpus:
        fuente = doc.metadata.get("source")
        chunks_por_fuente[fuente] = chunks_por_fuente.get(fuente, 0) + 1

    detalle = []
    for caso in golden_set:
        relevantes = documentos_relevantes(caso)
        recuperados = [r["fuente"] for r in rag.obtener_top_k(caso["pregunta"], modo=modo)]
        detalle.append({
            "pregunta": caso["pregunta"],
            "relevantes": sorted(relevantes),
            "recuperados": recuperados,
            "recall": recall_at_k(recuperados, relevantes, k),
            "precision": precision_at_k(recuperados, relevantes, k),
            "mrr": mrr_at_k(recuperados, relevantes, k),
            "precision_max": precision_maxima(chunks_por_fuente, relevantes, k),
        })

    n = len(detalle) or 1
    return {
        "modo": modo,
        "k": k,
        "detalle": detalle,
        **{f"{m}_promedio": sum(d[m] for d in detalle) / n
           for m in ("recall", "precision", "mrr", "precision_max")},
    }


# --- Reporte ----------------------------------------------------------------

def imprimir_detalle(reporte: dict) -> None:
    k = reporte["k"]
    print("=" * 90)
    print(f"🔍 Modo: {reporte['modo'].upper()}  (k={k})")
    print("=" * 90)
    for i, d in enumerate(reporte["detalle"], start=1):
        estado = "✅" if d["recall"] == 1.0 else ("🟡" if d["recall"] > 0 else "❌")
        marcados = ["*" + f if f in d["relevantes"] else f for f in d["recuperados"]]
        print(f"{estado} {i}. {d['pregunta']}")
        print(f"   Esperado:    {', '.join(d['relevantes'])}")
        print(f"   Recuperados: {marcados}   (* = relevante)")
        print(f"   Recall@{k}: {d['recall']:.0%} | Precision@{k}: {d['precision']:.0%} "
              f"(máx {d['precision_max']:.0%}) | MRR: {d['mrr']:.2f}\n")


def imprimir_resumen(reportes: list[dict]) -> None:
    k = reportes[0]["k"]
    print("=" * 90)
    print("📊 RESUMEN")
    print("=" * 90)
    encabezado = f"{'Modo':<12}{'Recall@' + str(k):>12}{'Precision@' + str(k):>15}{'P@' + str(k) + ' máx':>12}{'MRR@' + str(k):>10}"
    print(encabezado)
    print("-" * len(encabezado))
    for r in reportes:
        print(f"{r['modo']:<12}{r['recall_promedio']:>12.1%}{r['precision_promedio']:>15.1%}"
              f"{r['precision_max_promedio']:>12.1%}{r['mrr_promedio']:>10.3f}")
    print()
    for linea in conclusiones(reportes):
        print(linea)
    print()


def conclusiones(reportes: list[dict]) -> list[str]:
    """Resumen en lenguaje natural de la tabla."""
    k = reportes[0]["k"]
    lineas = []
    for r in reportes:
        n = len(r["detalle"])
        aciertos = sum(1 for d in r["detalle"] if d["recall"] == 1.0)
        primeros = sum(1 for d in r["detalle"] if d["mrr"] == 1.0)
        techo = r["precision_promedio"] / r["precision_max_promedio"] if r["precision_max_promedio"] else 0.0
        lineas.append(
            f"📝 {r['modo']}: encontró todos los documentos esperados en {aciertos}/{n} preguntas "
            f"(en 1ª posición en {primeros}/{n}); su Precision@{k} alcanza el {techo:.0%} del máximo posible."
        )
    if len(reportes) > 1:
        mejor = max(reportes, key=lambda r: (r["recall_promedio"], r["mrr_promedio"], r["precision_promedio"]))
        lineas.append(f"🏆 Mejor modo (Recall → MRR → Precision): {mejor['modo']}")
    return lineas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modo", choices=[*MODOS, "todos"], default="hibrido")
    parser.add_argument("--k", type=int, default=config.TOP_K)
    parser.add_argument("--golden", type=Path, default=config.GOLDEN_SET_PATH)
    parser.add_argument("--salida", type=Path, help="guarda el reporte completo en JSON")
    parser.add_argument("--resumen", action="store_true", help="omite el detalle por pregunta")
    args = parser.parse_args()

    golden_set = cargar_golden_set(args.golden)
    print(f"🧪 Golden set: {len(golden_set)} preguntas ({args.golden.name})")
    rag = RAGSystem.desde_pinecone(k=args.k)
    print(f"📚 Corpus cargado desde Pinecone: {len(rag.corpus)} chunks\n")

    modos = list(MODOS) if args.modo == "todos" else [args.modo]
    reportes = [evaluar(rag, golden_set, modo) for modo in modos]
    if not args.resumen:
        for reporte in reportes:
            imprimir_detalle(reporte)
    imprimir_resumen(reportes)

    if args.salida:
        args.salida.write_text(json.dumps(reportes, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"💾 Reporte guardado en {args.salida}")


if __name__ == "__main__":
    main()
