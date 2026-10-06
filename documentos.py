"""Carga y chunking de documentos (Markdown, JSON y PDF) con metadatos enriquecidos.

Esquema de metadatos de cada chunk (todos tipos válidos para Pinecone):
    source       nombre del archivo de origen; es el "documento_id" del golden set
    page         página del PDF / número de entrada del JSON / 1 para Markdown
    categoria    subcarpeta de data/ (fundamentos, seguridad, ...)
    etiquetas    lista de strings (front matter del .md o campo "etiquetas" del .json)
    titulo       título legible del documento
    tipo         markdown | json | pdf
    chunk_index  posición del chunk dentro de (source, page)
    chunk_id     ID determinista: re-ingestar sobreescribe en vez de duplicar
"""

import json
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

EXTENSIONES_SOPORTADAS = {".md", ".json", ".pdf"}

# Primero se intenta cortar por encabezados de Markdown, después por párrafos, etc.
SEPARADORES = ["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]


# --- Parsers ----------------------------------------------------------------

def parsear_front_matter(texto: str) -> tuple[dict, str]:
    """Separa un bloque `---` inicial con pares `clave: valor` del cuerpo del Markdown.

    Las listas se escriben como `[a, b, c]`. Es un parser mínimo a propósito: no
    necesitamos YAML completo y así evitamos una dependencia.
    """
    coincidencia = re.match(r"^---\s*\n(.*?)\n---\s*\n?", texto, flags=re.DOTALL)
    if not coincidencia:
        return {}, texto

    meta = {}
    for linea in coincidencia.group(1).splitlines():
        if ":" not in linea:
            continue
        clave, valor = (parte.strip() for parte in linea.split(":", 1))
        if valor.startswith("[") and valor.endswith("]"):
            meta[clave] = [v.strip() for v in valor[1:-1].split(",") if v.strip()]
        else:
            meta[clave] = valor
    return meta, texto[coincidencia.end():]


def _metadata_base(ruta: Path, categoria: str, tipo: str, titulo: str, etiquetas) -> dict:
    return {
        "source": ruta.name,
        "categoria": categoria,
        "tipo": tipo,
        "titulo": titulo,
        "etiquetas": [str(e) for e in (etiquetas or [])],
    }


def cargar_markdown(ruta: Path, categoria: str) -> list[Document]:
    meta, cuerpo = parsear_front_matter(ruta.read_text(encoding="utf-8"))
    titulo = meta.get("titulo") or ruta.stem.replace("_", " ")
    metadata = _metadata_base(ruta, categoria, "markdown", titulo, meta.get("etiquetas"))
    metadata["page"] = 1
    return [Document(page_content=cuerpo.strip(), metadata=metadata)]


def cargar_json(ruta: Path, categoria: str) -> list[Document]:
    """Un JSON con `entradas: [{pregunta, respuesta, etiquetas}]`: cada entrada es
    una "página" lógica, así la cita apunta a la entrada exacta."""
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    titulo = datos.get("titulo", ruta.stem)
    documentos = []
    for numero, entrada in enumerate(datos["entradas"], start=1):
        etiquetas = list(datos.get("etiquetas", [])) + list(entrada.get("etiquetas", []))
        metadata = _metadata_base(ruta, categoria, "json", titulo, etiquetas)
        metadata["page"] = numero
        contenido = f"Pregunta: {entrada['pregunta']}\nRespuesta: {entrada['respuesta']}"
        documentos.append(Document(page_content=contenido, metadata=metadata))
    return documentos


def cargar_pdf(ruta: Path, categoria: str) -> list[Document]:
    from pypdf import PdfReader

    lector = PdfReader(str(ruta))
    titulo = (lector.metadata.title if lector.metadata else None) or ruta.stem.replace("_", " ")
    palabras_clave = (lector.metadata.get("/Keywords") if lector.metadata else None) or ""
    etiquetas = [e.strip() for e in palabras_clave.split(",") if e.strip()]

    documentos = []
    for numero, pagina in enumerate(lector.pages, start=1):
        texto = (pagina.extract_text() or "").strip()
        if not texto:
            continue
        metadata = _metadata_base(ruta, categoria, "pdf", titulo, etiquetas)
        metadata["page"] = numero
        documentos.append(Document(page_content=texto, metadata=metadata))
    return documentos


CARGADORES = {".md": cargar_markdown, ".json": cargar_json, ".pdf": cargar_pdf}


def cargar_documentos(data_dir: Path = config.DATA_DIR) -> list[Document]:
    """Recorre data/<categoria>/<archivo> y carga cada archivo soportado."""
    data_dir = Path(data_dir)
    documentos = []
    for ruta in sorted(data_dir.rglob("*")):
        if ruta.suffix.lower() not in EXTENSIONES_SOPORTADAS or not ruta.is_file():
            continue
        relativa = ruta.relative_to(data_dir)
        categoria = relativa.parts[0] if len(relativa.parts) > 1 else "general"
        documentos.extend(CARGADORES[ruta.suffix.lower()](ruta, categoria))
    return documentos


# --- Chunking ---------------------------------------------------------------

def crear_splitter(chunk_size: int = config.CHUNK_SIZE,
                   chunk_overlap: int = config.CHUNK_OVERLAP) -> RecursiveCharacterTextSplitter:
    # from_tiktoken_encoder mide el tamaño en TOKENS (no caracteres), que es lo que
    # importa para el modelo de embeddings.
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARADORES,
    )


def generar_chunk_id(source: str, page: int, chunk_index: int) -> str:
    """ID ASCII y determinista (Pinecone exige IDs ASCII)."""
    base = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(source).stem).strip("-").lower()
    return f"{base}-p{page}-c{chunk_index}"


def dividir_en_chunks(documentos: list[Document],
                      chunk_size: int = config.CHUNK_SIZE,
                      chunk_overlap: int = config.CHUNK_OVERLAP) -> list[Document]:
    splitter = crear_splitter(chunk_size, chunk_overlap)
    chunks = []
    for documento in documentos:
        for indice, chunk in enumerate(splitter.split_documents([documento])):
            chunk_id = generar_chunk_id(chunk.metadata["source"], chunk.metadata["page"], indice)
            chunk.metadata["chunk_index"] = indice
            chunk.metadata["chunk_id"] = chunk_id
            chunk.id = chunk_id
            chunks.append(chunk)
    return chunks
