# Sistema RAG escalable con Pinecone — Recuperador Híbrido (BM25 + Vectorial)

PreEntrega 4 — AI Engineering (Coderhouse).

Módulo de recuperación que ejecuta el flujo completo de un RAG en la nube:

1. **Ingesta** de documentos técnicos (Markdown, JSON y PDF) a un índice **Pinecone Serverless**, con el texto y metadatos avanzados (fuente, página, categoría, etiquetas) dentro de cada vector.
2. **Recuperador híbrido**: `RAGSystem` encapsula un `EnsembleRetriever` de LangChain que fusiona **BM25** (léxico) y **Pinecone** (semántico) con Reciprocal Rank Fusion.
3. **Evaluación** con un Golden Set: `evaluate.py` calcula **Recall@5**, **Precision@5** y MRR@5, comparando los modos BM25, vectorial e híbrido.

El corpus es documentación de **FastAPI** escrita en castellano para este trabajo (10 archivos en 5 categorías).

---

## Estructura

```
├── config.py              # constantes y .env (modelo, dimensión, índice, namespace, chunking)
├── documentos.py          # loaders MD/JSON/PDF + chunking con RecursiveCharacterTextSplitter
├── pinecone_store.py      # SDK nativo de Pinecone: crear índice, upsert, corpus, PineconeRetriever
├── rag_system.py          # RAGSystem: BM25 + Pinecone -> EnsembleRetriever
├── setup_index.py         # paso 1: crea el índice Serverless si no existe
├── ingest.py              # paso 2: pipeline de ingesta
├── evaluate.py            # paso 3: Precision@k, Recall@k, MRR@k
├── consultar.py           # consultas por consola (interactivo o one-shot)
├── golden_set.json        # 10 preguntas con su documento fuente esperado
├── data/
│   ├── fundamentos/       # instalación, parámetros, Pydantic (.md)
│   ├── avanzado/          # dependencias, background tasks, middleware/CORS (.md)
│   ├── seguridad/         # OAuth2 + JWT (.md)
│   ├── operaciones/       # testing (.md) + despliegue (.pdf, 3 páginas)
│   └── faq/               # preguntas frecuentes (.json, 6 entradas)
├── scripts/generar_pdf_despliegue.py
└── tests/                 # pytest, sin red ni API keys
```

---

## Cómo replicar el índice

### 1. Requisitos

- Python 3.14 (probado con 3.14.7). También debería funcionar con 3.12 o 3.13.
- Una cuenta gratuita de [Pinecone](https://app.pinecone.io) con su API key.
- Una API key gratuita de Google AI Studio: [aistudio.google.com/apikey](https://aistudio.google.com/apikey). Se usa para los embeddings.

### 2. Instalación

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt               # solo para correr el sistema
pip install -r requirements-dev.txt           # + pytest y fpdf2
pip install -r requirements-huggingface.txt   # opcional: embeddings locales (instala torch)
```

Todas las dependencias tienen la versión fijada (`==`) para que la instalación sea reproducible.

### 3. Variables de entorno

```bash
cp .env.example .env
```

| Variable | Obligatoria | Descripción |
|---|---|---|
| `PINECONE_API_KEY` | sí | API key de Pinecone |
| `INDEX_NAME` | no (`fastapi-docs-rag`) | nombre del índice Serverless |
| `EMBEDDING_PROVIDER` | no (`google`) | `google` (768 dims), `huggingface` (local, 384 dims) u `openai` (1536 dims) |
| `GOOGLE_API_KEY` | con `google` | API key de Google AI Studio (también se acepta `GEMINI_API_KEY`) |
| `OPENAI_API_KEY` | solo con `openai` | API key de OpenAI |
| `PINECONE_NAMESPACE` | no (`fastapi-docs`) | namespace donde se guardan los vectores |
| `PINECONE_CLOUD` / `PINECONE_REGION` | no (`aws` / `us-east-1`) | ubicación del índice Serverless |
| `PESO_BM25` | no (`0.5`) | peso del BM25 en el ensemble (el vectorial recibe `1 - PESO_BM25`) |

### 4. Crear el índice

```bash
python setup_index.py
```

Verifica si el índice existe. Si no existe, lo crea en modo **Serverless** (`aws/us-east-1`, métrica coseno) con la dimensión del modelo de embeddings configurado. Si existe, **valida que la dimensión coincida** y, si no coincide, corta con un error explicativo en vez de fallar más tarde en el upsert.

### 5. Ingestar los documentos

```bash
python ingest.py --dry-run    # opcional: muestra los chunks sin subir nada (no necesita keys)
python ingest.py              # carga, chunkea, genera embeddings y sube a Pinecone
python ingest.py --limpiar    # vacía el namespace antes de subir
```

Con `google` los embeddings se calculan en la API de Gemini (el tier gratuito alcanza de sobra para los 25 chunks). Con `huggingface`, la primera vez se descarga el modelo (~470 MB).

> Si cambian de proveedor de embeddings, cambia la dimensión: usen otro `INDEX_NAME` o borren el índice anterior. `setup_index.py` lo detecta y avisa.

### 6. Consultar y evaluar

```bash
python consultar.py "¿Cómo configuro CORS con credenciales?"
python consultar.py "token JWT" --categoria seguridad
python consultar.py                       # modo interactivo
python evaluate.py --modo todos           # compara bm25 / vectorial / híbrido
```

### 7. Tests

```bash
pytest
```

Los tests no usan red: Pinecone y los embeddings se reemplazan por dobles en memoria (`tests/conftest.py`). Cubren los loaders, el chunking (tamaño en tokens, IDs únicos y metadata válida para Pinecone), la creación del índice y la validación de dimensión, el ida y vuelta upsert → corpus, el recuperador híbrido (fusión, deduplicación, filtros) y las métricas.

---

## Decisiones de diseño

### Esquema de cada vector en Pinecone

```json
{
  "id": "autenticacion_oauth2_jwt-p1-c0",
  "values": [0.013, -0.072, "... 768 floats ..."],
  "metadata": {
    "text": "# Autenticación con OAuth2 y JWT ...",
    "source": "autenticacion_oauth2_jwt.md",
    "page": 1,
    "categoria": "seguridad",
    "etiquetas": ["oauth2", "jwt", "oauth2passwordbearer", "autenticacion", "hash", "bearer"],
    "titulo": "Autenticación con OAuth2, contraseñas hasheadas y tokens JWT",
    "tipo": "markdown",
    "chunk_index": 0,
    "chunk_id": "autenticacion_oauth2_jwt-p1-c0"
  }
}
```

- **El texto vive en `metadata.text`**: una sola consulta a Pinecone devuelve vector, texto y fuente, sin una base relacional aparte.
- **`page`** es la página real en el PDF, el número de entrada en el JSON y `1` en Markdown.
- **`categoria`** sale de la subcarpeta de `data/` y permite filtrar del lado de Pinecone (`--categoria seguridad` envía `filter={"categoria": {"$eq": "seguridad"}}`).
- **IDs deterministas** (`<archivo>-p<página>-c<chunk>`): volver a correr `ingest.py` sobreescribe los vectores en vez de duplicarlos. En la guía de clase el conteo de vectores crecía en cada ejecución (16 → 20) porque `from_documents` genera UUIDs aleatorios.

### Namespace

Todos los vectores van al namespace `fastapi-docs`. Si mañana se agrega otra librería u otro inquilino, se usa otro namespace en el mismo índice y las búsquedas no se mezclan.

### Chunking

`RecursiveCharacterTextSplitter.from_tiktoken_encoder` con **600 tokens y 100 de overlap**, dentro del rango de 500 a 800 sugerido. El tamaño se mide en tokens, no en caracteres. Los separadores priorizan los encabezados de Markdown (`\n## `, `\n### `) y luego los párrafos, para no cortar una sección por la mitad.

### Embeddings y dimensión

La dimensión **no está hardcodeada**: `config.py` la deriva del proveedor elegido y `setup_index.py` la valida contra el índice existente.

| Proveedor | Modelo | Dimensión |
|---|---|---|
| `google` (por defecto) | `gemini-embedding-001` | 768 |
| `huggingface` | `paraphrase-multilingual-MiniLM-L12-v2` | 384 |
| `openai` | `text-embedding-3-small` | 1536 |

Elegimos **Google** porque su API key es gratuita, no requiere instalar torch y el modelo es multilingüe, algo clave porque el corpus y las preguntas están en castellano. `gemini-embedding-001` genera 3072 dimensiones, pero está entrenado con *Matryoshka Representation Learning*, así que se puede truncar a 768 (`output_dimensionality`) con una pérdida mínima de calidad y 4 veces menos almacenamiento en Pinecone. Además, la librería usa `task_type=RETRIEVAL_DOCUMENT` al indexar y `RETRIEVAL_QUERY` al consultar, que es la configuración que recomienda Google para búsqueda.

### SDK nativo de Pinecone en vez de `langchain-pinecone`

`langchain-pinecone` todavía exige Python < 3.14 y `numpy < 2`, así que no instala en Python 3.14. La consigna permite usar el SDK nativo, y `pinecone_store.PineconeRetriever` es un `BaseRetriever` de LangChain de unas 15 líneas sobre `index.query(...)`, con namespace y filtro de metadata. Se integra con el `EnsembleRetriever` igual que el retriever oficial.

### Recuperador híbrido

- **BM25 con preprocesamiento para castellano**: el `BM25Retriever` por defecto hace `texto.split()`, así que `"¿Qué"` y `"qué"`, o `"FastAPI?"` y `"FastAPI"`, serían tokens distintos. `tokenizar()` pasa a minúsculas, quita tildes y puntuación, conserva `_` (para `allow_origins` o `status_code`) y descarta stopwords.
- **BM25 sin resultados de puntaje 0**: el retriever original siempre devuelve *k* documentos aunque no compartan ningún término con la consulta, y en la fusión RRF esos documentos suman puntos solo por su posición. `BM25SinCeros` los descarta.
- **Corpus del BM25 desde Pinecone**: `RAGSystem.desde_pinecone()` descarga los chunks del namespace (`index.list` + `index.fetch`) y arma el BM25 con exactamente el mismo texto que está indexado. No hace falta tener `data/` en la máquina que consulta.
- **Fusión**: `EnsembleRetriever` con pesos `[0.5, 0.5]` y `id_key="chunk_id"` para deduplicar por ID. Cada retriever aporta 10 candidatos y se devuelve el top-5.

### Métricas

| Métrica | Definición |
|---|---|
| **Recall@5** | fracción de los documentos relevantes que aparecen en el top-5. Con un solo documento esperado vale 0 o 1. |
| **Precision@5** | fracción de los 5 chunks recuperados que provienen de un documento relevante (se divide siempre por 5). |
| **MRR@5** | 1 / posición del primer chunk relevante. Premia que el documento correcto aparezca arriba. |
| **P@5 máx** | techo de Precision@5: si el documento relevante tiene solo 2 chunks, la precisión no puede superar 40%. |

El golden set tiene 10 preguntas: 5 con términos técnicos exactos (`OAuth2PasswordBearer`, `allow_origins`, `422`...), 4 parafraseadas sin las palabras del documento ("mandar un mail sin que el usuario espere" → `BackgroundTasks`) y 1 con dos documentos relevantes (`documentos_relevantes`).

---

## Resultados de la evaluación

### Pinecone + Google `gemini-embedding-001` (configuración final)

Salida real de `python evaluate.py --modo todos`: índice Serverless `fastapi-docs-rag` (768 dims, coseno), 25 chunks, 10 preguntas, k=5 y `PESO_BM25=0.5`.

```
Modo            Recall@5    Precision@5     P@5 máx     MRR@5
-------------------------------------------------------------
hibrido           100.0%          42.0%       48.0%     1.000
bm25              100.0%          36.0%       48.0%     1.000
vectorial         100.0%          46.0%       48.0%     1.000
```

**Recall@5 = 100% y MRR@5 = 1.0 en los tres modos**: en las 10 preguntas el documento correcto aparece en el top-5, y además siempre en la **primera posición**.

### Comparación con la validación local (HuggingFace)

Antes de pasar a Google validamos el golden set sin API keys: Pinecone se reemplazó por un `InMemoryVectorStore` (misma métrica coseno) con embeddings `paraphrase-multilingual-MiniLM-L12-v2`, de 384 dimensiones.

| Modo | Recall@5 HF → Google | Precision@5 HF → Google | MRR@5 HF → Google |
|---|---|---|---|
| híbrido | 100% → 100% | 34% → **42%** | 0.92 → **1.00** |
| bm25 | 100% → 100% | 36% → 36% | 1.00 → 1.00 |
| vectorial | 90% → **100%** | 26% → **46%** | 0.77 → **1.00** |

(BM25 no depende de los embeddings, por eso no cambia.)

### Análisis

- **El cambio de embeddings fue lo que más impactó.** Con el modelo chico, el vectorial fallaba la pregunta 1 ("levantar el servidor para que se reinicie solo" → `fastapi dev` / `--reload`) y ponía el documento de testing en 5º lugar para "¿cómo pruebo mis endpoints sin levantar el servidor?". Gemini resuelve ambos casos en la posición 1.
- **Precision@5 está acotada por el chunking**: casi todos los documentos tienen 2 chunks, así que el máximo posible por pregunta suele ser 40% (48% en promedio). El vectorial llega a 46/48 y el híbrido a 42/48 (88% de lo alcanzable).
- **¿Por qué el híbrido no le gana al vectorial en este benchmark?** El BM25 agrega al top-5 chunks que comparten palabras con la pregunta pero son de otro documento (por ejemplo, "servidor" aparece en el PDF de despliegue). Con un modelo de embeddings fuerte, ese aporte léxico desplaza algún chunk relevante. Igual mantenemos el híbrido como configuración final, por tres motivos:
  1. **Robustez**: con el modelo chico, el híbrido salvó el Recall que el vectorial perdía (90% → 100%). Si el modelo falla con una sigla, un nombre propio o un identificador que no vio en entrenamiento, BM25 lo cubre.
  2. **Ranking perfecto**: el híbrido mantiene MRR = 1.0 en las 10 preguntas.
  3. **Sesgo del benchmark**: 10 preguntas escritas por nosotros mirando los documentos no alcanzan para afirmar que el vectorial puro es mejor. La diferencia (46% contra 42%) son 2 chunks en total.
- **Pesos del ensemble**: con Pinecone + Google probamos `PESO_BM25` en 0.3, 0.4 y 0.5. Con 0.3 y 0.4 la Precision@5 del híbrido sube a 44% (un solo chunk más en todo el benchmark), y Recall y MRR no cambian. Dejamos 0.5 por defecto para no sobreajustar a 10 preguntas. Se puede cambiar con `PESO_BM25` en el `.env`.
- **Mejoras posibles**: un re-ranker (cross-encoder) sobre el top-10 fusionado para recuperar la precisión del vectorial sin perder la red de seguridad léxica; chunks más chicos o por sección para subir el techo de Precision@5; y un golden set más grande escrito por terceros.
