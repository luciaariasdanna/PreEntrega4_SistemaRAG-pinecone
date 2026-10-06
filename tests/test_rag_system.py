import pytest

from rag_system import RAGSystem, tokenizar


def test_tokenizar_normaliza_tildes_mayusculas_y_puntuacion():
    assert tokenizar("¿Qué es FastAPI?") == ["fastapi"]
    assert tokenizar("Autenticación") == ["autenticacion"]


def test_tokenizar_conserva_identificadores_tecnicos():
    assert tokenizar("allow_origins de CORSMiddleware (422)") == ["allow_origins", "corsmiddleware", "422"]


def test_bm25_encuentra_termino_tecnico_exacto(retriever_fijo, corpus):
    rag = RAGSystem(retriever_fijo, corpus, k=3)
    [primero, *_] = rag.buscar("OAuth2PasswordBearer", modo="bm25")
    assert primero.metadata["source"] == "autenticacion_oauth2_jwt.md"


def test_bm25_no_devuelve_documentos_sin_terminos_en_comun(retriever_fijo, corpus):
    rag = RAGSystem(retriever_fijo, corpus, k=5)
    fuentes = [d.metadata["source"] for d in rag.buscar("OAuth2PasswordBearer", modo="bm25")]
    assert fuentes == ["autenticacion_oauth2_jwt.md"]
    assert rag.buscar("palabras inexistentes xyz", modo="bm25") == []


def test_vectorial_devuelve_el_orden_del_retriever(retriever_fijo, corpus):
    rag = RAGSystem(retriever_fijo, corpus, k=2)
    fuentes = [d.metadata["source"] for d in rag.buscar("cualquier cosa", modo="vectorial")]
    assert fuentes == ["testing_con_testclient.md", "tareas_en_segundo_plano.md"]


def test_hibrido_combina_ambos_retrievers_sin_duplicados(retriever_fijo, corpus):
    rag = RAGSystem(retriever_fijo, corpus, k=5)
    docs = rag.buscar("OAuth2PasswordBearer tokenUrl", modo="hibrido")
    ids = [d.metadata["chunk_id"] for d in docs]

    assert len(docs) <= 5
    assert len(ids) == len(set(ids))
    assert "oauth-c0" in ids  # aporte léxico (no está en el retriever vectorial)
    assert "test-c0" in ids   # aporte semántico


def test_hibrido_rrf_premia_documentos_que_aparecen_en_ambos(retriever_fijo, corpus):
    # "cors-c0" está en ambos rankings: debe quedar por encima de los que aparecen en uno solo.
    rag = RAGSystem(retriever_fijo, corpus, k=5)
    [primero, *_] = rag.buscar("CORSMiddleware allow_origins", modo="hibrido")
    assert primero.metadata["chunk_id"] == "cors-c0"


def test_filtro_por_categoria_aplica_a_ambos_retrievers(retriever_fijo, corpus):
    rag = RAGSystem(retriever_fijo, corpus, k=5)
    docs = rag.buscar("tareas dependencias cors token", modo="hibrido", categoria="avanzado")
    assert docs
    assert {d.metadata["categoria"] for d in docs} == {"avanzado"}


def test_obtener_top_k_formato_plano(retriever_fijo, corpus):
    rag = RAGSystem(retriever_fijo, corpus, k=1)
    [resultado] = rag.obtener_top_k("TestClient", modo="vectorial")
    assert resultado["fuente"] == "testing_con_testclient.md"
    assert set(resultado) == {"contenido", "fuente", "pagina", "categoria", "etiquetas", "chunk_id"}


def test_categorias_disponibles(retriever_fijo, corpus):
    rag = RAGSystem(retriever_fijo, corpus)
    assert rag.categorias == ["avanzado", "fundamentos", "operaciones", "seguridad"]


def test_errores_claros(retriever_fijo, corpus):
    with pytest.raises(ValueError, match="vacío"):
        RAGSystem(retriever_fijo, [])
    rag = RAGSystem(retriever_fijo, corpus)
    with pytest.raises(ValueError, match="Modo"):
        rag.buscar("x", modo="inexistente")
    with pytest.raises(ValueError, match="categoria"):
        rag.buscar("x", categoria="inexistente")
