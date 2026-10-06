import importlib

import dotenv
import pytest

import config


@pytest.fixture
def recargar_config(monkeypatch):
    """Recarga config.py con otras variables de entorno, ignorando el .env local."""
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)

    def _recargar(**variables):
        for nombre in ("EMBEDDING_PROVIDER", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
            monkeypatch.delenv(nombre, raising=False)
        for nombre, valor in variables.items():
            monkeypatch.setenv(nombre, valor)
        return importlib.reload(config)

    yield _recargar
    monkeypatch.undo()
    importlib.reload(config)


def test_google_es_el_proveedor_por_defecto(recargar_config):
    cfg = recargar_config()
    assert cfg.EMBEDDING_PROVIDER == "google"
    assert (cfg.EMBEDDING_MODEL, cfg.EMBEDDING_DIM) == ("gemini-embedding-001", 768)


@pytest.mark.parametrize(("proveedor", "dimension"), [("huggingface", 384), ("openai", 1536)])
def test_la_dimension_se_deriva_del_proveedor(recargar_config, proveedor, dimension):
    assert recargar_config(EMBEDDING_PROVIDER=proveedor).EMBEDDING_DIM == dimension


def test_proveedor_invalido(recargar_config):
    with pytest.raises(ValueError, match="no es válido"):
        recargar_config(EMBEDDING_PROVIDER="anthropic")


def test_google_sin_api_key_da_error_claro(recargar_config):
    cfg = recargar_config(EMBEDDING_PROVIDER="google")
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        cfg.obtener_embeddings()


def test_google_acepta_gemini_api_key(recargar_config):
    cfg = recargar_config(EMBEDDING_PROVIDER="google", GEMINI_API_KEY="clave-de-prueba")
    embeddings = cfg.obtener_embeddings()  # no hace llamadas de red al construirse
    assert embeddings.model.endswith("gemini-embedding-001")
    assert embeddings.output_dimensionality == 768
