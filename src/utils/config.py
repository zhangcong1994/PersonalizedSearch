import os
import yaml
from pathlib import Path
from typing import Any


def _find_project_root() -> Path:
    for start in (Path(__file__).resolve().parent, Path.cwd()):
        for parent in [start, *start.parents]:
            if (parent / "config.yaml").is_file():
                return parent
    return Path.cwd()


_project_root = _find_project_root()


def _load_raw_config() -> dict[str, Any]:
    config_path = _project_root / "config.yaml"
    if config_path.is_file():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _resolve_path(raw: str, base: Path = _project_root) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (base / path).resolve()


_data_root_raw = os.environ.get("PERSONALIZEDSEARCH_DATA_ROOT")
if _data_root_raw:
    DATA_ROOT = Path(_data_root_raw).resolve()
else:
    DATA_ROOT = _project_root


_config = _load_raw_config()
_paths_config = _config.get("paths", {})
_models_config = _config.get("models", {})


MODEL_CACHE_DIR = (
    _resolve_path(_models_config.get("local_cache_dir", "models"), DATA_ROOT)
    if _models_config.get("local_cache_dir")
    else _resolve_path("models", DATA_ROOT)
)


DATA_DIR = _resolve_path(_paths_config.get("data_dir", "data"), DATA_ROOT)
VECTOR_DB_DIR = _resolve_path(_paths_config.get("vector_db_dir", "data/vector_db"), DATA_ROOT)
RAW_DATA_DIR = _resolve_path(_paths_config.get("raw_data_dir", "data/raw"), DATA_ROOT)

PROJECT_ROOT = _project_root

EMBEDDING_MODEL = _config.get("indexing", {}).get("embedding_model", "BAAI/bge-small-zh-v1.5")

EMBEDDING_MODEL_REGISTRY = _models_config.get("registry", {})
EMBEDDING_MODEL_CHOICES = [v["hf_id"] for v in EMBEDDING_MODEL_REGISTRY.values() if "hf_id" in v] or [EMBEDDING_MODEL]


def model_short_name(model_id: str) -> str:
    return model_id.split("/")[-1]


def _resolve_hf_snapshot(cache_dir: Path) -> Path | None:
    snapshots_dir = cache_dir / "snapshots"
    if snapshots_dir.is_dir():
        snapshots = sorted(snapshots_dir.iterdir())
        if snapshots:
            return snapshots[-1]
    return None


def resolve_model_local_path(model_id: str) -> Path | None:
    for entry in EMBEDDING_MODEL_REGISTRY.values():
        if entry.get("hf_id") == model_id:
            local_dir = entry.get("local_dir")
            if local_dir:
                p = MODEL_CACHE_DIR / local_dir
                if p.is_dir():
                    resolved = _resolve_hf_snapshot(p)
                    return resolved if resolved is not None else p
            break

    short = model_id.split("/")[-1]
    for candidate in [
        MODEL_CACHE_DIR / short,
        MODEL_CACHE_DIR / f"models--{model_id.replace('/', '--')}",
    ]:
        if candidate.is_dir():
            resolved = _resolve_hf_snapshot(candidate)
            return resolved if resolved is not None else candidate
    return None

CHUNK_SIZE = _config.get("indexing", {}).get("chunk_size", 500)
CHUNK_OVERLAP = _config.get("indexing", {}).get("chunk_overlap", 50)

TOP_K = _config.get("retrieval", {}).get("top_k", 5)
SEARCH_TYPE = _config.get("retrieval", {}).get("search_type", "similarity")

LLM_CONFIG = _config.get("generation", {}).get("api_client", {})
LLM_MODEL = LLM_CONFIG.get("model", "deepseek-chat")
LLM_BASE_URL = LLM_CONFIG.get("base_url", "https://api.deepseek.com/v1")
LLM_TEMPERATURE = float(LLM_CONFIG.get("temperature", 0.3))
LLM_MAX_TOKENS = int(LLM_CONFIG.get("max_tokens", 1024))

LOG_LEVEL = _config.get("logging", {}).get("level", "INFO")
LOG_FORMAT = _config.get("logging", {}).get("format", "%(asctime)s - %(levelname)s - %(message)s")


def print_config():
    print(f"PROJECT_ROOT    = {PROJECT_ROOT}")
    print(f"DATA_ROOT       = {DATA_ROOT}")
    print(f"DATA_DIR        = {DATA_DIR}")
    print(f"VECTOR_DB_DIR   = {VECTOR_DB_DIR}")
    print(f"RAW_DATA_DIR    = {RAW_DATA_DIR}")
    print(f"MODEL_CACHE_DIR = {MODEL_CACHE_DIR}")
    print(f"EMBEDDING_MODEL = {EMBEDDING_MODEL}")
    print(f"LLM_MODEL       = {LLM_MODEL}")
