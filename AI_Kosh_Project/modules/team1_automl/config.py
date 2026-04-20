import os
import warnings
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"


def _load_env_files() -> None:
    """Load .env so values win over empty OS env vars (override=True). Project .env first, then cwd .env if different."""
    loaded: list[Path] = []
    if ENV_PATH.is_file():
        load_dotenv(ENV_PATH, override=True)
        loaded.append(ENV_PATH.resolve())
    cwd_env = Path.cwd() / ".env"
    if cwd_env.is_file() and cwd_env.resolve() not in loaded:
        load_dotenv(cwd_env, override=True)


_load_env_files()


def _resolve_azure_storage_connection_string() -> str:
    """Prefer AZURE_STORAGE_CONNECTION_STRING; else build from account name + key (Azure portal style)."""
    direct = (os.getenv("AZURE_STORAGE_CONNECTION_STRING") or "").strip()
    if direct:
        return direct
    name = (os.getenv("AZURE_STORAGE_ACCOUNT_NAME") or "").strip()
    key = (os.getenv("AZURE_STORAGE_ACCOUNT_KEY") or "").strip()
    if not name or not key:
        return ""
    suffix = (os.getenv("AZURE_STORAGE_ENDPOINT_SUFFIX") or "core.windows.net").strip()
    return (
        "DefaultEndpointsProtocol=https;"
        f"AccountName={name};AccountKey={key};EndpointSuffix={suffix}"
    )


AZURE_STORAGE_CONNECTION_STRING = _resolve_azure_storage_connection_string()
STORAGE_MODE = "azure" if AZURE_STORAGE_CONNECTION_STRING else "local"
AI_MODE = "azure" if os.getenv("AZURE_OPENAI_API_KEY") else "huggingface"

_n = (os.getenv("AZURE_STORAGE_ACCOUNT_NAME") or "").strip()
_k = (os.getenv("AZURE_STORAGE_ACCOUNT_KEY") or "").strip()
if (_n and not _k) or (_k and not _n):
    warnings.warn(
        "Azure Blob: set BOTH AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY "
        "(or use AZURE_STORAGE_CONNECTION_STRING alone). Otherwise STORAGE_MODE stays local.",
        stacklevel=1,
    )

if (os.getenv("AZURE_STORAGE_REQUIRE_AZURE") or "").strip().lower() in ("1", "true", "yes"):
    if not AZURE_STORAGE_CONNECTION_STRING:
        raise RuntimeError(
            "AZURE_STORAGE_REQUIRE_AZURE is set but Azure storage credentials are missing. "
            f"Add AZURE_STORAGE_ACCOUNT_NAME + AZURE_STORAGE_ACCOUNT_KEY (or AZURE_STORAGE_CONNECTION_STRING) "
            f"to {ENV_PATH} or the environment."
        )

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
# Single container for all Team1 AutoML blobs (virtual folders are path prefixes inside it).
AZURE_BLOB_CONTAINER_NAME = (
    os.getenv("AZURE_BLOB_CONTAINER_NAME")
    or os.getenv("AZURE_BLOB_CONTAINER")
    or os.getenv("AZURE_STORAGE_CONTAINER_NAME")
    or "aikosh-v2"
)
# Root "folder" inside the container: datasets, models, results live under {prefix}/datasets/..., etc.
AZURE_STORAGE_AUTOML_PREFIX = os.getenv("AZURE_STORAGE_AUTOML_PREFIX", "automl").strip().strip("/")

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

SHARED_WORKSPACE = PROJECT_ROOT / "shared_workspace"
RAW_UPLOADS_DIR = SHARED_WORKSPACE / "1_raw_uploads"
PROCESSED_DATA_DIR = SHARED_WORKSPACE / "2_processed_data"
MODELS_DIR = SHARED_WORKSPACE / "3_models"
TEAM_DB_PATH = BASE_DIR / "team1_automl.db"

# If set (YYYY-MM-DD), delete training_runs with created_at before this date on API startup (SQLite date comparison).
TRAINING_HISTORY_PRUNE_BEFORE = (os.getenv("TRAINING_HISTORY_PRUNE_BEFORE") or "").strip()

H2O_MAX_MODELS = int(os.getenv("H2O_MAX_MODELS", "20"))
H2O_MAX_RUNTIME_SECS = int(os.getenv("H2O_MAX_RUNTIME_SECS", "300"))
H2O_NFOLDS = int(os.getenv("H2O_NFOLDS", "5"))
H2O_SEED = int(os.getenv("H2O_SEED", "42"))

for d in [RAW_UPLOADS_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
