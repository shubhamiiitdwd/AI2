"""
Azure Blob Storage implementation.
Activated when AZURE_STORAGE_CONNECTION_STRING is set, or when
AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY are set (connection string is built the same way as the Azure portal / CLI).

Layout (single container, e.g. AZURE_BLOB_CONTAINER_NAME=aikosh-v2):
  {AUTOML_PREFIX}/datasets/{dataset_id}/{filename}
  {AUTOML_PREFIX}/models/{run_id}/{filename}
  {AUTOML_PREFIX}/results/training/{run_id}/result.json
  {AUTOML_PREFIX}/results/clustering/{run_id}/result.json
  (Module data library: separate — see AZURE_BLOB_DATA_LIBRARY_CONTAINER_PREFIX, container-relative, not under AUTOML_PREFIX.)

AUTOML_PREFIX defaults to "automl" (see AZURE_STORAGE_AUTOML_PREFIX).
"""
import json
import logging
from pathlib import Path
from .config import (
    AZURE_STORAGE_CONNECTION_STRING,
    AZURE_BLOB_CONTAINER_NAME,
    AZURE_STORAGE_AUTOML_PREFIX,
    AZURE_BLOB_DATA_LIBRARY_CONTAINER_PREFIX,
    RAW_UPLOADS_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
)

logger = logging.getLogger(__name__)

_blob_service = None


def _automl_prefix() -> str:
    """Virtual folder under the container; empty string = blobs at container root."""
    return (AZURE_STORAGE_AUTOML_PREFIX or "").strip().strip("/")


def _blob_path(*segments: str) -> str:
    """Build blob name: automl/datasets/... — no leading slash."""
    base = _automl_prefix()
    rest = [s.strip("/").replace("\\", "/") for s in segments if s is not None and s != ""]
    parts = ([base] if base else []) + rest
    return "/".join(parts)


def _get_blob_service():
    global _blob_service
    if _blob_service is None:
        try:
            from azure.core.exceptions import ResourceExistsError
            from azure.storage.blob import BlobServiceClient

            _blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
            container = _blob_service.get_container_client(AZURE_BLOB_CONTAINER_NAME)
            try:
                container.create_container()
            except ResourceExistsError:
                pass
            logger.info(
                "Azure Blob Storage initialized: container=%s prefix=%s/",
                AZURE_BLOB_CONTAINER_NAME,
                _automl_prefix(),
            )
        except Exception as e:
            logger.error(f"Azure Blob init failed: {e}")
            raise
    return _blob_service


class AzureStorage:
    # ── Dataset Operations (existing) ──────────────────────────────────────

    def save_dataset(self, dataset_id: str, filename: str, content: bytes) -> str:
        blob_service = _get_blob_service()
        blob_name = _blob_path("datasets", dataset_id, filename)
        blob_client = blob_service.get_blob_client(AZURE_BLOB_CONTAINER_NAME, blob_name)
        blob_client.upload_blob(content, overwrite=True)

        local_dir = RAW_UPLOADS_DIR / dataset_id
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / filename
        local_path.write_bytes(content)
        return str(local_path)

    def get_dataset_path(self, dataset_id: str, filename: str) -> str:
        local_path = RAW_UPLOADS_DIR / dataset_id / filename
        if local_path.exists():
            return str(local_path)

        blob_service = _get_blob_service()
        blob_name = _blob_path("datasets", dataset_id, filename)
        blob_client = blob_service.get_blob_client(AZURE_BLOB_CONTAINER_NAME, blob_name)
        data = blob_client.download_blob().readall()

        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(data)
        return str(local_path)

    def delete_dataset(self, dataset_id: str, filename: str):
        try:
            blob_service = _get_blob_service()
            blob_name = _blob_path("datasets", dataset_id, filename)
            blob_client = blob_service.get_blob_client(AZURE_BLOB_CONTAINER_NAME, blob_name)
            blob_client.delete_blob()
        except Exception as e:
            logger.warning(f"Azure blob delete failed: {e}")

        import shutil
        local_dir = RAW_UPLOADS_DIR / dataset_id
        if local_dir.exists():
            shutil.rmtree(local_dir)

    def save_model(self, run_id: str, filename: str, content: bytes) -> str:
        blob_service = _get_blob_service()
        blob_name = _blob_path("models", run_id, filename)
        blob_client = blob_service.get_blob_client(AZURE_BLOB_CONTAINER_NAME, blob_name)
        blob_client.upload_blob(content, overwrite=True)

        local_dir = MODELS_DIR / run_id
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / filename
        local_path.write_bytes(content)
        return str(local_path)

    def list_datasets(self) -> list[str]:
        try:
            blob_service = _get_blob_service()
            container_client = blob_service.get_container_client(AZURE_BLOB_CONTAINER_NAME)
            prefix = _blob_path("datasets") + "/"
            blobs = container_client.list_blobs(name_starts_with=prefix)
            ids = set()
            for blob in blobs:
                rel = blob.name[len(prefix) :]
                if "/" in rel:
                    ids.add(rel.split("/", 1)[0])
            return list(ids)
        except Exception:
            return []

    # ── Training Results Persistence (new) ─────────────────────────────────

    def save_training_result(self, run_id: str, result_dict: dict) -> str:
        """Upload full training result JSON to Azure results prefix + local cache."""
        blob_service = _get_blob_service()
        blob_name = _blob_path("results", "training", run_id, "result.json")
        content = json.dumps(result_dict, indent=2, default=str)
        blob_client = blob_service.get_blob_client(AZURE_BLOB_CONTAINER_NAME, blob_name)
        blob_client.upload_blob(content, overwrite=True)

        local_dir = PROCESSED_DATA_DIR / run_id
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / "training_result.json"
        local_path.write_text(content)
        logger.info(f"Training result persisted to Azure + local: {run_id}")
        return str(local_path)

    def get_training_result(self, run_id: str) -> dict | None:
        """Load training result — try local cache first, then Azure."""
        local_path = PROCESSED_DATA_DIR / run_id / "training_result.json"
        if local_path.exists():
            try:
                return json.loads(local_path.read_text())
            except Exception:
                pass

        try:
            blob_service = _get_blob_service()
            blob_name = _blob_path("results", "training", run_id, "result.json")
            blob_client = blob_service.get_blob_client(AZURE_BLOB_CONTAINER_NAME, blob_name)
            data = blob_client.download_blob().readall()
            result = json.loads(data)

            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text(data.decode("utf-8") if isinstance(data, bytes) else data)
            logger.info(f"Training result fetched from Azure and cached: {run_id}")
            return result
        except Exception as e:
            logger.debug(f"No Azure training result for {run_id}: {e}")
            return None

    # ── Clustering Results Persistence (new) ───────────────────────────────

    def save_clustering_result(self, run_id: str, result_dict: dict) -> str:
        """Upload full clustering result JSON to Azure + local cache."""
        blob_service = _get_blob_service()
        blob_name = _blob_path("results", "clustering", run_id, "result.json")
        content = json.dumps(result_dict, indent=2, default=str)
        blob_client = blob_service.get_blob_client(AZURE_BLOB_CONTAINER_NAME, blob_name)
        blob_client.upload_blob(content, overwrite=True)

        local_dir = PROCESSED_DATA_DIR / run_id
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / "clustering_result.json"
        local_path.write_text(content)
        logger.info(f"Clustering result persisted to Azure + local: {run_id}")
        return str(local_path)

    def get_clustering_result(self, run_id: str) -> dict | None:
        """Load clustering result — try local cache, then Azure."""
        local_path = PROCESSED_DATA_DIR / run_id / "clustering_result.json"
        if local_path.exists():
            try:
                return json.loads(local_path.read_text())
            except Exception:
                pass

        try:
            blob_service = _get_blob_service()
            blob_name = _blob_path("results", "clustering", run_id, "result.json")
            blob_client = blob_service.get_blob_client(AZURE_BLOB_CONTAINER_NAME, blob_name)
            data = blob_client.download_blob().readall()
            result = json.loads(data)

            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text(data.decode("utf-8") if isinstance(data, bytes) else data)
            return result
        except Exception as e:
            logger.debug(f"No Azure clustering result for {run_id}: {e}")
            return None

    # ── Model Binary Persistence (new) ─────────────────────────────────────

    def save_model_binary(self, run_id: str, filename: str, local_path: str) -> str:
        """Upload an H2O model binary file from local disk to Azure models prefix."""
        try:
            src = Path(local_path)
            if src.exists():
                blob_service = _get_blob_service()
                blob_name = _blob_path("models", run_id, filename)
                blob_client = blob_service.get_blob_client(AZURE_BLOB_CONTAINER_NAME, blob_name)
                with open(str(src), "rb") as f:
                    blob_client.upload_blob(f, overwrite=True)
                logger.info(f"Model binary uploaded to Azure: {blob_name}")
            return local_path
        except Exception as e:
            logger.warning(f"Failed to upload model binary to Azure: {e}")
            return local_path

    def get_model_binary(self, run_id: str, filename: str) -> str | None:
        """Download model binary from Azure to local cache if it doesn't exist locally."""
        run_dir = MODELS_DIR / run_id
        local_path = run_dir / filename
        if local_path.exists():
            return str(local_path)

        if run_dir.exists():
            for f in run_dir.iterdir():
                if f.is_file():
                    return str(f)

        try:
            blob_service = _get_blob_service()
            blob_name = _blob_path("models", run_id, filename)
            blob_client = blob_service.get_blob_client(AZURE_BLOB_CONTAINER_NAME, blob_name)
            data = blob_client.download_blob().readall()

            run_dir.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(data)
            logger.info(f"Model binary downloaded from Azure: {blob_name}")
            return str(local_path)
        except Exception as e:
            logger.debug(f"No Azure model binary for {run_id}/{filename}: {e}")
            return None

    # ── Run Listing & Cleanup (new) ────────────────────────────────────────

    def list_training_runs(self) -> list[str]:
        """List all training run IDs from Azure results prefix."""
        run_ids = set()
        try:
            blob_service = _get_blob_service()
            container_client = blob_service.get_container_client(AZURE_BLOB_CONTAINER_NAME)
            prefix = _blob_path("results", "training") + "/"
            blobs = container_client.list_blobs(name_starts_with=prefix)
            for blob in blobs:
                rel = blob.name[len(prefix) :]
                if rel and "/" in rel:
                    run_ids.add(rel.split("/", 1)[0])
        except Exception:
            pass

        if PROCESSED_DATA_DIR.exists():
            for d in PROCESSED_DATA_DIR.iterdir():
                if d.is_dir() and (d / "training_result.json").exists():
                    run_ids.add(d.name)

        return list(run_ids)

    def delete_training_result(self, run_id: str):
        """Delete training result from both Azure and local."""
        try:
            blob_service = _get_blob_service()
            container_client = blob_service.get_container_client(AZURE_BLOB_CONTAINER_NAME)
            for base in (
                _blob_path("results", "training", run_id) + "/",
                _blob_path("results", "clustering", run_id) + "/",
            ):
                blobs = container_client.list_blobs(name_starts_with=base)
                for blob in blobs:
                    container_client.delete_blob(blob.name)
        except Exception as e:
            logger.warning(f"Azure result cleanup failed for {run_id}: {e}")

        import shutil
        for base_dir in [PROCESSED_DATA_DIR, MODELS_DIR]:
            target = base_dir / run_id
            if target.exists():
                shutil.rmtree(target)

    # ── Module data library (container root or AZURE_BLOB_DATA_LIBRARY_CONTAINER_PREFIX) ─

    @staticmethod
    def _data_library_list_prefix() -> str:
        """List prefix under the container (not under automl). Empty = container root."""
        base = (AZURE_BLOB_DATA_LIBRARY_CONTAINER_PREFIX or "").strip().strip("/")
        return f"{base}/" if base else ""

    @staticmethod
    def _data_library_root_folder_id() -> str:
        """Group blobs with no '/' in name (at prefix root) for download API."""
        return "__root__"

    def list_data_library(self) -> list[dict]:
        """
        List blobs under the container, grouped by the first path segment (portal "folders"),
        e.g. anonymization/..., feature eng/..., automl/... Loose blobs at the prefix root use folder __root__.
        """
        from collections import defaultdict

        out: dict[str, list[dict]] = defaultdict(list)
        list_prefix = self._data_library_list_prefix()
        try:
            blob_service = _get_blob_service()
            container_client = blob_service.get_container_client(AZURE_BLOB_CONTAINER_NAME)
            if list_prefix:
                blobs = container_client.list_blobs(name_starts_with=list_prefix)
            else:
                blobs = container_client.list_blobs()
            root_id = self._data_library_root_folder_id()
            for blob in blobs:
                rel = blob.name[len(list_prefix) :] if list_prefix else blob.name
                if not rel or rel.endswith("/") or ".." in rel:
                    continue
                # Do not list internal Team1 training blobs (automl/datasets|models|results/...).
                if not list_prefix and rel.startswith("automl/"):
                    p2 = rel.split("/")
                    if len(p2) >= 2 and p2[0] == "automl" and p2[1] in ("datasets", "models", "results"):
                        continue
                parts = rel.split("/")
                if len(parts) == 1:
                    folder = root_id
                    file_rel = parts[0]
                else:
                    folder = parts[0]
                    file_rel = "/".join(parts[1:])
                if not folder or not file_rel or ".." in file_rel:
                    continue
                try:
                    sz = int(getattr(blob, "size", 0) or 0)
                except (TypeError, ValueError):
                    sz = 0
                out[folder].append({"name": file_rel, "size_bytes": sz})
        except Exception as e:
            logger.warning("list_data_library Azure: %s", e)
            return []

        result: list[dict] = []
        for folder in sorted(out.keys(), key=str.lower):
            by_name: dict[str, int] = {}
            for f in out[folder]:
                n = f.get("name") or ""
                if not n:
                    continue
                by_name[n] = max(by_name.get(n, 0), int(f.get("size_bytes") or 0))
            file_list = [{"name": n, "size_bytes": s} for n, s in sorted(by_name.items(), key=lambda x: x[0].lower())]
            if file_list:
                result.append({"folder": folder, "files": file_list})
        return result

    def download_data_library_file(self, folder: str, file_name: str) -> bytes:
        """Blob path: {list_prefix} or {list_prefix}{folder}/ for nested files; __root__ = only filename under list_prefix."""
        list_prefix = self._data_library_list_prefix()
        root_id = self._data_library_root_folder_id()
        safe_folder = (folder or "").strip().strip("/").replace("\\", "/")
        safe_name = (file_name or "").strip().lstrip("/").replace("\\", "/")
        if ".." in safe_name or (safe_folder and ".." in safe_folder):
            raise ValueError("Invalid folder or file name")
        if not safe_name:
            raise ValueError("Invalid file name")
        if not safe_folder:
            raise ValueError("Invalid folder id")

        if safe_folder == root_id:
            blob_name = f"{list_prefix}{safe_name}" if list_prefix else safe_name
        else:
            blob_name = f"{list_prefix}{safe_folder}/{safe_name}" if list_prefix else f"{safe_folder}/{safe_name}"
        blob_service = _get_blob_service()
        blob_client = blob_service.get_blob_client(AZURE_BLOB_CONTAINER_NAME, blob_name)
        return blob_client.download_blob().readall()
