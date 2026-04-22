import json
import shutil
from pathlib import Path
from .config import RAW_UPLOADS_DIR, MODELS_DIR, PROCESSED_DATA_DIR, DATA_LIBRARY_LOCAL_ROOT


class LocalStorage:
    def save_dataset(self, dataset_id: str, filename: str, content: bytes) -> str:
        dest_dir = RAW_UPLOADS_DIR / dataset_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        filepath = dest_dir / filename
        filepath.write_bytes(content)
        return str(filepath)

    def get_dataset_path(self, dataset_id: str, filename: str) -> str:
        return str(RAW_UPLOADS_DIR / dataset_id / filename)

    def delete_dataset(self, dataset_id: str, filename: str):
        dest_dir = RAW_UPLOADS_DIR / dataset_id
        if dest_dir.exists():
            shutil.rmtree(dest_dir)

    def save_model(self, run_id: str, filename: str, content: bytes) -> str:
        dest_dir = MODELS_DIR / run_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        filepath = dest_dir / filename
        filepath.write_bytes(content)
        return str(filepath)

    def list_datasets(self) -> list[str]:
        if not RAW_UPLOADS_DIR.exists():
            return []
        return [d.name for d in RAW_UPLOADS_DIR.iterdir() if d.is_dir()]

    # ── Results Persistence (new) ──────────────────────────────────────────

    def save_training_result(self, run_id: str, result_dict: dict) -> str:
        """Persist full training result JSON to local processed_data dir."""
        dest_dir = PROCESSED_DATA_DIR / run_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        filepath = dest_dir / "training_result.json"
        filepath.write_text(json.dumps(result_dict, indent=2, default=str))
        return str(filepath)

    def get_training_result(self, run_id: str) -> dict | None:
        """Load persisted training result from local disk."""
        filepath = PROCESSED_DATA_DIR / run_id / "training_result.json"
        if filepath.exists():
            try:
                return json.loads(filepath.read_text())
            except Exception:
                return None
        return None

    def save_clustering_result(self, run_id: str, result_dict: dict) -> str:
        """Persist full clustering result + elbow JSON to local disk."""
        dest_dir = PROCESSED_DATA_DIR / run_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        filepath = dest_dir / "clustering_result.json"
        filepath.write_text(json.dumps(result_dict, indent=2, default=str))
        return str(filepath)

    def get_clustering_result(self, run_id: str) -> dict | None:
        """Load persisted clustering result from local disk."""
        filepath = PROCESSED_DATA_DIR / run_id / "clustering_result.json"
        if filepath.exists():
            try:
                return json.loads(filepath.read_text())
            except Exception:
                return None
        return None

    def save_model_binary(self, run_id: str, filename: str, local_path: str) -> str:
        """Copy model binary to models dir (may already be there)."""
        dest_dir = MODELS_DIR / run_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / filename
        src = Path(local_path)
        if src.exists() and str(src) != str(dest):
            shutil.copy2(str(src), str(dest))
        return str(dest)

    def get_model_binary(self, run_id: str, filename: str) -> str | None:
        """Return local path to model binary if it exists."""
        filepath = MODELS_DIR / run_id / filename
        if filepath.exists():
            return str(filepath)
        # Also check if the model was saved directly (H2O saves with model name)
        run_dir = MODELS_DIR / run_id
        if run_dir.exists():
            for f in run_dir.iterdir():
                if f.is_file():
                    return str(f)
        return None

    def list_training_runs(self) -> list[str]:
        """List all run IDs that have persisted results."""
        results: list[str] = []
        if PROCESSED_DATA_DIR.exists():
            for d in PROCESSED_DATA_DIR.iterdir():
                if d.is_dir() and (d / "training_result.json").exists():
                    results.append(d.name)
        return results

    def list_data_library(self) -> list[dict]:
        """Match Azure: subfolders = categories; files directly in DATA_LIBRARY_LOCAL_ROOT → folder __root__."""
        _ROOT = "__root__"
        result: list[dict] = []
        if not DATA_LIBRARY_LOCAL_ROOT.exists():
            return result
        base = DATA_LIBRARY_LOCAL_ROOT.resolve()
        root_files: list[dict] = []
        for f in sorted(base.iterdir(), key=lambda p: p.name.lower()):
            if f.is_file() and f.suffix.lower() in (".csv", ".tsv", ".txt"):
                root_files.append({"name": f.name, "size_bytes": f.stat().st_size})
        if root_files:
            result.append({"folder": _ROOT, "files": root_files})
        for sub in sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
            files: list[dict] = []
            for f in sorted(sub.rglob("*"), key=lambda p: str(p).lower()):
                if not f.is_file() or f.suffix.lower() not in (".csv", ".tsv", ".txt"):
                    continue
                rel = f.relative_to(sub)
                name = str(rel).replace("\\", "/")
                files.append({"name": name, "size_bytes": f.stat().st_size})
            if files:
                result.append({"folder": sub.name, "files": files})
        return result

    def download_data_library_file(self, folder: str, file_name: str) -> bytes:
        _ROOT = "__root__"
        base = DATA_LIBRARY_LOCAL_ROOT.resolve()
        safe_name = (file_name or "").strip().lstrip("/").replace("\\", "/")
        if ".." in safe_name or not safe_name:
            raise ValueError("Invalid file name")
        folder_id = (folder or "").strip()
        if folder_id == _ROOT:
            path = (DATA_LIBRARY_LOCAL_ROOT / safe_name).resolve()
        else:
            path = (DATA_LIBRARY_LOCAL_ROOT / folder_id / safe_name).resolve()
        try:
            path.relative_to(base)
        except ValueError:
            raise ValueError("File not found") from None
        if not path.is_file():
            raise ValueError("File not found")
        return path.read_bytes()

    def delete_training_result(self, run_id: str):
        """Delete persisted result from local disk."""
        result_dir = PROCESSED_DATA_DIR / run_id
        if result_dir.exists():
            shutil.rmtree(result_dir)
        model_dir = MODELS_DIR / run_id
        if model_dir.exists():
            shutil.rmtree(model_dir)
