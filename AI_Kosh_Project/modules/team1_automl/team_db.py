import sqlite3
import json
from pathlib import Path
from .config import TEAM_DB_PATH
from .schemas import DatasetMetadata
from .enums import TrainingStatus

_DB_PATH = str(TEAM_DB_PATH)


def _get_conn():
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                total_rows INTEGER,
                total_columns INTEGER,
                size_bytes INTEGER,
                category TEXT DEFAULT 'Uploaded Dataset',
                description TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS training_runs (
                run_id TEXT PRIMARY KEY,
                dataset_id TEXT,
                config TEXT,
                status TEXT DEFAULT 'queued',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)


def _migrate_db():
    """Add new columns to training_runs table if they don't exist."""
    new_columns = [
        ("ml_task", "TEXT DEFAULT ''"),
        ("target_column", "TEXT DEFAULT ''"),
        ("feature_columns", "TEXT DEFAULT '[]'"),
        ("best_model_id", "TEXT DEFAULT ''"),
        ("best_algorithm", "TEXT DEFAULT ''"),
        ("primary_metric", "TEXT DEFAULT ''"),
        ("best_metric_value", "REAL DEFAULT 0"),
        ("model_count", "INTEGER DEFAULT 0"),
        ("dataset_filename", "TEXT DEFAULT ''"),
        ("run_type", "TEXT DEFAULT 'training'"),
    ]
    with _get_conn() as conn:
        # Get existing columns
        cursor = conn.execute("PRAGMA table_info(training_runs)")
        existing_cols = {row["name"] for row in cursor.fetchall()}

        for col_name, col_def in new_columns:
            if col_name not in existing_cols:
                try:
                    conn.execute(f"ALTER TABLE training_runs ADD COLUMN {col_name} {col_def}")
                except Exception:
                    pass

        try:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_training_runs_status_created "
                "ON training_runs(status, created_at DESC)"
            )
        except Exception:
            pass


init_db()
_migrate_db()


def _cleanup_duplicate_datasets():
    with _get_conn() as conn:
        conn.execute("""
            DELETE FROM datasets WHERE rowid NOT IN (
                SELECT MAX(rowid) FROM datasets GROUP BY filename
            )
        """)


_cleanup_duplicate_datasets()


def save_dataset(meta: DatasetMetadata):
    with _get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO datasets (id, filename, total_rows, total_columns, size_bytes, category, description) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (meta.id, meta.filename, meta.total_rows, meta.total_columns, meta.size_bytes, meta.category, meta.description),
        )


def list_datasets() -> list[DatasetMetadata]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM datasets GROUP BY filename ORDER BY rowid DESC"
        ).fetchall()
    return [DatasetMetadata(**dict(r)) for r in rows]


def get_dataset(dataset_id: str):
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
    if row:
        return DatasetMetadata(**dict(row))
    return None


def find_dataset_by_filename(filename: str):
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM datasets WHERE filename = ?", (filename,)).fetchone()
    if row:
        return DatasetMetadata(**dict(row))
    return None


def delete_dataset(dataset_id: str):
    with _get_conn() as conn:
        conn.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))


def save_training_run(run_id: str, req, status: TrainingStatus):
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO training_runs (run_id, dataset_id, config, status) VALUES (?, ?, ?, ?)",
            (run_id, req.dataset_id, json.dumps(req.model_dump(), default=str), status.value),
        )


def update_training_run(run_id: str, status: TrainingStatus):
    with _get_conn() as conn:
        conn.execute(
            "UPDATE training_runs SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE run_id = ?",
            (status.value, run_id),
        )


def get_training_run(run_id: str):
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM training_runs WHERE run_id = ?", (run_id,)).fetchone()
    return dict(row) if row else None


# ── New: Rich run metadata for result persistence ─────────────────────────

def update_run_results(
    run_id: str,
    ml_task: str = "",
    target_column: str = "",
    feature_columns: list[str] | None = None,
    best_model_id: str = "",
    best_algorithm: str = "",
    primary_metric: str = "",
    best_metric_value: float = 0.0,
    model_count: int = 0,
    dataset_filename: str = "",
    run_type: str = "training",
):
    """Update training run with full result metadata after completion."""
    features_json = json.dumps(feature_columns or [])
    with _get_conn() as conn:
        conn.execute(
            """UPDATE training_runs
               SET ml_task = ?, target_column = ?, feature_columns = ?,
                   best_model_id = ?, best_algorithm = ?,
                   primary_metric = ?, best_metric_value = ?,
                   model_count = ?, dataset_filename = ?, run_type = ?,
                   updated_at = CURRENT_TIMESTAMP
               WHERE run_id = ?""",
            (
                ml_task, target_column, features_json,
                best_model_id, best_algorithm,
                primary_metric, best_metric_value,
                model_count, dataset_filename, run_type,
                run_id,
            ),
        )


def list_completed_runs(limit: int | None = None) -> list[dict]:
    """Get completed training/clustering runs with metadata (newest first). Optional limit for faster reads."""
    lim = int(limit) if limit is not None else None
    if lim is not None and lim < 1:
        lim = 1
    sql = (
        """SELECT run_id, dataset_id, status, ml_task, target_column,
                  best_model_id, best_algorithm, primary_metric,
                  best_metric_value, model_count, dataset_filename,
                  run_type, created_at, updated_at
           FROM training_runs
           WHERE status = 'complete'
           ORDER BY datetime(created_at) DESC"""
    )
    if lim is not None:
        sql += f" LIMIT {lim}"
    with _get_conn() as conn:
        rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]


def get_runs_by_dataset(dataset_id: str) -> list[dict]:
    """Find all training runs that used a specific dataset."""
    with _get_conn() as conn:
        rows = conn.execute(
            """SELECT run_id, dataset_id, status, ml_task, target_column,
                      best_model_id, best_algorithm, primary_metric,
                      best_metric_value, model_count, dataset_filename,
                      run_type, created_at, updated_at
               FROM training_runs
               WHERE dataset_id = ? AND status = 'complete'
               ORDER BY created_at DESC""",
            (dataset_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def save_clustering_run(run_id: str, dataset_id: str):
    """Insert a clustering run entry into training_runs."""
    with _get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO training_runs
               (run_id, dataset_id, config, status, run_type)
               VALUES (?, ?, '{}', 'queued', 'clustering')""",
            (run_id, dataset_id),
        )


def prune_training_runs_before(cutoff_date: str) -> int:
    """Delete rows where date(created_at) < cutoff_date (YYYY-MM-DD). Returns deleted row count."""
    with _get_conn() as conn:
        cur = conn.execute(
            "DELETE FROM training_runs WHERE date(created_at) < date(?)",
            (cutoff_date,),
        )
        return cur.rowcount


def find_matching_run(
    dataset_id: str,
    ml_task: str,
    target_column: str,
    feature_columns: list[str],
    run_type: str = "training",
) -> dict | None:
    """Find a previous completed run with the same config (dataset + task + target + features).
    
    This enables automatic cache hits: if the user trains the same dataset 
    with the same config, we return the previous result instead of retraining.
    """
    features_sorted = json.dumps(sorted(feature_columns))
    with _get_conn() as conn:
        rows = conn.execute(
            """SELECT run_id, dataset_id, status, ml_task, target_column,
                      feature_columns, best_model_id, best_algorithm,
                      primary_metric, best_metric_value, model_count,
                      dataset_filename, run_type, created_at
               FROM training_runs
               WHERE dataset_id = ? AND ml_task = ? AND target_column = ?
                     AND run_type = ? AND status = 'complete'
               ORDER BY created_at DESC""",
            (dataset_id, ml_task, target_column, run_type),
        ).fetchall()

    for row in rows:
        r = dict(row)
        # Compare sorted feature lists for exact match
        try:
            stored_features = json.loads(r.get("feature_columns", "[]"))
            if json.dumps(sorted(stored_features)) == features_sorted:
                return r
        except Exception:
            continue
    return None

