"""
Core orchestrator for the AutoML wizard.
All business logic lives here; router.py delegates to these functions.
"""
import uuid
import re
import math
import asyncio
import threading
import logging
import json
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from .config import (
    STORAGE_MODE, AI_MODE, RAW_UPLOADS_DIR, MODELS_DIR, PROCESSED_DATA_DIR,
)
from .schemas import (
    DatasetMetadata, DatasetColumnsResponse, DatasetPreviewResponse,
    AIRecommendRequest, AIRecommendResponse,
    ValidateConfigRequest, ValidateConfigResponse,
    TrainingStartRequest, TrainingStartResponse, TrainingStatusResponse,
    LeaderboardResponse, ModelResult, FeatureImportanceResponse,
    ConfusionMatrixResponse, ResidualsResponse, ExportResponse, ColumnInfo,
    HoldoutEvaluationResponse, ClassificationHoldoutRow, RegressionHoldoutRow,
    UseCaseSuggestion, UseCaseSuggestionsResponse,
    PredictRequest, PredictResponse, PredictionModelResult,
    GainsLiftResponse, GainsLiftRow,
    BestModelSummary, AISummaryResponse,
    AutoDetectTaskResponse,
    ClusteringStartRequest, ClusteringStartResponse,
    ClusteringResultResponse, ClusterMetrics, StabilityResult,
    CandidateModelResult, ClusterSummary, ClusterFeatureImportance,
    DimensionReductionPoint, ElbowResponse, ElbowDataPoint,
    TrainingRunSummary, TrainingHistoryResponse,
    DatasetWorkflowInsightResponse,
    ClusteringLabeledPreviewResponse,
    TextInsightResponse,
    DataLibraryFileRef, DataLibraryFolderInfo, DataLibraryIndexResponse,
    DataLibraryImportRequest, DataLibraryImportResponse,
)
from .enums import MLTask, TrainingStatus
from . import data_processor
from . import storage_service
from . import ai_service
from .ai_huggingface import _is_id_like as _column_name_id_like
from . import h2o_engine
from . import team_db
from . import clustering_engine

logger = logging.getLogger(__name__)

_active_runs: dict[str, dict] = {}
_websocket_connections: dict[str, list[WebSocket]] = {}


# ── Dataset Management ─────────────────────────────────────────────────────

def _sanitize_for_json(obj):
    """Recursively convert numpy/pandas types to native Python for JSON serialization."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_sanitize_for_json(v) for v in obj.tolist()]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


def _rule_dataset_workflow_insight(
    columns: list[ColumnInfo],
    filename: str,
    total_rows: int,
    total_cols: int,
) -> DatasetWorkflowInsightResponse:
    """Heuristic tabular vs raw/unstructured when Azure OpenAI is unavailable."""
    if not columns:
        return DatasetWorkflowInsightResponse(
            is_structured_tabular=False,
            needs_data_exchange=True,
            suggest_automl=False,
            headline="No column metadata",
            detail="Upload a CSV with readable headers. If this is raw text or logs, use Data Exchange to parse features first.",
            source="rules",
            data_characteristics="No columns were inferred — the file may be empty, wrong format, or missing a header row.",
            preprocessing_guidance="Open the file in Data Exchange or a spreadsheet tool: confirm delimiter, encoding (UTF-8), and a single header row.",
            feature_engineering_guidance="After the dataset parses into columns, revisit this wizard; feature engineering only applies once fields are identifiable.",
        )

    n_num = sum(1 for c in columns if (c.dtype or "") in ("int64", "float64", "int32", "float32"))
    n_obj = sum(1 for c in columns if "object" in (c.dtype or "") or c.dtype == "object")

    long_text_cols = 0
    for c in columns:
        sv = c.sample_values or []
        if not sv:
            continue
        lens = [len(str(v)) for v in sv[:5]]
        if lens and sum(lens) / len(lens) > 100:
            long_text_cols += 1

    single_text_blob = total_cols == 1 and n_obj >= 1
    mostly_long_text = long_text_cols >= max(2, (total_cols + 1) // 2) and n_num == 0
    sparse_types = total_cols >= 2 and n_num == 0 and n_obj == total_cols

    needs_exchange = single_text_blob or mostly_long_text or (sparse_types and long_text_cols >= 1)

    structured = (
        not needs_exchange
        and total_cols >= 2
        and (n_num >= 1 or (n_obj >= 2 and all(c.unique_count <= max(100, total_rows // 3) for c in columns)))
    )

    if needs_exchange:
        return DatasetWorkflowInsightResponse(
            is_structured_tabular=False,
            needs_data_exchange=True,
            suggest_automl=False,
            headline="Dataset looks unstructured or needs feature engineering",
            detail=(
                "Columns look like free text, a single unparsed field, or lack numeric/categorical structure for AutoML. "
                "Use Data Exchange to clean, parse, and engineer features, then return here for AutoML or clustering."
            ),
            source="rules",
            data_characteristics=(
                "Semi-structured or unstructured for classic AutoML: long text fields, a single blob column, or mostly "
                "string columns without clear numeric/categorical signals."
            ),
            preprocessing_guidance=(
                "Use Data Exchange (or similar) to tokenize or split text, standardize dates and IDs, drop empty rows, "
                "and fix inconsistent labels before training."
            ),
            feature_engineering_guidance=(
                "Plan explicit features from text (keywords, TF-IDF, embeddings), geospatial fields, or nested JSON — "
                "AutoML expects one row per observation with scalar or low-cardinality categorical inputs."
            ),
        )

    if structured:
        return DatasetWorkflowInsightResponse(
            is_structured_tabular=True,
            needs_data_exchange=False,
            suggest_automl=True,
            headline="Structured tabular data detected",
            detail=(
                f"'{filename}' has {total_cols} columns with types suitable for H2O AutoML or clustering. "
                "Continue with Configure Data, pick a target (for supervised tasks), and run AutoML—or choose Clustering for unsupervised exploration."
            ),
            source="rules",
            data_characteristics=(
                f"Tabular layout: {n_num} numeric-like column(s), {n_obj} object/string column(s), {total_rows} rows — "
                "typical for supervised or clustering workflows."
            ),
            preprocessing_guidance=(
                "Handle missing values and outliers, align dtypes with semantics (e.g. IDs as strings), and cap "
                "extreme high-cardinality categoricals before training."
            ),
            feature_engineering_guidance=(
                "Optional: scaling for distance-based models, encoding for categoricals, and derived ratios or "
                "binning for skewed numerics. Revisit Data Exchange if domain-specific transforms are required."
            ),
        )

    return DatasetWorkflowInsightResponse(
        is_structured_tabular=True,
        needs_data_exchange=False,
        suggest_automl=True,
        headline="Likely tabular — review columns in the next step",
        detail="This CSV is probably usable for AutoML or clustering. If any column is raw text, drop or transform it before training.",
        source="rules",
        data_characteristics=(
            f"Mixed or ambiguous structure across {total_cols} columns — treat as tabular candidate but verify each "
            "column’s role before modeling."
        ),
        preprocessing_guidance=(
            "Inspect dtypes, null rates, and cardinality in Configure Data; fix misclassified columns and leakage risks."
        ),
        feature_engineering_guidance=(
            "If performance is weak, try feature interactions or encoding changes in Data Exchange, then re-import or "
            "re-upload the refined CSV."
        ),
    )


async def get_dataset_workflow_insight(dataset_id: str) -> DatasetWorkflowInsightResponse:
    ds = team_db.get_dataset(dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")
    storage = storage_service.get_storage()
    filepath = storage.get_dataset_path(dataset_id, ds.filename)
    cols_resp = data_processor.get_columns(filepath, dataset_id)
    columns = cols_resp.columns
    base = _rule_dataset_workflow_insight(columns, ds.filename, ds.total_rows, ds.total_columns)

    if AI_MODE != "azure":
        return base

    try:
        ai = ai_service.get_ai_service()
        if not hasattr(ai, "dataset_workflow_insight"):
            return base
        data = await ai.dataset_workflow_insight(columns, ds.filename, ds.total_rows, ds.total_columns)
        dc = str(data.get("data_characteristics") or "").strip()
        pg = str(data.get("preprocessing_guidance") or "").strip()
        fe = str(data.get("feature_engineering_guidance") or "").strip()
        return DatasetWorkflowInsightResponse(
            is_structured_tabular=bool(data.get("is_structured_tabular", base.is_structured_tabular)),
            needs_data_exchange=bool(data.get("needs_data_exchange", base.needs_data_exchange)),
            suggest_automl=bool(data.get("suggest_automl", base.suggest_automl)),
            headline=data.get("headline") or base.headline,
            detail=data.get("detail") or base.detail,
            source="azure",
            data_characteristics=dc or base.data_characteristics,
            preprocessing_guidance=pg or base.preprocessing_guidance,
            feature_engineering_guidance=fe or base.feature_engineering_guidance,
        )
    except Exception as e:
        logger.debug("dataset_workflow_insight Azure fallback: %s", e)
        return base


def register_dataset_from_bytes(
    filename: str,
    content: bytes,
    *,
    category: str | None = None,
    description: str | None = None,
) -> DatasetMetadata:
    """Save bytes like an upload and persist metadata (used by multipart upload and data.gov.in import)."""
    existing = team_db.find_dataset_by_filename(filename)
    if existing:
        storage = storage_service.get_storage()
        try:
            storage.delete_dataset(existing.id, existing.filename)
        except Exception:
            pass
        team_db.delete_dataset(existing.id)

    dataset_id = str(uuid.uuid4())[:8]
    storage = storage_service.get_storage()
    filepath = storage.save_dataset(dataset_id, filename, content)

    meta = data_processor.get_metadata(filepath, dataset_id, filename)
    updates: dict = {}
    if category is not None:
        updates["category"] = category
    if description is not None:
        updates["description"] = description
    if updates:
        meta = meta.model_copy(update=updates)
    team_db.save_dataset(meta)
    return meta


async def upload_dataset(file: UploadFile) -> DatasetMetadata:
    content = await file.read()
    filename = file.filename or "dataset.csv"
    return register_dataset_from_bytes(filename, content)


async def list_datasets() -> list[DatasetMetadata]:
    return team_db.list_datasets()


def list_data_library_index() -> DataLibraryIndexResponse:
    storage = storage_service.get_storage()
    list_fn = getattr(storage, "list_data_library", None)
    if not callable(list_fn):
        return DataLibraryIndexResponse(source="none", folders=[])
    try:
        raw: list[dict] = list_fn()  # type: ignore[assignment]
    except Exception as e:
        logger.warning("list_data_library: %s", e)
        hint = (
            "Check AI_Kosh_Project/.env: set AZURE_STORAGE_CONNECTION_STRING (or both "
            "AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY) for Azure, "
            "or use local module files under shared_workspace/data_library (or DATA_LIBRARY_LOCAL_DIR). "
            "Optionally set AZURE_BLOB_CONTAINER_NAME and AZURE_BLOB_DATA_LIBRARY_CONTAINER_PREFIX. "
            "Restart the API after changing .env."
        )
        raise HTTPException(
            status_code=503,
            detail=f"{e!s}. {hint}" if f"{e!s}".strip() else hint,
        ) from e

    folders: list[DataLibraryFolderInfo] = []
    for item in raw:
        files = [
            DataLibraryFileRef(name=str(f.get("name") or ""), size_bytes=int(f.get("size_bytes") or 0))
            for f in (item.get("files") or [])
            if f.get("name")
        ]
        if files:
            folders.append(DataLibraryFolderInfo(folder=str(item.get("folder") or ""), files=files))
    return DataLibraryIndexResponse(
        source="azure" if STORAGE_MODE == "azure" else "local",
        folders=folders,
    )


_LIBRARY_REJECT_SUFFIX = (
    " This file is not supported in the AutoML module, which only accepts tabular datasets for "
    "classification, regression, or clustering. Choose another file from the module library, upload a different CSV, "
    "or use Data Exchange to prepare your data first."
)


async def import_data_library_dataset(req: DataLibraryImportRequest) -> DataLibraryImportResponse:
    storage = storage_service.get_storage()
    down = getattr(storage, "download_data_library_file", None)
    if not callable(down):
        raise HTTPException(501, "Data library is not available in this deployment (storage backend missing).")

    folder = (req.folder or "").strip()
    filename = (req.filename or "").strip()
    if not folder or not filename:
        raise HTTPException(400, "folder and filename are required")

    try:
        content: bytes = down(folder, filename)  # type: ignore[misc]
    except Exception as e:
        logger.warning("Data library download failed: %s", e)
        raise HTTPException(404, "File not found in the data library") from e

    leaf = filename.rsplit("/", 1)[-1].strip() or "dataset.csv"
    meta = register_dataset_from_bytes(
        leaf,
        content,
        category=f"Module library / {folder}",
        description="Imported from shared module data library (Azure/local)",
    )
    try:
        insight = await get_dataset_workflow_insight(meta.id)
    except Exception as e:
        await delete_dataset(meta.id)
        raise HTTPException(500, f"Failed to profile imported file: {e}") from e

    if not insight.is_structured_tabular:
        await delete_dataset(meta.id)
        base = f"{insight.headline} {insight.detail}".strip()
        if insight.data_characteristics:
            base = f"{base} {insight.data_characteristics}".strip()
        msg = (base or "This file does not look like structured, tabular data for AutoML.") + _LIBRARY_REJECT_SUFFIX
        if AI_MODE == "azure" and (
            (insight.preprocessing_guidance and len(insight.preprocessing_guidance) > 20)
            or (insight.feature_engineering_guidance and len(insight.feature_engineering_guidance) > 20)
        ):
            msg = f"{base}\n\n{insight.preprocessing_guidance} {insight.feature_engineering_guidance}".strip() + _LIBRARY_REJECT_SUFFIX
        return DataLibraryImportResponse(accepted=False, message=msg, insight=insight)

    return DataLibraryImportResponse(accepted=True, dataset=meta, insight=insight)


def _heuristic_mall_customers_clustering(filename: str, columns: list[ColumnInfo]) -> bool:
    """
    UCI / Kaggle Mall Customers pattern: filename OR columns (income + spending, optional Age/Gender).
    """
    fn = re.sub(r"[^a-z0-9]+", " ", (filename or "").lower()).strip()
    name_mall = "mall" in fn and ("customer" in fn or "customers" in fn)
    blob = " ".join(c.name.lower() for c in columns)
    has_spend = "spend" in blob
    has_income = "income" in blob or "k$" in blob
    has_demo = "age" in blob or "gender" in blob
    num_cols = sum(
        1 for c in columns
        if (c.dtype or "").lower() in ("int64", "int32", "float64", "float32")
    )
    if has_spend and has_income and (has_demo or num_cols >= 2):
        return True
    if name_mall and has_spend and has_income:
        return True
    if name_mall and has_income and has_demo and num_cols >= 2:
        return True
    return False


def _no_vetted_supervised_targets(classification_names: list[str], regression_names: list[str]) -> bool:
    return not classification_names and not regression_names


def _filename_suggests_clustering(filename: str) -> bool:
    """Filename keywords common for unsupervised / segmentation datasets (not exact science)."""
    fn = (filename or "").lower()
    keys = (
        "cluster", "kmeans", "k-means", "k_means", "unsupervis", "unsup_",
        "segment", "rfm", "grouping", " moons", "blob", "customer_segment", "anomaly_group",
    )
    if any(k in fn for k in keys):
        return True
    return False


def _columns_suggest_unsupervised_grouping(columns: list[ColumnInfo]) -> bool:
    """Several numeric + optional categoricals — typical for clustering when not a clear Y column."""
    blob = " ".join(c.name.lower() for c in columns)
    numish = [
        c for c in columns
        if (c.dtype or "").lower() in ("int64", "int32", "float64", "float32")
    ]
    if len(numish) >= 3 and any(
        w in blob for w in (
            "spend", "income", "revenue", "recency", "frequency", "monetary", "rfm", "amount", "score", "lat", "long",
        )
    ):
        return True
    return False


MALL_CUSTOMERS_AUTO_DETECT_AZURE_HINT = (
    "The host detected a Mall Customers / customer-segmentation style table. Your JSON field \"task\" MUST be "
    '"clustering". Unsupervised customer segments (income, spending, age) — NOT classifying Gender as the main task. '
    "Include a clustering suggestion with target_hint exactly \"No target (unsupervised)\" first."
)

GENERIC_NO_SUPERVISED_TARGETS_HINT = (
    "The host found no vetted classification (2–4 classes) or regression targets. For this app, the appropriate "
    'default is often task \"clustering\" for exploratory grouping / segments, unless the user metadata clearly '
    'implies a different supervised use case. Include at least one clustering suggestion first when clustering fits.'
)

FILENAME_CLUSTERING_HINT = (
    "The file name suggests an unsupervised clustering/segmentation style problem. If columns support finding groups "
    'without a single prediction target, set \"task\" to \"clustering\" and lead with a clustering use case.'
)

COLUMN_GROUPING_CLUSTERING_HINT = (
    "The columns look like a behavioral / RFM / basket-style feature set (several numerics, segment-like names). If the "
    "goal is to discover customer or entity groups, prefer task \"clustering\" over classifying a weak side column."
)


def _build_clustering_host_hint(
    filename: str,
    columns: list[ColumnInfo],
    cls_t: list[str],
    reg_t: list[str],
) -> str | None:
    """Single strongest hint to append for Azure; None if we add nothing extra."""
    if _heuristic_mall_customers_clustering(filename, columns):
        return MALL_CUSTOMERS_AUTO_DETECT_AZURE_HINT
    if _no_vetted_supervised_targets(cls_t, reg_t) and len(columns) >= 2:
        return GENERIC_NO_SUPERVISED_TARGETS_HINT
    if _filename_suggests_clustering(filename):
        return FILENAME_CLUSTERING_HINT
    if not reg_t and _columns_suggest_unsupervised_grouping(columns) and len(cls_t) <= 1:
        return COLUMN_GROUPING_CLUSTERING_HINT
    return None


def _should_force_output_clustering(
    filename: str,
    columns: list[ColumnInfo],
    cls_t: list[str],
    reg_t: list[str],
) -> bool:
    """
    When to pin task=clustering after auto-detect (Let AI decide → clustering pipeline).
    Conservative: if vetted regression targets exist, do not force (use Azure / user choice).
    """
    if reg_t:
        if _heuristic_mall_customers_clustering(filename, columns) or _filename_suggests_clustering(filename):
            return True
        return False
    if _heuristic_mall_customers_clustering(filename, columns):
        return True
    if _no_vetted_supervised_targets(cls_t, reg_t) and len(columns) >= 2:
        return True
    if _filename_suggests_clustering(filename):
        return True
    if _columns_suggest_unsupervised_grouping(columns) and len(cls_t) <= 1:
        return True
    return False


def _prioritize_clustering_suggestions(suggestions: list[UseCaseSuggestion]) -> list[UseCaseSuggestion]:
    cl = [s for s in suggestions if s.ml_task == "clustering"]
    rest = [s for s in suggestions if s.ml_task != "clustering"]
    return (cl + rest)[:6]


async def preview_dataset(dataset_id: str, rows: int = 10) -> DatasetPreviewResponse:
    ds = team_db.get_dataset(dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")
    storage = storage_service.get_storage()
    filepath = storage.get_dataset_path(dataset_id, ds.filename)
    return data_processor.get_preview(filepath, dataset_id, rows)


async def get_dataset_columns(dataset_id: str) -> DatasetColumnsResponse:
    ds = team_db.get_dataset(dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")
    storage = storage_service.get_storage()
    filepath = storage.get_dataset_path(dataset_id, ds.filename)
    return data_processor.get_columns(filepath, dataset_id)


async def delete_dataset(dataset_id: str):
    ds = team_db.get_dataset(dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")
    storage = storage_service.get_storage()
    storage.delete_dataset(dataset_id, ds.filename)
    team_db.delete_dataset(dataset_id)
    return {"message": "Dataset deleted"}


# ── AI Recommendation ──────────────────────────────────────────────────────

async def ai_recommend(req: AIRecommendRequest) -> AIRecommendResponse:
    ds = team_db.get_dataset(req.dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")
    storage = storage_service.get_storage()
    filepath = storage.get_dataset_path(req.dataset_id, ds.filename)
    columns_info = data_processor.get_columns(filepath, req.dataset_id)

    ai = ai_service.get_ai_service()
    return await ai.recommend(columns_info.columns, req.use_case)


async def suggest_usecases(dataset_id: str) -> UseCaseSuggestionsResponse:
    ds = team_db.get_dataset(dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")
    storage = storage_service.get_storage()
    filepath = storage.get_dataset_path(dataset_id, ds.filename)
    columns_info = data_processor.get_columns(filepath, dataset_id)

    import pandas as pd

    cols_all = columns_info.columns
    cols_ai = cols_all[:40]
    cls_targets = [c.name for c in _classification_target_columns(cols_all)]
    reg_targets = [c.name for c in _regression_target_columns(cols_all, ds.total_rows)]
    sample_rows = pd.read_csv(filepath, nrows=25).fillna("").to_dict(orient="records")

    ai = ai_service.get_ai_service()
    try:
        suggestions, meta = await ai.suggest_usecases(
            cols_ai,
            sample_rows,
            ds.filename,
            classification_targets=cls_targets,
            regression_targets=reg_targets,
            total_rows=int(ds.total_rows or 0),
        )
        return UseCaseSuggestionsResponse(
            dataset_id=dataset_id,
            suggestions=suggestions,
            reasoning=str(meta.get("reasoning", "") or ""),
            confidence=str(meta.get("confidence", "") or ""),
            recommended_task=str(meta.get("task", "") or ""),
        )
    except Exception as exc:
        logger.warning("AI use-case suggestion failed, falling back to built-in rules: %s", exc)
        from . import ai_huggingface

        suggestions, _ = await ai_huggingface.suggest_usecases(
            cols_ai,
            sample_rows,
            ds.filename,
            classification_targets=cls_targets,
            regression_targets=reg_targets,
            total_rows=int(ds.total_rows or 0),
        )
        return UseCaseSuggestionsResponse(dataset_id=dataset_id, suggestions=suggestions)


def _classification_target_columns(columns: list[ColumnInfo]) -> list[ColumnInfo]:
    """Targets valid for classification in this app: 2–4 distinct classes only."""
    out: list[ColumnInfo] = []
    for c in columns:
        if _column_name_id_like(c.name):
            continue
        u = int(c.unique_count or 0)
        if 2 <= u <= 4:
            out.append(c)
    return sorted(out, key=lambda x: (-int(x.unique_count or 0), x.name))


def _regression_target_columns(columns: list[ColumnInfo], total_rows: int) -> list[ColumnInfo]:
    """Numeric columns with enough distinct values to behave as a regression target (not low-cardinality class)."""
    n = max(int(total_rows or 0), 1)
    out: list[ColumnInfo] = []
    for c in columns:
        if _column_name_id_like(c.name):
            continue
        dt = (c.dtype or "").lower()
        u = int(c.unique_count or 0)
        if dt not in ("float64", "float32", "int64", "int32"):
            continue
        if u <= 4:
            continue
        min_unique = max(12, min(200, max(15, n // 25)))
        if u < min_unique:
            continue
        if "int" in dt and u > n * 0.97 and n > 100:
            continue
        out.append(c)
    return sorted(out, key=lambda x: (-int(x.unique_count or 0), x.name))[:20]


async def auto_detect_task(dataset_id: str) -> AutoDetectTaskResponse:
    ds = team_db.get_dataset(dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")
    storage = storage_service.get_storage()
    filepath = storage.get_dataset_path(dataset_id, ds.filename)
    columns_info = data_processor.get_columns(filepath, dataset_id)

    import pandas as pd

    cols_all = columns_info.columns
    cols_ai = cols_all[:40]
    sample_rows = pd.read_csv(filepath, nrows=25).fillna("").to_dict(orient="records")
    cls_targets = [c.name for c in _classification_target_columns(cols_all)]
    reg_targets = [c.name for c in _regression_target_columns(cols_all, ds.total_rows)]

    host_hint = _build_clustering_host_hint(ds.filename, cols_all, cls_targets, reg_targets)
    force_clustering = _should_force_output_clustering(ds.filename, cols_all, cls_targets, reg_targets)
    detect_kw: dict = {
        "classification_targets": cls_targets,
        "regression_targets": reg_targets,
        "total_rows": int(ds.total_rows or 0),
    }
    if host_hint:
        detect_kw["host_task_hint"] = host_hint

    ai = ai_service.get_ai_service()
    try:
        out = await ai.auto_detect_task(
            cols_ai,
            sample_rows,
            ds.filename,
            **detect_kw,
        )
    except Exception as exc:
        logger.warning("AI auto-detect failed, using rule-based: %s", exc)
        from . import ai_huggingface

        out = await ai_huggingface.auto_detect_task(
            cols_ai,
            sample_rows,
            ds.filename,
            **detect_kw,
        )

    # Pin to clustering for Let-AI-Decide when heuristics + host rules say so (Mall, no Y, filename, RFM-style, etc.)
    if force_clustering:
        merged_reason = (out.reasoning or "").strip()
        if AI_MODE == "azure":
            prefix = "Azure + host: clustering is the best fit for this table. "
        else:
            prefix = "Host rules: clustering is the best fit for this table. "
        out = out.model_copy(
            update={
                "task": "clustering",
                "reasoning": f"{prefix}{merged_reason}".strip(),
                "suggestions": _prioritize_clustering_suggestions(list(out.suggestions)),
            }
        )
    elif (out.task or "").lower() == "clustering":
        out = out.model_copy(
            update={
                "suggestions": _prioritize_clustering_suggestions(list(out.suggestions)),
            }
        )
    return out


async def _generate_usecase_suggestions_gpt(columns: list[ColumnInfo], filename: str) -> list[UseCaseSuggestion]:
    """Use Azure OpenAI GPT-4o to generate intelligent use-case suggestions."""
    from .config import (
        AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
        AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION,
    )
    from openai import AsyncAzureOpenAI

    client = AsyncAzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )

    col_desc = "\n".join(
        f"- {c.name} ({c.dtype}, {c.null_count} nulls, {c.unique_count} unique)"
        for c in columns
    )

    response = await client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a data science expert. Given a dataset's column metadata and filename, "
                    "suggest up to 5 practical machine-learning use cases. "
                    "Each use case must include:\n"
                    "  - use_case: a clear, one-sentence description of the ML objective.\n"
                    "  - ml_task: one of 'classification', 'regression', or 'clustering'.\n"
                    "  - target_hint: the column name to use as the target (or 'No target (unsupervised)' for clustering).\n\n"
                    "Return ONLY a JSON array (no markdown, no explanations):\n"
                    '[{"use_case": "...", "ml_task": "...", "target_hint": "..."}, ...]'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Dataset filename: {filename}\n\n"
                    f"Columns:\n{col_desc}\n\n"
                    f"Suggest up to 5 use cases. JSON array only."
                ),
            },
        ],
        max_tokens=600,
        temperature=0.4,
    )

    text = response.choices[0].message.content.strip()

    # Parse the JSON array from the response (handle markdown fences)
    cleaned = text
    if cleaned.startswith("```"):
        first_nl = cleaned.index("\n") if "\n" in cleaned else len(cleaned)
        cleaned = cleaned[first_nl + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    items = json.loads(cleaned)
    if isinstance(items, dict) and "suggestions" in items:
        items = items["suggestions"]

    valid_cols = {c.name for c in columns}
    suggestions: list[UseCaseSuggestion] = []
    for item in items[:5]:
        ml_task = item.get("ml_task", "classification").lower()
        if ml_task not in ("classification", "regression", "clustering"):
            ml_task = "classification"
        target = item.get("target_hint", "")
        # Validate target column exists (skip bad hallucinations)
        if ml_task != "clustering" and target not in valid_cols:
            target_lower = target.lower()
            matched = next((c for c in valid_cols if c.lower() == target_lower), None)
            if matched:
                target = matched
            else:
                continue
        suggestions.append(UseCaseSuggestion(
            use_case=item.get("use_case", ""),
            ml_task=ml_task,
            target_hint=target,
        ))

    if not suggestions:
        raise ValueError("GPT-4o returned no valid suggestions")

    return suggestions


def _generate_usecase_suggestions(columns: list[ColumnInfo], filename: str) -> list[UseCaseSuggestion]:
    """Rule-based fallback for use-case suggestions (used when Azure is unavailable)."""
    if not columns:
        return []

    def _is_numeric(col: ColumnInfo) -> bool:
        return col.dtype in ("float64", "int64")

    def _is_id_like(name: str) -> bool:

        n = name.lower().strip()
        if n in {"id", "uuid", "guid", "index", "rownum", "serial"}:
            return True
        if n in {"customerid", "orderid", "userid", "productid"}:
            return True
        if n.endswith("_id") or n.startswith("id_"):
            return True
        return re.search(r"(^|_)(uuid|guid|rownum|serial|index)($|_)", n) is not None

    def _task_scores(col: ColumnInfo) -> tuple[int, int]:
        """Return (classification_score, regression_score) for this column as target."""
        name = col.name.lower()
        cls_score = 0
        reg_score = 0

        # Strong name priors
        if any(k in name for k in ("class", "label", "species", "segment", "status", "category", "type", "churn", "survived", "fraud", "default", "outcome")):
            cls_score += 70
        if any(k in name for k in ("price", "cost", "amount", "revenue", "sales", "value", "score", "rate", "salary", "income", "demand", "load", "weight")):
            reg_score += 70

        # Data-type priors
        if col.dtype == "object" or col.unique_count <= 20:
            cls_score += 30
        if _is_numeric(col) and col.unique_count > 20:
            reg_score += 30

        # Avoid likely identifiers
        if _is_id_like(col.name):
            cls_score -= 120
            reg_score -= 120

        # Penalize mostly-missing columns
        if col.null_count > 0:
            cls_score -= 5
            reg_score -= 5

        return cls_score, reg_score

    numeric_cols = [c for c in columns if _is_numeric(c) and not _is_id_like(c.name)]
    candidate_cols = [c for c in columns if not _is_id_like(c.name)]

    scored: list[tuple[ColumnInfo, int, int]] = []
    for c in candidate_cols:
        cls, reg = _task_scores(c)
        scored.append((c, cls, reg))

    # Sort by strongest target suitability per task
    cls_sorted = [x for x in sorted(scored, key=lambda t: t[1], reverse=True) if x[1] > 0]
    reg_sorted = [x for x in sorted(scored, key=lambda t: t[2], reverse=True) if x[2] > 0]

    suggestions: list[UseCaseSuggestion] = []

    # File-level priors can lift specific tasks
    fn = filename.lower()
    if any(k in fn for k in ("iris", "species", "fraud", "churn", "class", "customer_segment")):
        if cls_sorted:
            c = cls_sorted[0][0]
            suggestions.append(UseCaseSuggestion(
                use_case=f"Classify {c.name} from the remaining features",
                ml_task="classification",
                target_hint=c.name,
            ))
    if any(k in fn for k in ("housing", "price", "sales", "revenue", "cost", "regression")):
        if reg_sorted:
            c = reg_sorted[0][0]
            suggestions.append(UseCaseSuggestion(
                use_case=f"Predict {c.name} as a continuous value",
                ml_task="regression",
                target_hint=c.name,
            ))

    # Top classification suggestion(s)
    for c, _, _ in cls_sorted[:2]:
        suggestions.append(UseCaseSuggestion(
            use_case=f"Classify {c.name} based on other columns",
            ml_task="classification",
            target_hint=c.name,
        ))

    # Top regression suggestion(s)
    for c, _, _ in reg_sorted[:2]:
        suggestions.append(UseCaseSuggestion(
            use_case=f"Predict {c.name} using regression",
            ml_task="regression",
            target_hint=c.name,
        ))

    # Clustering suggestion for datasets with sufficient numeric signals.
    if len(numeric_cols) >= 2:
        top_feats = ", ".join(c.name for c in numeric_cols[:3])
        suggestions.append(UseCaseSuggestion(
            use_case=f"Cluster records into groups using numeric features ({top_feats})",
            ml_task="clustering",
            target_hint="No target (unsupervised)",
        ))

    # Deduplicate and keep diverse top 5
    unique: list[UseCaseSuggestion] = []
    seen = set()
    for s in suggestions:
        key = (s.ml_task, s.target_hint)
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)

    # Ensure at least one fallback exists
    if not unique:
        c = columns[-1]
        fallback_task = "regression" if _is_numeric(c) and c.unique_count > 20 else "classification"
        unique.append(UseCaseSuggestion(
            use_case=f"Model {c.name} from the remaining features",
            ml_task=fallback_task,
            target_hint=c.name,
        ))

    return unique[:5]


async def validate_config(req: ValidateConfigRequest) -> ValidateConfigResponse:
    ds = team_db.get_dataset(req.dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")
    storage = storage_service.get_storage()
    filepath = storage.get_dataset_path(req.dataset_id, ds.filename)
    return data_processor.validate_config(filepath, req)


# ── Training ───────────────────────────────────────────────────────────────

async def start_training(req: TrainingStartRequest) -> TrainingStartResponse:
    ds = team_db.get_dataset(req.dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")

    # ── Check for cached results (same dataset + config) ──────────────
    cached = team_db.find_matching_run(
        dataset_id=req.dataset_id,
        ml_task=req.ml_task.value,
        target_column=req.target_column,
        feature_columns=req.feature_columns,
        run_type="training",
    )
    if cached:
        cached_run_id = cached["run_id"]
        # Verify that persisted results still exist on disk/Azure
        storage = storage_service.get_storage()
        persisted = storage.get_training_result(cached_run_id)
        if persisted:
            logger.info(
                f"Cache hit! Returning previous run {cached_run_id} "
                f"(dataset={ds.filename}, target={req.target_column}, task={req.ml_task.value})"
            )
            return TrainingStartResponse(
                run_id=cached_run_id,
                status=TrainingStatus.COMPLETE,
                message=(
                    f"Results loaded from previous run! "
                    f"Best model: {cached.get('best_algorithm', '?')} "
                    f"({cached.get('primary_metric', '?')}: "
                    f"{cached.get('best_metric_value', 0):.4f}). "
                    f"Trained {cached.get('model_count', '?')} models on {cached.get('created_at', '?')}."
                ),
            )

    # ── No cache hit — start new training ──────────────────────────────
    run_id = str(uuid.uuid4())[:8]
    _active_runs[run_id] = {
        "status": TrainingStatus.QUEUED,
        "progress": 0,
        "stage": "queued",
        "request": req,
        "dataset": ds,
        "logs": [],
        "result": None,
        "aml": None,
        "frame": None,
        "started_at": datetime.now().isoformat(),
    }

    team_db.save_training_run(run_id, req, TrainingStatus.QUEUED)
    asyncio.create_task(_run_training(run_id))

    return TrainingStartResponse(
        run_id=run_id,
        status=TrainingStatus.QUEUED,
        message="Training queued successfully",
    )


KNOWN_ALGOS = ['StackedEnsemble', 'DeepLearning', 'XGBoost', 'GBM', 'GLM', 'DRF', 'XRT']


def _extract_algo(mid: str) -> str:
    for algo in KNOWN_ALGOS:
        if mid.startswith(algo):
            return algo
    return mid.split("_")[0] if "_" in mid else mid


async def _run_training(run_id: str):


    run = _active_runs.get(run_id)
    if not run:
        return

    req: TrainingStartRequest = run["request"]
    ds: DatasetMetadata = run["dataset"]

    try:
        await _update_run(run_id, TrainingStatus.QUEUED, 5, "Initializing H2O...")

        if not h2o_engine.init_h2o():
            await _update_run(run_id, TrainingStatus.FAILED, 0, "H2O initialization failed. Ensure Java 17+ is installed.")
            return

        await _update_run(run_id, TrainingStatus.DATA_CHECK, 10, "Loading dataset...")

        storage = storage_service.get_storage()
        filepath = storage.get_dataset_path(req.dataset_id, ds.filename)
        frame = h2o_engine.load_dataset(str(filepath))
        run["frame"] = frame
        train_frame, test_frame = h2o_engine.split_train_holdout(frame, req.train_test_split, seed=42)
        run["train_frame"] = train_frame
        run["test_frame"] = test_frame

        await _update_run(
            run_id,
            TrainingStatus.FEATURES,
            20,
            f"Loaded {frame.nrows} rows, {frame.ncols} columns — "
            f"train {train_frame.nrows} ({float(req.train_test_split) * 100:.0f}%) / "
            f"holdout test {test_frame.nrows} ({(1 - float(req.train_test_split)) * 100:.0f}%)",
        )

        models_list = [m.value for m in req.models] if req.models else None

        loop = asyncio.get_event_loop()
        aml = await loop.run_in_executor(
            None,
            lambda: h2o_engine.setup_automl(
                frame=train_frame,
                target=req.target_column,
                ml_task=req.ml_task.value,
                include_algos=models_list,
                max_models=req.max_models,
                max_runtime_secs=req.max_runtime_secs,
                nfolds=req.nfolds,
                seed=42,
            )
        )
        run["aml"] = aml

        await _update_run(run_id, TrainingStatus.TRAINING, 30, "Starting AutoML training...")

        # Surface runtime capability constraints early so users know why some algos are skipped.
        try:
            if not h2o_engine.is_xgboost_available():
                await _update_run(
                    run_id,
                    TrainingStatus.TRAINING,
                    31,
                    "XGBoost is not available in this local H2O runtime; continuing with GBM/DRF/GLM/DeepLearning.",
                )
        except Exception:
            pass

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            h2o_engine.train_automl, aml, req.feature_columns, req.target_column, train_frame
        )

        heartbeat_progress = 31
        heartbeat_step = 2
        heartbeat_chars = ["|", "/", "-", "\\"]
        heartbeat_i = 0
        started = datetime.now()

        # H2O AutoML train() is blocking, so stream synthetic heartbeat logs to reassure users
        # that training is actively running in real-time.
        while not future.done():
            await asyncio.sleep(3)
            elapsed_s = int((datetime.now() - started).total_seconds())
            heartbeat_i += 1
            heartbeat_progress = min(heartbeat_progress + heartbeat_step, 74)
            await _update_run(
                run_id,
                TrainingStatus.TRAINING,
                heartbeat_progress,
                f"AutoML training in progress {heartbeat_chars[heartbeat_i % len(heartbeat_chars)]} ({elapsed_s}s elapsed)...",
            )

        future.result()
        executor.shutdown(wait=False)

        lb_snapshot = h2o_engine.poll_leaderboard(aml)
        if lb_snapshot:
            metric_cols = [k for k in lb_snapshot[0] if k != "model_id"]
            primary_metric = metric_cols[0] if metric_cols else None
            best_val = None
            last_leader = None

            for i, row in enumerate(lb_snapshot):
                model_id = row.get("model_id", "")
                progress = min(30 + (i + 1) * 3, 80)
                await _update_run(
                    run_id, TrainingStatus.TRAINING, progress,
                    f"AutoML: starting {model_id} model training"
                )

                if primary_metric:
                    val = row.get(primary_metric)
                    if val is not None:
                        if best_val is None or val <= best_val:
                            best_val = val
                            if model_id != last_leader:
                                last_leader = model_id
                                await _update_run(
                                    run_id, TrainingStatus.TRAINING, progress,
                                    f"New leader: {model_id}, {primary_metric}: {val}"
                                )

        await _update_run(run_id, TrainingStatus.EVALUATION, 85, "Evaluating models...")

        lb = h2o_engine.get_leaderboard(aml)
        best = h2o_engine.get_best_model(aml)
        varimp = h2o_engine.get_variable_importance(best)

        model_save_dir = str(MODELS_DIR / run_id)
        Path(model_save_dir).mkdir(parents=True, exist_ok=True)
        saved_path = h2o_engine.save_model(best, model_save_dir)

        models_result = []
        for i, row in enumerate(lb):
            model_id = row.get("model_id", f"model_{i}")
            metrics = {k: v for k, v in row.items() if k != "model_id"}
            models_result.append(ModelResult(
                model_id=model_id,
                algorithm=_extract_algo(model_id),
                metrics=metrics,
                rank=i + 1,
                is_best=(i == 0),
            ))

        # Detect number of target classes for binary vs multiclass metric selection
        n_target_classes = 2
        try:
            if req.ml_task.value == "classification" and frame is not None:
                n_target_classes = frame[req.target_column].nlevels()[0]
        except Exception:
            pass

        run["n_target_classes"] = n_target_classes
        run["result"] = {
            "leaderboard": models_result,
            "feature_importance": varimp,
            "best_model_id": best.model_id if best else None,
            "saved_model_path": saved_path,
            "ml_task": req.ml_task.value,
        }

        team_db.update_training_run(run_id, TrainingStatus.COMPLETE)
        await _update_run(run_id, TrainingStatus.COMPLETE, 100, f"Training complete! {len(lb)} models trained.")

    except Exception as e:
        logger.exception(f"Training failed for run {run_id}")
        await _update_run(run_id, TrainingStatus.FAILED, 0, f"Training failed: {str(e)}")
        team_db.update_training_run(run_id, TrainingStatus.FAILED)

    # ── Persist results to Azure/local storage (after try/except so
    #    a persistence failure doesn't mark a successful run as FAILED) ──
    run = _active_runs.get(run_id)
    if run and run.get("result"):
        await _persist_training_result(run_id, run, run["request"], run["dataset"])


def _h2o_selection_metrics_from_best(best, ml_task: str, n_target_classes: int) -> dict:
    """Leaderboard / CV metrics only (H2O AutoML) — never holdout sklearn."""
    if not best:
        return {}
    metrics = best.metrics if isinstance(best, ModelResult) else (best.get("metrics") or {})
    out: dict[str, float] = {}
    if ml_task == "classification":
        pref = ["auc", "logloss"] if n_target_classes <= 2 else ["mean_per_class_error", "logloss", "auc"]
    else:
        pref = ["mean_residual_deviance", "rmse", "mae"]
    for k in pref:
        if k not in metrics or metrics[k] is None:
            continue
        try:
            fv = float(metrics[k])
            if math.isnan(fv) or math.isinf(fv):
                continue
            prec = 8 if k == "logloss" else 6
            out[k] = round(fv, prec)
        except (TypeError, ValueError):
            continue
    if "mean_residual_deviance" in out:
        out["deviance"] = out["mean_residual_deviance"]
    return out


async def _persist_training_result(run_id: str, run: dict, req, ds):
    """Persist full training result to Azure/local storage + enrich DB metadata."""
    try:
        result = run.get("result", {})
        lb = result.get("leaderboard", [])
        best = next((m for m in lb if m.is_best), None) if lb else None

        n_cls = int(run.get("n_target_classes", 2))
        selection_metrics: dict = _h2o_selection_metrics_from_best(best, result.get("ml_task") or "", n_cls)
        validation_metrics: dict = {}
        all_metrics: dict = dict(selection_metrics)
        metrics_eval_set = "train"
        metrics_eval_rows = 0
        aml = run.get("aml")
        frame = run.get("frame")
        train_frame = run.get("train_frame") or frame
        test_frame = run.get("test_frame")
        ml_task = result.get("ml_task") or getattr(getattr(req, "ml_task", None), "value", None) or ""
        confusion_matrix_data = None
        residuals_data = None
        holdout_evaluation = None

        if aml and train_frame:
            try:
                best_model = h2o_engine.get_best_model(aml)
                use_test = (
                    test_frame is not None
                    and int(test_frame.nrows) > 0
                )
                eval_frame = test_frame if use_test else train_frame
                metrics_eval_set = "holdout" if use_test else "train"
                metrics_eval_rows = int(eval_frame.nrows) if eval_frame is not None else 0

                if ml_task == "classification":
                    m, cm_data, cls_rows = h2o_engine.evaluate_classification_frame(
                        best_model,
                        eval_frame,
                        req.target_column,
                        max_detail_rows=10,
                        feature_columns=req.feature_columns,
                    )
                    for k, v in m.items():
                        if v is not None:
                            try:
                                fv = float(v)
                                if not (math.isnan(fv) or math.isinf(fv)):
                                    validation_metrics[k] = fv
                            except (ValueError, TypeError):
                                pass
                    if cm_data and cm_data.get("labels"):
                        confusion_matrix_data = cm_data
                    tr_n = int(train_frame.nrows) if train_frame is not None else 0
                    te_n = int(test_frame.nrows) if test_frame is not None else 0
                    tot = tr_n + te_n
                    holdout_evaluation = {
                        "train_ratio_config": float(req.train_test_split),
                        "test_ratio_config": round(1.0 - float(req.train_test_split), 6),
                        "train_rows": tr_n,
                        "test_rows": te_n,
                        "train_fraction_actual": round(tr_n / tot, 6) if tot else 0.0,
                        "test_fraction_actual": round(te_n / tot, 6) if tot else 0.0,
                        "classification_rows": cls_rows,
                    }
                elif ml_task == "regression":
                    _m, reg_rows_full = h2o_engine.evaluate_regression_frame(
                        best_model,
                        eval_frame,
                        req.target_column,
                        feature_columns=req.feature_columns,
                        max_detail_rows=10,
                    )
                    reg_rows = reg_rows_full
                    act = [r["actual"] for r in reg_rows_full]
                    prd = [r["predicted"] for r in reg_rows_full]
                    residuals_data = {
                        "actual": act,
                        "predicted": prd,
                        "errors": [round(float(act[i]) - float(prd[i]), 8) for i in range(min(len(act), len(prd)))],
                    }
                    tr_n = int(train_frame.nrows) if train_frame is not None else 0
                    te_n = int(test_frame.nrows) if test_frame is not None else 0
                    tot = tr_n + te_n
                    holdout_evaluation = {
                        "train_ratio_config": float(req.train_test_split),
                        "test_ratio_config": round(1.0 - float(req.train_test_split), 6),
                        "train_rows": tr_n,
                        "test_rows": te_n,
                        "train_fraction_actual": round(tr_n / tot, 6) if tot else 0.0,
                        "test_fraction_actual": round(te_n / tot, 6) if tot else 0.0,
                        "regression_rows": reg_rows,
                    }
                else:
                    raw_metrics = h2o_engine.get_model_metrics(
                        best_model, train_frame, ml_task, req.target_column, eval_frame=eval_frame
                    )
                    for k, v in raw_metrics.items():
                        if v is not None:
                            try:
                                fv = float(v)
                                if not (math.isnan(fv) or math.isinf(fv)):
                                    all_metrics[k] = fv
                            except (ValueError, TypeError):
                                pass
            except Exception:
                pass

        if not selection_metrics and best:
            raw_sel = {k: v for k, v in best.metrics.items() if k != "model_id" and v is not None}
            selection_metrics = dict(raw_sel)
            all_metrics = dict(selection_metrics)

        gains_lift_rows: list = []
        gains_lift_note = ""
        if ml_task == "classification" and aml and train_frame is not None:
            try:
                best_model = h2o_engine.get_best_model(aml)
                if h2o_engine.model_is_binomial(best_model):
                    gains_lift_rows = h2o_engine.get_gains_lift(best_model, train_frame)
                    if not gains_lift_rows and frame is not None and int(frame.nrows) > int(train_frame.nrows):
                        gains_lift_rows = h2o_engine.get_gains_lift(best_model, frame)
                        if gains_lift_rows:
                            gains_lift_note = (
                                "Deciles computed on the full imported dataset (before train/holdout split) "
                                "because the post-split training slice was too small for stable deciles."
                            )
                    if not gains_lift_rows:
                        gains_lift_note = (
                            "Gains/lift deciles were empty — common with very small training sets after the train/holdout split."
                        )
                else:
                    gains_lift_note = (
                        "Gains/lift charts are only available for binary classification in H2O. "
                        "This task is multiclass, so decile lift is not shown."
                    )
            except Exception as e:
                gains_lift_note = f"Gains/lift could not be computed: {str(e)[:180]}"

        if ml_task == "classification" and aml and test_frame is not None and int(test_frame.nrows) > 0:
            try:
                bm_csv = h2o_engine.get_best_model(aml)
                out_csv = PROCESSED_DATA_DIR / run_id / "holdout_predictions.csv"
                out_csv.parent.mkdir(parents=True, exist_ok=True)
                h2o_engine.write_classification_holdout_csv(
                    bm_csv,
                    test_frame,
                    req.target_column,
                    list(req.feature_columns or []),
                    str(out_csv),
                )
            except Exception as e:
                logger.debug("holdout csv export: %s", e)

        if ml_task == "regression" and aml and test_frame is not None and int(test_frame.nrows) > 0:
            try:
                bm_reg = h2o_engine.get_best_model(aml)
                out_reg = PROCESSED_DATA_DIR / run_id / "holdout_regression_predictions.csv"
                out_reg.parent.mkdir(parents=True, exist_ok=True)
                h2o_engine.write_regression_holdout_csv(
                    bm_reg,
                    test_frame,
                    req.target_column,
                    list(req.feature_columns or []),
                    str(out_reg),
                )
            except Exception as e:
                logger.debug("holdout regression csv export: %s", e)

        if holdout_evaluation is None:
            try:
                tr_n = int(train_frame.nrows) if train_frame is not None else 0
                te_n = int(test_frame.nrows) if test_frame is not None else 0
                tot = tr_n + te_n
                holdout_evaluation = {
                    "train_ratio_config": float(req.train_test_split),
                    "test_ratio_config": round(1.0 - float(req.train_test_split), 6),
                    "train_rows": tr_n,
                    "test_rows": te_n,
                    "train_fraction_actual": round(tr_n / tot, 6) if tot else 0.0,
                    "test_fraction_actual": round(te_n / tot, 6) if tot else 0.0,
                    "classification_rows": [],
                    "regression_rows": [],
                }
            except Exception:
                pass

        # Build serializable result dict
        serializable_result = {
            "run_id": run_id,
            "ml_task": ml_task,
            "leaderboard": [m.model_dump() if isinstance(m, ModelResult) else m for m in lb],
            "feature_importance": result.get("feature_importance", []),
            "best_model_id": best.model_id if best else "",
            "all_metrics": all_metrics,
            "selection_metrics": selection_metrics,
            "validation_metrics": validation_metrics,
            "metrics_eval_set": metrics_eval_set,
            "metrics_eval_rows": metrics_eval_rows,
            "config": {
                "dataset_id": req.dataset_id,
                "dataset_filename": ds.filename,
                "target_column": req.target_column,
                "feature_columns": req.feature_columns,
                "ml_task": ml_task,
                "max_models": req.max_models,
                "max_runtime_secs": req.max_runtime_secs,
                "nfolds": req.nfolds,
                "train_test_split": float(req.train_test_split),
            },
            "n_target_classes": run.get("n_target_classes", 2),
            "created_at": run.get("started_at", datetime.now().isoformat()),
        }
        if confusion_matrix_data:
            serializable_result["confusion_matrix"] = confusion_matrix_data
        if residuals_data:
            serializable_result["residuals"] = residuals_data
        if holdout_evaluation:
            serializable_result["holdout_evaluation"] = holdout_evaluation
        if ml_task == "classification":
            serializable_result["gains_lift"] = gains_lift_rows
            serializable_result["gains_lift_note"] = gains_lift_note

        # Determine primary metric and its value (needed for DB + blob)
        primary_metric = ""
        best_metric_value = 0.0
        if selection_metrics:
            if ml_task == "classification":
                pref = ["auc", "logloss", "mean_per_class_error"]
            else:
                pref = ["mean_residual_deviance", "deviance", "rmse", "mae"]
            primary_metric = next((m for m in pref if m in selection_metrics), next(iter(selection_metrics), ""))
            best_metric_value = selection_metrics.get(primary_metric, 0.0) or 0.0

        # Enrich SQLite first so history stays correct even if Azure/local blob upload fails
        team_db.update_run_results(
            run_id=run_id,
            ml_task=ml_task,
            target_column=req.target_column,
            feature_columns=req.feature_columns,
            best_model_id=best.model_id if best else "",
            best_algorithm=best.algorithm if best else "",
            primary_metric=primary_metric,
            best_metric_value=best_metric_value,
            model_count=len(lb),
            dataset_filename=ds.filename,
            run_type="training",
        )

        # Merge evaluation artifacts into in-memory result so APIs work before restart
        if run:
            r = run.setdefault("result", {})
            r["all_metrics"] = serializable_result.get("all_metrics", {})
            r["config"] = serializable_result.get("config", {})
            for opt in (
                "confusion_matrix",
                "residuals",
                "holdout_evaluation",
                "gains_lift",
                "gains_lift_note",
                "metrics_eval_set",
                "metrics_eval_rows",
                "selection_metrics",
                "validation_metrics",
            ):
                if opt in serializable_result:
                    r[opt] = serializable_result[opt]

        # Persist JSON + model binary via storage layer (Azure or local)
        storage = storage_service.get_storage()
        storage.save_training_result(run_id, _sanitize_for_json(serializable_result))

        saved_path = result.get("saved_model_path", "")
        if saved_path:
            model_path = Path(saved_path)
            if model_path.exists():
                storage.save_model_binary(run_id, model_path.name, str(model_path))

        logger.info(f"Training result persisted for run {run_id} ({len(lb)} models, metric={primary_metric}:{best_metric_value})")

    except Exception as e:
        logger.warning(f"Failed to persist training result for {run_id}: {e}")


async def _update_run(run_id: str, status: TrainingStatus, progress: int, message: str):
    run = _active_runs.get(run_id)
    if run:
        run["status"] = status
        run["progress"] = progress
        run["stage"] = status.value
        log_entry = {"timestamp": datetime.now().isoformat(), "stage": status.value, "progress": progress, "message": message}
        run["logs"].append(log_entry)

    for ws in _websocket_connections.get(run_id, []):
        try:
            await ws.send_json({
                "status": status.value,
                "progress": progress,
                "stage": status.value,
                "message": message,
                "timestamp": datetime.now().isoformat(),
            })
        except Exception:
            pass


async def get_training_status(run_id: str) -> TrainingStatusResponse:
    run = _active_runs.get(run_id)
    if not run:
        db_run = team_db.get_training_run(run_id)
        if not db_run:
            raise HTTPException(404, "Training run not found")
        return TrainingStatusResponse(
            run_id=run_id,
            status=TrainingStatus(db_run["status"]),
            progress_percent=100 if db_run["status"] == "complete" else 0,
            current_stage=db_run["status"],
            logs=[],
        )
    log_src = run.get("logs") or []
    log_lines = [
        f"[{e.get('progress', run.get('progress', 0))}%] {e.get('message', '')}"
        for e in log_src[-120:]
    ]
    return TrainingStatusResponse(
        run_id=run_id,
        status=run["status"],
        progress_percent=run["progress"],
        current_stage=run["stage"],
        message=log_src[-1]["message"] if log_src else "",
        logs=log_lines,
    )


async def stop_training(run_id: str):
    run = _active_runs.get(run_id)
    if not run:
        raise HTTPException(404, "Training run not found")
    await _update_run(run_id, TrainingStatus.STOPPED, run["progress"], "Training stopped by user")
    team_db.update_training_run(run_id, TrainingStatus.STOPPED)
    return {"message": "Training stopped"}


async def training_websocket(websocket: WebSocket, run_id: str):
    await websocket.accept()
    if run_id not in _websocket_connections:
        _websocket_connections[run_id] = []
    _websocket_connections[run_id].append(websocket)

    try:
        run = _active_runs.get(run_id)
        if run:
            for log in run["logs"]:
                try:
                    await websocket.send_json(log)
                except WebSocketDisconnect:
                    return

        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            except asyncio.TimeoutError:
                run = _active_runs.get(run_id)
                if run and run["status"] in (TrainingStatus.COMPLETE, TrainingStatus.FAILED, TrainingStatus.STOPPED):
                    try:
                        await websocket.send_json({
                            "status": run["status"].value,
                            "progress": run["progress"],
                            "stage": run["stage"],
                            "message": "done",
                            "timestamp": datetime.now().isoformat(),
                        })
                    except WebSocketDisconnect:
                        pass
                    break
            except WebSocketDisconnect:
                break
    finally:
        if run_id in _websocket_connections:
            _websocket_connections[run_id] = [ws for ws in _websocket_connections[run_id] if ws != websocket]
        try:
            await websocket.close()
        except Exception:
            pass


# ── Results ────────────────────────────────────────────────────────────────

def _get_result_or_load(run_id: str) -> tuple[dict | None, dict | None]:
    """
    Fallback chain: in-memory → local/Azure storage.
    Returns (run_dict_or_None, result_dict_or_None).
    When loaded from storage, the result dict has serialized models (plain dicts, not ModelResult).
    """
    run = _active_runs.get(run_id)
    if run and run.get("result"):
        return run, run["result"]

    # Try loading from storage
    storage = storage_service.get_storage()
    persisted = storage.get_training_result(run_id)
    if persisted:
        return None, persisted

    return None, None


def _models_from_persisted(persisted_leaderboard: list[dict]) -> list[ModelResult]:
    """Convert persisted leaderboard dicts back to ModelResult objects."""
    models = []
    for m in persisted_leaderboard:
        if isinstance(m, ModelResult):
            models.append(m)
        else:
            models.append(ModelResult(
                model_id=m.get("model_id", ""),
                algorithm=m.get("algorithm", ""),
                metrics=m.get("metrics", {}),
                rank=m.get("rank", 0),
                is_best=m.get("is_best", False),
            ))
    return models


async def get_leaderboard(run_id: str) -> LeaderboardResponse:
    run, result = _get_result_or_load(run_id)
    if not result:
        raise HTTPException(404, "Results not found. Training may still be in progress.")

    # Normalize leaderboard to ModelResult objects
    lb = result.get("leaderboard", [])
    if lb and isinstance(lb[0], dict):
        models = _models_from_persisted(lb)
    else:
        models = lb

    # H2O default sort metric per Appendix D:
    #   binary classification → AUC
    #   multiclass classification → mean_per_class_error
    #   regression → deviance (mean_residual_deviance)
    metric_keys: list[str] = []
    if models:
        first = models[0]
        metric_keys = list(first.metrics.keys()) if isinstance(first, ModelResult) else list((first.get("metrics") or {}).keys())

    ml_task = result.get("ml_task", "classification")
    _ntc = result.get("n_target_classes")
    target_classes = int(_ntc) if _ntc is not None else int(run.get("n_target_classes", 2) if run else 2)

    if ml_task == "classification":
        if target_classes <= 2:
            pref = ["auc", "logloss", "mean_per_class_error"]
        else:
            pref = ["mean_per_class_error", "logloss", "auc"]
    else:
        pref = ["mean_residual_deviance", "rmse", "mae", "mse", "rmsle", "r2"]

    primary_metric = next((m for m in pref if m in metric_keys), (metric_keys[0] if metric_keys else "score"))

    return LeaderboardResponse(
        run_id=run_id,
        ml_task=ml_task,
        primary_metric=primary_metric,
        models=models,
    )


async def get_best_model(run_id: str):
    run, result = _get_result_or_load(run_id)
    if not result:
        raise HTTPException(404, "Results not found")

    lb = result.get("leaderboard", [])
    models = _models_from_persisted(lb) if lb and isinstance(lb[0], dict) else lb
    best = next((m for m in models if m.is_best), None) if models else None

    target_column = ""
    feature_columns = []
    dataset_id = ""
    if run and run.get("request"):
        req = run["request"]
        target_column = req.target_column
        feature_columns = req.feature_columns
        dataset_id = getattr(req, "dataset_id", "") or ""
    elif result.get("config"):
        target_column = result["config"].get("target_column", "")
        feature_columns = result["config"].get("feature_columns", [])
        dataset_id = str(result["config"].get("dataset_id", "") or "")

    ml_task_res = result.get("ml_task", "")
    n_cls = int(result.get("n_target_classes", 2))

    selection_metrics: dict = dict(result.get("selection_metrics") or {})
    if not selection_metrics and best:
        selection_metrics = _h2o_selection_metrics_from_best(best, ml_task_res, n_cls)
    if not selection_metrics and best:
        selection_metrics = {k: v for k, v in best.metrics.items() if k != "model_id" and v is not None}

    validation_metrics: dict = dict(result.get("validation_metrics") or {})
    if not validation_metrics and ml_task_res == "classification":
        am = result.get("all_metrics") or {}
        for k in ("precision", "recall", "f1"):
            if k in am and am[k] is not None:
                validation_metrics[k] = am[k]

    metrics_eval_set = result.get("metrics_eval_set") or "train"
    metrics_eval_rows = int(result.get("metrics_eval_rows") or 0)
    evaluation_warnings: list[str] = []

    if run:
        aml = run.get("aml")
        train_frame = run.get("train_frame") or run.get("frame")
        test_frame = run.get("test_frame")
        use_test = test_frame is not None and int(test_frame.nrows) > 0
        eval_frame = test_frame if use_test else train_frame
        run_status = run.get("status")
        training_done = run_status == TrainingStatus.COMPLETE
        skip_live_scoring = bool(training_done and aml and eval_frame)

        if skip_live_scoring:
            metrics_eval_set = str(result.get("metrics_eval_set") or metrics_eval_set)
            metrics_eval_rows = int(result.get("metrics_eval_rows") or metrics_eval_rows or 0)
        elif aml and eval_frame and ml_task_res == "classification":
            try:
                best_model = h2o_engine.get_best_model(aml)
                live = h2o_engine.get_model_metrics(
                    best_model,
                    train_frame,
                    ml_task_res,
                    target_column or None,
                    eval_frame=eval_frame,
                    feature_columns=(run.get("request").feature_columns if run.get("request") else None),
                )
                validation_metrics.update(live)
                metrics_eval_set = "holdout" if use_test else "train"
                metrics_eval_rows = int(eval_frame.nrows)
                ok, err = h2o_engine.sample_prediction_check(best_model, eval_frame)
                if not ok:
                    algo = _extract_algo(best_model.model_id)
                    evaluation_warnings.append(
                        f"Sample scoring check failed for {algo} (H2O may log “prediction progress failed”). {err}"
                    )
            except Exception as e:
                evaluation_warnings.append(f"Live validation scoring failed: {str(e)[:200]}")

    def _clean_metric_map(m: dict) -> dict:
        out: dict[str, float] = {}
        for k, v in (m or {}).items():
            if v is None:
                continue
            try:
                fv = float(v)
                if math.isnan(fv) or math.isinf(fv):
                    continue
                prec = 8 if k == "logloss" else 6
                out[k] = round(fv, prec)
            except (ValueError, TypeError):
                continue
        return out

    cleaned_selection = _clean_metric_map(selection_metrics)
    cleaned_validation = _clean_metric_map(validation_metrics)

    if ml_task_res == "classification":
        auc_v = cleaned_selection.get("auc")
        prec_v = cleaned_validation.get("precision")
        if auc_v is not None and prec_v is not None and auc_v >= 0.999 and prec_v < 0.95:
            evaluation_warnings.append(
                "Leaderboard AUC (cross-validation) is near-perfect while holdout macro-precision is lower: "
                "ranking can look excellent even when some rows are still misclassified. Review the confusion matrix "
                "and per-class probabilities on the holdout set."
            )

    feature_importance = result.get("feature_importance", [])

    return {
        "run_id": run_id,
        "best_model": best.model_dump() if best else None,
        "all_metrics": cleaned_selection,
        "selection_metrics": cleaned_selection,
        "validation_metrics": cleaned_validation,
        "feature_importance": feature_importance[:10],
        "ml_task": ml_task_res,
        "target_column": target_column,
        "feature_columns": feature_columns,
        "dataset_id": dataset_id,
        "evaluation_warnings": evaluation_warnings,
        "metrics_eval_set": metrics_eval_set,
        "metrics_eval_rows": metrics_eval_rows,
        "n_target_classes": n_cls,
    }


async def get_feature_importance(run_id: str) -> FeatureImportanceResponse:
    run, result = _get_result_or_load(run_id)
    if not result:
        raise HTTPException(404, "Results not found")

    lb = result.get("leaderboard", [])
    models = _models_from_persisted(lb) if lb and isinstance(lb[0], dict) else lb
    best = next((m for m in models if m.is_best), None) if models else None

    return FeatureImportanceResponse(
        run_id=run_id,
        model_id=best.model_id if best else "unknown",
        features=result.get("feature_importance", [])[:15],
    )


async def get_confusion_matrix(run_id: str) -> ConfusionMatrixResponse:
    run, result = _get_result_or_load(run_id)
    if not result:
        raise HTTPException(404, "Results not found")
    if result.get("ml_task") != "classification":
        raise HTTPException(400, "Confusion matrix only available for classification tasks")

    if "confusion_matrix" in result:
        cm = result["confusion_matrix"]
        labels = cm.get("labels") or []
        matrix = cm.get("matrix") or []
        if labels and matrix:
            return ConfusionMatrixResponse(
                run_id=run_id,
                model_id=result.get("best_model_id", "unknown"),
                labels=labels,
                matrix=matrix,
            )

    # Live H2O (same sklearn CM as training-time evaluation)
    if run:
        aml = run.get("aml")
        frame = run.get("frame")
        train_frame = run.get("train_frame") or frame
        test_frame = run.get("test_frame")
        req = run.get("request")
        tc = (req.target_column if req else None) or result.get("config", {}).get("target_column", "")
        if aml and frame and tc:
            best = h2o_engine.get_best_model(aml)
            eval_fr = test_frame if test_frame is not None and int(test_frame.nrows) > 0 else train_frame
            if eval_fr is not None and int(eval_fr.nrows) > 0:
                _, cm_data, _ = h2o_engine.evaluate_classification_frame(
                    best, eval_fr, str(tc), max_detail_rows=0
                )
                if cm_data and cm_data.get("labels"):
                    return ConfusionMatrixResponse(
                        run_id=run_id,
                        model_id=best.model_id,
                        labels=cm_data["labels"],
                        matrix=cm_data["matrix"],
                    )

    raise HTTPException(404, "Confusion matrix data not available")


def _errors_from_actual_predicted(actual: list, predicted: list) -> list[float]:
    errors = []
    n = min(len(actual), len(predicted))
    for i in range(n):
        try:
            a = float(actual[i])
            p = float(predicted[i])
            errors.append(round(a - p, 8))
        except (TypeError, ValueError):
            errors.append(0.0)
    return errors


async def get_residuals(run_id: str) -> ResidualsResponse:
    run, result = _get_result_or_load(run_id)
    if not result:
        raise HTTPException(404, "Results not found")
    if result.get("ml_task") != "regression":
        raise HTTPException(400, "Residuals only available for regression tasks")

    # Try live H2O (holdout when available, else training slice — not full pre-split frame)
    if run:
        aml = run.get("aml")
        train_frame = run.get("train_frame") or run.get("frame")
        test_frame = run.get("test_frame")
        eval_frame = test_frame if test_frame is not None and int(test_frame.nrows) > 0 else train_frame
        if aml and eval_frame:
            best = h2o_engine.get_best_model(aml)
            preds = best.predict(eval_frame)
            pred_col = preds.as_data_frame().iloc[:, 0].tolist()
            req = run["request"]
            actual_col = eval_frame[req.target_column].as_data_frame().iloc[:, 0].tolist()
            n = min(len(pred_col), len(actual_col), 500)
            act = actual_col[:n]
            prd = pred_col[:n]
            return ResidualsResponse(
                run_id=run_id,
                model_id=best.model_id,
                actual=act,
                predicted=prd,
                errors=_errors_from_actual_predicted(act, prd),
            )

    # Try persisted residuals
    if "residuals" in result:
        res = result["residuals"]
        act = res.get("actual", [])
        prd = res.get("predicted", [])
        err = res.get("errors")
        if not err:
            err = _errors_from_actual_predicted(act, prd)
        return ResidualsResponse(
            run_id=run_id,
            model_id=result.get("best_model_id", "unknown"),
            actual=act,
            predicted=prd,
            errors=err,
        )

    raise HTTPException(404, "Residual data not available")


async def get_holdout_evaluation(run_id: str) -> HoldoutEvaluationResponse:
    _run, result = _get_result_or_load(run_id)
    if not result:
        raise HTTPException(404, "Results not found")

    ho = result.get("holdout_evaluation")
    if not ho:
        raise HTTPException(
            404,
            "Holdout evaluation not available. Run training again with the latest app to populate test-set predictions.",
        )

    cls_rows = [ClassificationHoldoutRow(**r) for r in ho.get("classification_rows") or []]
    reg_rows = [RegressionHoldoutRow(**r) for r in ho.get("regression_rows") or []]

    return HoldoutEvaluationResponse(
        run_id=run_id,
        model_id=result.get("best_model_id", "unknown"),
        train_ratio_config=float(ho.get("train_ratio_config", 0.8)),
        test_ratio_config=float(ho.get("test_ratio_config", 0.2)),
        train_rows=int(ho.get("train_rows", 0)),
        test_rows=int(ho.get("test_rows", 0)),
        train_fraction_actual=float(ho.get("train_fraction_actual", 0.0)),
        test_fraction_actual=float(ho.get("test_fraction_actual", 0.0)),
        classification_rows=cls_rows,
        regression_rows=reg_rows,
    )


async def export_holdout_predictions_csv(run_id: str):
    """Full holdout table: features, actual, predicted, class probabilities (CSV)."""
    run, result = _get_result_or_load(run_id)
    if not result:
        raise HTTPException(404, "Results not found")
    if result.get("ml_task") != "classification":
        raise HTTPException(400, "Holdout predictions export is only available for classification runs.")

    path = PROCESSED_DATA_DIR / run_id / "holdout_predictions.csv"
    if path.is_file():
        return FileResponse(
            path=str(path),
            filename=f"holdout_predictions_{run_id}.csv",
            media_type="text/csv",
        )

    if run and run.get("aml") and run.get("test_frame") is not None and int(run["test_frame"].nrows) > 0:
        try:
            req = run.get("request")
            tc = (getattr(req, "target_column", None) or result.get("config", {}).get("target_column", "")) or ""
            feats = list(
                (getattr(req, "feature_columns", None) or result.get("config", {}).get("feature_columns") or [])
            )
            bm = h2o_engine.get_best_model(run["aml"])
            path.parent.mkdir(parents=True, exist_ok=True)
            if tc and h2o_engine.write_classification_holdout_csv(
                bm, run["test_frame"], str(tc), feats, str(path)
            ) and path.is_file():
                return FileResponse(
                    path=str(path),
                    filename=f"holdout_predictions_{run_id}.csv",
                    media_type="text/csv",
                )
        except Exception as e:
            logger.warning("export_holdout_predictions_csv live rebuild failed: %s", e)

    raise HTTPException(
        404,
        "Holdout predictions CSV not found. Run training again with the latest app to generate the export file.",
    )


async def export_holdout_regression_predictions_csv(run_id: str):
    """Full holdout regression table: features, actual, predicted, error (CSV)."""
    run, result = _get_result_or_load(run_id)
    if not result:
        raise HTTPException(404, "Results not found")
    if result.get("ml_task") != "regression":
        raise HTTPException(400, "Regression holdout export is only available for regression runs.")

    path = PROCESSED_DATA_DIR / run_id / "holdout_regression_predictions.csv"
    if path.is_file():
        return FileResponse(
            path=str(path),
            filename=f"holdout_regression_predictions_{run_id}.csv",
            media_type="text/csv",
        )

    if run and run.get("aml") and run.get("test_frame") is not None and int(run["test_frame"].nrows) > 0:
        try:
            req = run.get("request")
            tc = (getattr(req, "target_column", None) or result.get("config", {}).get("target_column", "")) or ""
            feats = list(
                (getattr(req, "feature_columns", None) or result.get("config", {}).get("feature_columns") or [])
            )
            bm = h2o_engine.get_best_model(run["aml"])
            path.parent.mkdir(parents=True, exist_ok=True)
            if tc and h2o_engine.write_regression_holdout_csv(
                bm, run["test_frame"], str(tc), feats, str(path)
            ) and path.is_file():
                return FileResponse(
                    path=str(path),
                    filename=f"holdout_regression_predictions_{run_id}.csv",
                    media_type="text/csv",
                )
        except Exception as e:
            logger.warning("export_holdout_regression_predictions_csv live rebuild failed: %s", e)

    raise HTTPException(
        404,
        "Regression holdout CSV not found. Run training again with the latest app to generate the export file.",
    )


async def export_results(run_id: str, format: str = "csv"):
    run, result = _get_result_or_load(run_id)
    if not result:
        raise HTTPException(404, "Results not found")

    lb = result.get("leaderboard", [])
    models = _models_from_persisted(lb) if lb and isinstance(lb[0], dict) else lb

    export_dir = PROCESSED_DATA_DIR / run_id
    export_dir.mkdir(parents=True, exist_ok=True)

    if format == "json":
        export_path = export_dir / "results.json"
        data = {
            "run_id": run_id,
            "ml_task": result.get("ml_task", ""),
            "leaderboard": [m.model_dump() if isinstance(m, ModelResult) else m for m in models],
            "feature_importance": result.get("feature_importance", [])[:15],
        }

        def _json_default(obj):
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            return str(obj)

        export_path.write_text(json.dumps(data, indent=2, default=_json_default))
    else:
        import pandas as pd
        export_path = export_dir / "results.csv"
        rows = []
        for m in models:
            if isinstance(m, ModelResult):
                row = {"rank": m.rank, "model_id": m.model_id, "algorithm": m.algorithm, "is_best": m.is_best}
                row.update(m.metrics)
            else:
                row = {"rank": m.get("rank"), "model_id": m.get("model_id"), "algorithm": m.get("algorithm"), "is_best": m.get("is_best")}
                row.update(m.get("metrics", {}))
            rows.append(row)
        pd.DataFrame(rows).to_csv(export_path, index=False)

    return FileResponse(
        path=str(export_path),
        filename=f"automl_results_{run_id}.{format}",
        media_type="application/octet-stream",
    )


# ── Predict ───────────────────────────────────────────────────────────────

async def predict(run_id: str, req: PredictRequest) -> PredictResponse:
    run = _active_runs.get(run_id)
    if not run or not run.get("result"):
        raise HTTPException(404, "Results not found")

    aml = run.get("aml")
    frame = run.get("frame")
    if not aml:
        raise HTTPException(400, "Models not available in memory")

    ml_task = run["result"]["ml_task"]

    loop = asyncio.get_event_loop()
    predictions = await loop.run_in_executor(
        None, lambda: h2o_engine.predict_all_models(aml, req.feature_values, ml_task, frame)
    )

    results = []
    for p in predictions:
        results.append(PredictionModelResult(
            model_id=p["model_id"],
            prediction=str(p.get("prediction", "")),
            class_probabilities=p.get("class_probabilities"),
            error=p.get("error"),
        ))
    return PredictResponse(run_id=run_id, predictions=results)


async def get_random_row(run_id: str) -> dict:
    run = _active_runs.get(run_id)
    if not run or not run.get("result"):
        raise HTTPException(404, "Results not found")

    frame = run.get("frame")
    req = run["request"]
    if not frame:
        raise HTTPException(400, "Data frame not in memory")

    loop = asyncio.get_event_loop()
    row = await loop.run_in_executor(
        None, lambda: h2o_engine.get_random_row(frame, req.target_column, req.feature_columns)
    )
    return {"feature_values": row}


async def get_gains_lift(run_id: str) -> GainsLiftResponse:
    run, result = _get_result_or_load(run_id)
    if not result:
        raise HTTPException(404, "Results not found")

    if result.get("ml_task") != "classification":
        return GainsLiftResponse(
            run_id=run_id,
            rows=[],
            note="Gains/lift applies to classification runs only.",
        )

    persisted_note = (result.get("gains_lift_note") or "").strip()
    if "gains_lift" in result:
        raw_rows = result.get("gains_lift") or []
        rows_out: list[GainsLiftRow] = []
        for r in raw_rows:
            if isinstance(r, dict):
                rows_out.append(GainsLiftRow(**r))
        return GainsLiftResponse(run_id=run_id, rows=rows_out, note=persisted_note)

    if not run:
        return GainsLiftResponse(
            run_id=run_id,
            rows=[],
            note=persisted_note or "Gains/lift was not stored for this run; re-train with the latest app to capture it.",
        )

    aml = run.get("aml")
    train_frame = run.get("train_frame") or run.get("frame")
    full_frame = run.get("frame")
    if not aml or not train_frame:
        return GainsLiftResponse(run_id=run_id, rows=[], note=persisted_note)

    best = h2o_engine.get_best_model(aml)
    if not h2o_engine.model_is_binomial(best):
        return GainsLiftResponse(
            run_id=run_id,
            rows=[],
            note=(
                "Gains/lift charts are only available for binary classification in H2O. "
                "This model is multiclass."
            ),
        )

    loop = asyncio.get_event_loop()
    rows_data = await loop.run_in_executor(None, lambda: h2o_engine.get_gains_lift(best, train_frame))
    note = persisted_note
    if (
        not rows_data
        and full_frame is not None
        and int(full_frame.nrows) > int(train_frame.nrows)
    ):
        rows_data = await loop.run_in_executor(None, lambda: h2o_engine.get_gains_lift(best, full_frame))
        if rows_data and not note:
            note = (
                "Deciles computed on the full imported dataset (before train/holdout split) "
                "because the post-split training slice was too small for stable deciles."
            )
    rows = [GainsLiftRow(**r) for r in rows_data]
    if not rows and not note:
        note = "Gains/lift deciles were empty for this dataset size or model."
    return GainsLiftResponse(run_id=run_id, rows=rows, note=note)


async def generate_ai_summary(run_id: str) -> AISummaryResponse:
    run, result = _get_result_or_load(run_id)
    if not result:
        raise HTTPException(404, "Results not found")

    lb = result.get("leaderboard", [])
    models = _models_from_persisted(lb) if lb and isinstance(lb[0], dict) else lb
    best = next((m for m in models if m.is_best), None) if models else None
    ml_task = result.get("ml_task", "")
    n_cls = int(result.get("n_target_classes", 2))

    tc = ""
    if run and run.get("request"):
        tc = run["request"].target_column
    elif result.get("config"):
        tc = result["config"].get("target_column", "")

    selection_metrics: dict = dict(result.get("selection_metrics") or {})
    if not selection_metrics and best:
        selection_metrics = _h2o_selection_metrics_from_best(best, ml_task, n_cls)
    if not selection_metrics and best:
        selection_metrics = {k: v for k, v in best.metrics.items() if k != "model_id" and v is not None}

    validation_metrics: dict = dict(result.get("validation_metrics") or {})
    if not validation_metrics and ml_task == "classification":
        am = result.get("all_metrics") or {}
        for k in ("precision", "recall", "f1"):
            if k in am and am[k] is not None:
                validation_metrics[k] = am[k]

    metrics_for_ai: dict = {}
    for k, v in selection_metrics.items():
        if v is not None:
            metrics_for_ai[f"model_selection_cv_{k}"] = v
    for k, v in validation_metrics.items():
        if v is not None:
            metrics_for_ai[f"validation_holdout_{k}"] = v

    best_algo = best.algorithm if best else "Unknown"
    best_id = best.model_id if best else "Unknown"
    target = tc

    num_models = len(lb)

    ai = ai_service.get_ai_service()
    try:
        summary = await ai.generate_results_summary(
            best_algo=best_algo,
            best_id=best_id,
            target=target,
            ml_task=ml_task,
            metrics=metrics_for_ai,
            num_models=num_models,
        )
        return summary
    except Exception as e:
        logger.warning(f"AI summary generation failed: {e}")
        return _rule_based_summary(
            best_algo, best_id, target, ml_task, selection_metrics, validation_metrics, num_models
        )


def _rule_based_summary(
    best_algo, best_id, target, ml_task, selection_metrics: dict, validation_metrics: dict, num_models: int
):


    def _safe(v):
        if v is None:
            return None
        try:
            fv = float(v)
            return None if (math.isnan(fv) or math.isinf(fv)) else fv
        except (ValueError, TypeError):
            return None

    auc = _safe(selection_metrics.get("auc"))
    logloss = _safe(selection_metrics.get("logloss"))
    mpce = _safe(selection_metrics.get("mean_per_class_error"))
    rmse = _safe(selection_metrics.get("rmse"))
    r2 = _safe(selection_metrics.get("r2"))
    _dev_raw = selection_metrics.get("mean_residual_deviance")
    if _dev_raw is None:
        _dev_raw = selection_metrics.get("deviance")
    dev = _safe(_dev_raw)
    f1v = _safe(validation_metrics.get("f1"))
    prec = _safe(validation_metrics.get("precision"))
    rec = _safe(validation_metrics.get("recall"))

    if ml_task == "classification":
        cv_bits = []
        if auc is not None:
            cv_bits.append(f"leaderboard AUC (CV) {auc:.4f}")
        if logloss is not None:
            cv_bits.append(f"log loss (CV) {logloss:.4f}")
        if mpce is not None:
            cv_bits.append(f"mean per-class error (CV) {mpce:.4f}")
        cv_desc = ", ".join(cv_bits) if cv_bits else "the best available leaderboard metrics"
        val_bits = []
        if prec is not None and rec is not None and f1v is not None:
            val_bits.append(f"holdout macro precision/recall/F1 {prec:.3f} / {rec:.3f} / {f1v:.3f} (sklearn)")
        val_desc = ("; " + "; ".join(val_bits)) if val_bits else ""
        exec_summary = (
            f"The H2O AutoML run for predicting '{target}' selected {best_id} ({best_algo}) among {num_models} models. "
            f"Model selection (cross-validation): {cv_desc}.{val_desc}"
        )
    else:
        cv_bits = []
        if dev is not None:
            cv_bits.append(f"mean residual deviance (CV) {dev:.4f}")
        if rmse is not None:
            cv_bits.append(f"RMSE (CV) {rmse:.4f}")
        cv_desc = ", ".join(cv_bits) if cv_bits else "the best available leaderboard metrics"
        exec_summary = (
            f"The H2O AutoML run for predicting '{target}' selected {best_id} ({best_algo}) among {num_models} models. "
            f"Model selection (cross-validation): {cv_desc}."
        )

    insights = [
        f"The best model, {best_id}, ranked first on the AutoML leaderboard among {num_models} trained models.",
    ]
    if ml_task == "classification":
        if auc and auc < 0.7:
            insights.append(f"Leaderboard AUC (CV) of {auc:.4f} suggests moderate separability with room for improvement.")
        if f1v is not None:
            insights.append(f"Holdout macro F1 (sklearn) is {f1v:.4f}, consistent with the reported confusion matrix.")
    else:
        if r2 is not None:
            insights.append(f"Leaderboard R² (if reported) is {r2:.4f}; review holdout prediction errors in the results view.")

    recommendations = [
        f"Focus on improving model performance by addressing potential class imbalance or feature engineering.",
        f"Investigate the features contributing most to predictions and consider feature engineering.",
        f"Experiment with hyperparameter tuning for the {best_algo} model to further optimize performance.",
        f"Consider collecting more data or enriching the dataset with additional features.",
    ]

    real_world = (
        f"This model could be used to predict '{target}' based on the provided features. "
        f"For instance, it could assist in automated decision-making workflows, "
        f"enabling targeted interventions to improve outcomes."
    )

    return AISummaryResponse(
        executive_summary=exec_summary,
        key_insights=insights,
        recommendations=recommendations,
        real_world_example=real_world,
        source="rule-based",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Clustering AutoML Pipeline
# ═══════════════════════════════════════════════════════════════════════════


def _append_clustering_run_log(run: dict, progress_pct: int, message: str) -> None:
    """Thread-safe progress, message, and log lines (HTTP status reads `progress` + `logs`)."""
    line = f"[{progress_pct}%] {message}"
    lock = run.get("log_lock")

    def _body() -> None:
        run["progress"] = progress_pct
        run["message"] = message
        lines: list = run.setdefault("log_lines", [])
        lines.append(line)
        if len(lines) > 400:
            del lines[: len(lines) - 300]

    if lock:
        with lock:
            _body()
    else:
        _body()


_clustering_runs: dict[str, dict] = {}
_clustering_ws: dict[str, list[WebSocket]] = {}


async def start_clustering(req: ClusteringStartRequest) -> ClusteringStartResponse:
    ds = team_db.get_dataset(req.dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")

    # ── Check for cached clustering results ───────────────────────────
    cached = team_db.find_matching_run(
        dataset_id=req.dataset_id,
        ml_task="clustering",
        target_column="",
        feature_columns=req.feature_columns,
        run_type="clustering",
    )
    if cached:
        cached_run_id = cached["run_id"]
        storage = storage_service.get_storage()
        persisted = storage.get_clustering_result(cached_run_id)
        if persisted:
            logger.info(
                f"Clustering cache hit! Returning previous run {cached_run_id} "
                f"(dataset={ds.filename}, features={req.feature_columns})"
            )
            return ClusteringStartResponse(
                run_id=cached_run_id,
                status="complete",
                message=(
                    f"Results loaded from previous clustering! "
                    f"Best: {cached.get('best_algorithm', '?')} "
                    f"(score: {cached.get('best_metric_value', 0):.3f}). "
                    f"Tested {cached.get('model_count', '?')} models on {cached.get('created_at', '?')}."
                ),
            )

    # ── No cache hit — start new clustering ────────────────────────────
    run_id = "cl-" + str(uuid.uuid4())[:8]
    _clustering_runs[run_id] = {
        "status": "queued",
        "progress": 0,
        "message": "Queued",
        "log_lines": [],
        "log_lock": threading.Lock(),
        "request": req,
        "dataset": ds,
        "result": None,
        "elbow": None,
    }
    _clustering_ws.setdefault(run_id, [])

    asyncio.create_task(_run_clustering(run_id))
    return ClusteringStartResponse(run_id=run_id, status="queued", message="Clustering queued")


async def _clustering_broadcast(run_id: str, pct: int, msg: str):
    run = _clustering_runs.get(run_id)
    if run:
        _append_clustering_run_log(run, pct, msg)

    payload = json.dumps({
        "type": "progress",
        "progress": pct,
        "message": msg,
        "timestamp": datetime.now().isoformat(),
    })
    dead = []
    for ws in _clustering_ws.get(run_id, []):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _clustering_ws[run_id].remove(ws)


async def _run_clustering(run_id: str):
    import pandas as pd

    run = _clustering_runs.get(run_id)
    if not run:
        return

    req: ClusteringStartRequest = run["request"]
    ds: DatasetMetadata = run["dataset"]

    try:
        run["status"] = "clustering"
        await _clustering_broadcast(run_id, 5, "Loading dataset...")

        storage = storage_service.get_storage()
        filepath = storage.get_dataset_path(req.dataset_id, ds.filename)
        df = pd.read_csv(filepath)

        def progress_cb(pct, msg):
            # Record while the sync pipeline runs in the executor so HTTP polling shows live detail.
            _append_clustering_run_log(run, pct, msg)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: clustering_engine.run_full_pipeline(
                df=df,
                feature_cols=req.feature_columns,
                user_algorithm=req.algorithm,
                user_n_clusters=req.n_clusters,
                user_eps=req.eps,
                user_min_samples=req.min_samples,
                run_stability=req.run_stability_check,
                progress_callback=progress_cb,
            ),
        )

        sanitized = _sanitize_for_json(result)
        run["result"] = sanitized
        run["elbow"] = {
            "data": sanitized["elbow_data"],
            "recommended_k": sanitized["recommended_k"],
        }

        # Save clustered CSV with cluster_label column
        await _clustering_broadcast(run_id, 95, "Saving clustered dataset...")
        try:
            df["cluster_label"] = result["best_labels"]
            clustered_filename = ds.filename.replace(".csv", "_clustered.csv")
            clustered_path = Path(filepath).parent / clustered_filename
            df.to_csv(str(clustered_path), index=False)
            run["clustered_filepath"] = str(clustered_path)
            run["clustered_filename"] = clustered_filename
            logger.info(f"Saved clustered dataset: {clustered_path}")
        except Exception as e:
            logger.warning(f"Failed to save clustered CSV: {e}")

        run["status"] = "complete"
        n_tested = result["total_candidates_tested"]
        best_algo = result["best_algorithm"]
        best_score = result["best_metrics"]["composite_score"]

        # Enrich SQLite first so /training/history stays correct even if blob upload fails (same order as training)
        try:
            team_db.save_clustering_run(run_id, req.dataset_id)
            team_db.update_training_run(run_id, TrainingStatus.COMPLETE)
            team_db.update_run_results(
                run_id=run_id,
                ml_task="clustering",
                target_column="",
                feature_columns=req.feature_columns,
                best_model_id="",
                best_algorithm=best_algo,
                primary_metric="composite_score",
                best_metric_value=best_score,
                model_count=n_tested,
                dataset_filename=ds.filename,
                run_type="clustering",
            )
            logger.info(f"Clustering run metadata saved for {run_id}")
        except Exception as e:
            logger.warning(f"Failed to persist clustering run metadata for {run_id}: {e}")

        try:
            storage = storage_service.get_storage()
            full_clustering_data = {
                "result": run["result"],
                "elbow": run["elbow"],
                "config": {
                    "dataset_id": req.dataset_id,
                    "dataset_filename": ds.filename,
                    "feature_columns": req.feature_columns,
                    "algorithm": req.algorithm,
                    "n_clusters": req.n_clusters,
                },
                "created_at": datetime.now().isoformat(),
            }
            storage.save_clustering_result(run_id, full_clustering_data)

            persist_dir = PROCESSED_DATA_DIR / run_id
            persist_dir.mkdir(parents=True, exist_ok=True)
            with open(persist_dir / "clustering_elbow.json", "w") as f:
                json.dump(run["elbow"], f)

            logger.info(f"Clustering result persisted to storage for {run_id}")
        except Exception as e:
            logger.warning(f"Failed to persist clustering result to storage for {run_id}: {e}")

        await _clustering_broadcast(
            run_id, 100,
            f"Tested {n_tested} models. Best: {best_algo} (score: {best_score:.3f})",
        )

    except Exception as e:
        logger.error(f"Clustering run {run_id} failed: {e}", exc_info=True)
        run["status"] = "failed"
        run["message"] = str(e)
        await _clustering_broadcast(run_id, 0, f"Clustering failed: {e}")


async def get_clustering_status(run_id: str) -> TrainingStatusResponse:
    run = _clustering_runs.get(run_id)
    if not run:
        # Fallback to DB (handles persisted/cached runs after restart)
        db_run = team_db.get_training_run(run_id)
        if not db_run:
            raise HTTPException(404, "Clustering run not found")
        return TrainingStatusResponse(
            run_id=run_id,
            status=TrainingStatus(db_run["status"]),
            progress_percent=100 if db_run["status"] == "complete" else 0,
            current_stage=db_run["status"],
            logs=[],
        )
    try:
        st = TrainingStatus(run["status"])
    except ValueError:
        st = TrainingStatus.TRAINING
    log_lock = run.get("log_lock")
    log_lines = run.get("log_lines") or []
    if log_lock:
        with log_lock:
            tail_logs = list(log_lines[-120:])
    else:
        tail_logs = list(log_lines[-120:])
    return TrainingStatusResponse(
        run_id=run_id,
        status=st,
        progress_percent=run["progress"],
        current_stage=run["status"],
        message=run["message"],
        logs=tail_logs,
    )


def _load_clustering_result_from_disk(run_id: str) -> dict | None:
    """Load persisted clustering result — try storage layer (Azure/local), then raw disk."""
    # Try storage layer (handles Azure + local cache)
    storage = storage_service.get_storage()
    persisted = storage.get_clustering_result(run_id)
    if persisted:
        # The new format wraps result inside a dict with config/elbow
        if "result" in persisted:
            return persisted["result"]
        return persisted

    # Legacy raw disk fallback
    result_file = PROCESSED_DATA_DIR / run_id / "clustering_result.json"
    if result_file.exists():
        try:
            with open(result_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load clustering result from disk: {e}")
    return None


def _load_clustering_elbow_from_disk(run_id: str) -> dict | None:
    """Load persisted elbow analysis from disk as fallback."""
    elbow_file = PROCESSED_DATA_DIR / run_id / "clustering_elbow.json"
    if elbow_file.exists():
        try:
            with open(elbow_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load elbow data from disk: {e}")
    return None


async def get_clustering_result(run_id: str) -> ClusteringResultResponse:
    run = _clustering_runs.get(run_id)
    r = run["result"] if run and run.get("result") else None

    if not r:
        r = _load_clustering_result_from_disk(run_id)

    if not r:
        raise HTTPException(404, "Clustering result not found")

    return ClusteringResultResponse(
        run_id=run_id,
        best_algorithm=r["best_algorithm"],
        best_params=r["best_params"],
        best_metrics=ClusterMetrics(**r["best_metrics"]),
        stability=StabilityResult(**r["stability"]) if r.get("stability") else None,
        cluster_summaries=[ClusterSummary(**s) for s in r["cluster_summaries"]],
        leaderboard=[CandidateModelResult(**m) for m in r["leaderboard"]],
        feature_importance=[
            ClusterFeatureImportance(feature=f["feature"], importance=f["importance"] if f["importance"] is not None else 0.0)
            for f in r["feature_importance"]
        ],
        feature_columns=r["feature_columns"],
        total_candidates_tested=r["total_candidates_tested"],
        pca_points=[DimensionReductionPoint(**p) for p in r["pca_points"]] if r.get("pca_points") else None,
    )


async def get_elbow_analysis(run_id: str) -> ElbowResponse:
    run = _clustering_runs.get(run_id)
    e = run["elbow"] if run and run.get("elbow") else None

    if not e:
        e = _load_clustering_elbow_from_disk(run_id)

    if not e:
        raise HTTPException(404, "Elbow analysis not found")

    return ElbowResponse(
        run_id=run_id,
        data=[ElbowDataPoint(**d) for d in e["data"]],
        recommended_k=e["recommended_k"],
    )


def _get_clustering_bundle(run_id: str) -> dict | None:
    """In-memory clustering run or persisted blob (includes result, config, elbow when stored)."""
    run = _clustering_runs.get(run_id)
    if run and run.get("result"):
        req = run.get("request")
        ds = run.get("dataset")
        cfg: dict = {}
        if req is not None:
            cfg["dataset_id"] = getattr(req, "dataset_id", "")
            cfg["feature_columns"] = list(getattr(req, "feature_columns", []) or [])
        if ds is not None:
            cfg["dataset_filename"] = getattr(ds, "filename", "") or ""
        out: dict = {"result": run["result"], "config": cfg}
        if run.get("elbow"):
            out["elbow"] = run["elbow"]
        return out
    persisted = storage_service.get_storage().get_clustering_result(run_id)
    return persisted


ELBOW_INSIGHT_FALLBACK = (
    "Heuristic K from the elbow chart reflects the best silhouette among KMeans fits for K = 2…10 only. "
    "The leaderboard picks the best model by a combined score across all algorithms (KMeans, GMM, DBSCAN) "
    "and hyperparameters (for example K=6 can beat K=5). The elbow chart is only a KMeans-only guide. "
    "So silhouette can peak at one K here while the global best model uses a different K — that is expected, not a bug."
)


async def get_clustering_elbow_insight(run_id: str) -> TextInsightResponse:
    bundle = _get_clustering_bundle(run_id)
    if not bundle or not bundle.get("result"):
        raise HTTPException(404, "Clustering result not found")
    r = bundle["result"]
    e = bundle.get("elbow") or _load_clustering_elbow_from_disk(run_id) or {}
    recommended_k = int(e.get("recommended_k") or 0)
    best_metrics = r.get("best_metrics") or {}
    best_k = int(best_metrics.get("n_clusters") or 0)
    best_algo = str(r.get("best_algorithm") or "")
    lb = r.get("leaderboard") or []
    top = lb[:4] if isinstance(lb, list) else []
    lb_summary = "; ".join(
        f"{m.get('algorithm')} K={m.get('n_clusters')} score={m.get('composite_score')}"
        for m in top if isinstance(m, dict)
    ) or "n/a"

    if AI_MODE == "azure":
        try:
            ai = ai_service.get_ai_service()
            if hasattr(ai, "clustering_elbow_insight"):
                text = await ai.clustering_elbow_insight(
                    best_algo, best_k, recommended_k, lb_summary,
                )
                if text:
                    return TextInsightResponse(text=text, source="azure")
        except Exception as ex:
            logger.debug("clustering_elbow_insight Azure fallback: %s", ex)

    return TextInsightResponse(text=ELBOW_INSIGHT_FALLBACK, source="rules")


async def get_clustering_labeled_preview(
    run_id: str,
    max_rows: int = 10,
    max_cols: int = 10,
) -> ClusteringLabeledPreviewResponse:
    import pandas as pd

    bundle = _get_clustering_bundle(run_id)
    if not bundle or not bundle.get("result"):
        raise HTTPException(404, "Clustering result not found")
    res = bundle["result"]
    labels = res.get("best_labels")
    if labels is None:
        raise HTTPException(404, "Cluster labels not available for this run")
    cfg = bundle.get("config") or {}
    dataset_id = cfg.get("dataset_id")
    filename = cfg.get("dataset_filename")
    if not dataset_id or not filename:
        raise HTTPException(400, "Cannot resolve dataset for this clustering run")

    storage = storage_service.get_storage()
    filepath = storage.get_dataset_path(dataset_id, filename)
    df = pd.read_csv(filepath)
    label_list = list(labels)
    m = min(len(df), len(label_list))
    if m == 0:
        raise HTTPException(400, "Dataset or labels are empty")
    df = df.iloc[:m].reset_index(drop=True)
    label_list = label_list[:m]

    feat_cols = list(res.get("feature_columns") or [])
    cols_use = [c for c in feat_cols if c in df.columns][:max_cols]
    if not cols_use:
        cols_use = [c for c in df.columns][:max_cols]

    preview = df.loc[:, cols_use].copy()
    preview["cluster_label"] = label_list
    preview = preview.head(max_rows)
    rows = [_sanitize_for_json(row) for row in preview.to_dict(orient="records")]
    return ClusteringLabeledPreviewResponse(
        run_id=run_id,
        columns=list(preview.columns),
        rows=rows,
    )


async def export_clustering_labeled_csv(run_id: str) -> StreamingResponse:
    import io
    import pandas as pd

    bundle = _get_clustering_bundle(run_id)
    if not bundle or not bundle.get("result"):
        raise HTTPException(404, "Clustering result not found")
    res = bundle["result"]
    labels = res.get("best_labels")
    if labels is None:
        raise HTTPException(404, "Cluster labels not available for this run")
    cfg = bundle.get("config") or {}
    dataset_id = cfg.get("dataset_id")
    filename = cfg.get("dataset_filename")
    if not dataset_id or not filename:
        raise HTTPException(400, "Cannot resolve dataset for this clustering run")

    storage = storage_service.get_storage()
    filepath = storage.get_dataset_path(dataset_id, filename)
    df = pd.read_csv(filepath)
    label_list = [int(x) for x in list(labels)]
    m = min(len(df), len(label_list))
    df = df.iloc[:m].reset_index(drop=True)
    label_list = label_list[:m]
    out_df = df.copy()
    out_df["cluster_label"] = label_list

    buf = io.StringIO()
    out_df.to_csv(buf, index=False)
    body = buf.getvalue()
    safe_name = f"clustering_{run_id}_labeled.csv".replace(" ", "_")
    return StreamingResponse(
        iter([body]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}"'},
    )


async def clustering_websocket(websocket: WebSocket, run_id: str):
    await websocket.accept()
    _clustering_ws.setdefault(run_id, []).append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _clustering_ws.get(run_id, []):
            _clustering_ws[run_id].remove(websocket)


# ═══════════════════════════════════════════════════════════════════════════
# Training History & Persistence Endpoints
# ═══════════════════════════════════════════════════════════════════════════

def maybe_prune_training_history_at_startup() -> int:
    """Optional DB cleanup: see config.TRAINING_HISTORY_PRUNE_BEFORE."""
    from .config import TRAINING_HISTORY_PRUNE_BEFORE
    if not TRAINING_HISTORY_PRUNE_BEFORE:
        return 0
    return team_db.prune_training_runs_before(TRAINING_HISTORY_PRUNE_BEFORE)


async def list_training_history(limit: Optional[int] = None) -> TrainingHistoryResponse:
    """Return completed training/clustering runs from the DB (newest first). Pass limit for faster responses."""
    rows = team_db.list_completed_runs(limit=limit)
    runs = []
    for r in rows:
        runs.append(TrainingRunSummary(
            run_id=r.get("run_id", ""),
            dataset_id=r.get("dataset_id", ""),
            dataset_name=r.get("dataset_filename", ""),
            ml_task=r.get("ml_task", ""),
            target_column=r.get("target_column", ""),
            best_model_id=r.get("best_model_id", ""),
            best_algorithm=r.get("best_algorithm", ""),
            primary_metric=r.get("primary_metric", ""),
            best_metric_value=r.get("best_metric_value", 0.0) or 0.0,
            model_count=r.get("model_count", 0) or 0,
            status=r.get("status", "complete"),
            run_type=r.get("run_type", "training"),
            created_at=str(r.get("created_at", "")),
        ))
    return TrainingHistoryResponse(runs=runs)


async def get_dataset_training_history(dataset_id: str) -> TrainingHistoryResponse:
    """Return all completed runs that used a specific dataset."""
    rows = team_db.get_runs_by_dataset(dataset_id)
    runs = []
    for r in rows:
        runs.append(TrainingRunSummary(
            run_id=r.get("run_id", ""),
            dataset_id=r.get("dataset_id", ""),
            dataset_name=r.get("dataset_filename", ""),
            ml_task=r.get("ml_task", ""),
            target_column=r.get("target_column", ""),
            best_model_id=r.get("best_model_id", ""),
            best_algorithm=r.get("best_algorithm", ""),
            primary_metric=r.get("primary_metric", ""),
            best_metric_value=r.get("best_metric_value", 0.0) or 0.0,
            model_count=r.get("model_count", 0) or 0,
            status=r.get("status", "complete"),
            run_type=r.get("run_type", "training"),
            created_at=str(r.get("created_at", "")),
        ))
    return TrainingHistoryResponse(runs=runs)


async def load_persisted_result(run_id: str) -> dict:
    """Load a previously trained result from Azure/local storage into the response."""
    # Already in memory?
    run = _active_runs.get(run_id)
    if run and run.get("result"):
        return {"status": "loaded", "source": "memory", "run_id": run_id}

    # Try storage
    storage = storage_service.get_storage()
    result = storage.get_training_result(run_id)
    if result:
        return {
            "status": "loaded",
            "source": "storage",
            "run_id": run_id,
            "ml_task": result.get("ml_task", ""),
            "model_count": len(result.get("leaderboard", [])),
            "best_model_id": result.get("best_model_id", ""),
        }

    # Try clustering
    cl_result = storage.get_clustering_result(run_id)
    if cl_result:
        return {
            "status": "loaded",
            "source": "storage",
            "run_id": run_id,
            "ml_task": "clustering",
        }

    raise HTTPException(404, "No persisted result found for this run")
