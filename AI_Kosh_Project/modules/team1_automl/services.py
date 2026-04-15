"""
Core orchestrator for the AutoML wizard.
All business logic lives here; router.py delegates to these functions.
"""
import uuid
import re
import math
import asyncio
import logging
import json
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse

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
)
from .enums import MLTask, TrainingStatus
from . import data_processor
from . import storage_service
from . import ai_service
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


async def upload_dataset(file: UploadFile) -> DatasetMetadata:
    content = await file.read()
    filename = file.filename or "dataset.csv"

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
    team_db.save_dataset(meta)
    return meta


async def list_datasets() -> list[DatasetMetadata]:
    return team_db.list_datasets()


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
    sample_rows = pd.read_csv(filepath, nrows=15).fillna("").to_dict(orient="records")

    ai = ai_service.get_ai_service()
    try:
        suggestions = await ai.suggest_usecases(columns_info.columns, sample_rows, ds.filename)
        return UseCaseSuggestionsResponse(dataset_id=dataset_id, suggestions=suggestions)
    except Exception as exc:
        logger.warning("AI use-case suggestion failed, falling back to built-in rules: %s", exc)

    suggestions = _generate_usecase_suggestions(columns_info.columns, ds.filename)
    return UseCaseSuggestionsResponse(dataset_id=dataset_id, suggestions=suggestions)


async def auto_detect_task(dataset_id: str) -> AutoDetectTaskResponse:
    ds = team_db.get_dataset(dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")
    storage = storage_service.get_storage()
    filepath = storage.get_dataset_path(dataset_id, ds.filename)
    columns_info = data_processor.get_columns(filepath, dataset_id)

    import pandas as pd
    sample_rows = pd.read_csv(filepath, nrows=15).fillna("").to_dict(orient="records")

    ai = ai_service.get_ai_service()
    try:
        return await ai.auto_detect_task(columns_info.columns, sample_rows, ds.filename)
    except Exception as exc:
        logger.warning("AI auto-detect failed, using rule-based: %s", exc)

    from . import ai_huggingface
    return await ai_huggingface.auto_detect_task(columns_info.columns, sample_rows, ds.filename)


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

        await _update_run(run_id, TrainingStatus.FEATURES, 20, f"Loaded {frame.nrows} rows, {frame.ncols} columns")

        models_list = [m.value for m in req.models] if req.models else None

        loop = asyncio.get_event_loop()
        aml = await loop.run_in_executor(
            None,
            lambda: h2o_engine.setup_automl(
                frame=frame,
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
            h2o_engine.train_automl, aml, req.feature_columns, req.target_column, frame
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


async def _persist_training_result(run_id: str, run: dict, req, ds):
    """Persist full training result to Azure/local storage + enrich DB metadata."""
    try:
        result = run.get("result", {})
        lb = result.get("leaderboard", [])
        best = next((m for m in lb if m.is_best), None) if lb else None

        # Gather all_metrics for persistence
        all_metrics = {}
        aml = run.get("aml")
        frame = run.get("frame")
        ml_task = result.get("ml_task") or getattr(getattr(req, "ml_task", None), "value", None) or ""
        if aml and frame:
            try:
                best_model = h2o_engine.get_best_model(aml)
                raw_metrics = h2o_engine.get_model_metrics(best_model, frame, ml_task, req.target_column)
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

        if not all_metrics and best:
            all_metrics = {k: v for k, v in best.metrics.items() if k != "model_id" and v is not None}

        # Gather confusion matrix for classification tasks
        confusion_matrix_data = None
        if ml_task == "classification" and aml and frame:
            try:
                best_model = h2o_engine.get_best_model(aml)
                cm_data = h2o_engine.get_confusion_matrix(best_model, frame)
                if cm_data:
                    confusion_matrix_data = cm_data
            except Exception:
                pass

        # Gather residuals for regression tasks
        residuals_data = None
        if ml_task == "regression" and aml and frame:
            try:
                best_model = h2o_engine.get_best_model(aml)
                preds = best_model.predict(frame)
                pred_col = preds.as_data_frame().iloc[:, 0].tolist()
                actual_col = frame[req.target_column].as_data_frame().iloc[:, 0].tolist()
                residuals_data = {
                    "actual": actual_col[:500],
                    "predicted": pred_col[:500],
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
            "config": {
                "dataset_id": req.dataset_id,
                "dataset_filename": ds.filename,
                "target_column": req.target_column,
                "feature_columns": req.feature_columns,
                "ml_task": ml_task,
                "max_models": req.max_models,
                "max_runtime_secs": req.max_runtime_secs,
                "nfolds": req.nfolds,
            },
            "n_target_classes": run.get("n_target_classes", 2),
            "created_at": run.get("started_at", datetime.now().isoformat()),
        }
        if confusion_matrix_data:
            serializable_result["confusion_matrix"] = confusion_matrix_data
        if residuals_data:
            serializable_result["residuals"] = residuals_data

        # Determine primary metric and its value (needed for DB + blob)
        primary_metric = ""
        best_metric_value = 0.0
        if all_metrics:
            if ml_task == "classification":
                pref = ["auc", "accuracy", "logloss", "mean_per_class_error"]
            else:
                pref = ["rmse", "mae", "r2", "mean_residual_deviance"]
            primary_metric = next((m for m in pref if m in all_metrics), next(iter(all_metrics), ""))
            best_metric_value = all_metrics.get(primary_metric, 0.0) or 0.0

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
        )
    return TrainingStatusResponse(
        run_id=run_id,
        status=run["status"],
        progress_percent=run["progress"],
        current_stage=run["stage"],
        message=run["logs"][-1]["message"] if run["logs"] else "",
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
                await websocket.send_json(log)

        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            except asyncio.TimeoutError:
                run = _active_runs.get(run_id)
                if run and run["status"] in (TrainingStatus.COMPLETE, TrainingStatus.FAILED, TrainingStatus.STOPPED):
                    await websocket.send_json({
                        "status": run["status"].value,
                        "progress": run["progress"],
                        "stage": run["stage"],
                        "message": "done",
                        "timestamp": datetime.now().isoformat(),
                    })
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
    target_classes = run.get("n_target_classes", 2) if run else 2

    if ml_task == "classification":
        if target_classes <= 2:
            pref = ["auc", "logloss", "mean_per_class_error", "accuracy"]
        else:
            pref = ["mean_per_class_error", "logloss", "auc", "accuracy"]
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
    if run and run.get("request"):
        req = run["request"]
        target_column = req.target_column
        feature_columns = req.feature_columns
    elif result.get("config"):
        target_column = result["config"].get("target_column", "")
        feature_columns = result["config"].get("feature_columns", [])

    # Persisted metrics first, then overlay live H2O (so PRF from storage is kept if live omits them)
    all_metrics: dict = dict(result.get("all_metrics") or {})
    if run:
        aml = run.get("aml")
        frame = run.get("frame")
        if aml and frame:
            try:
                best_model = h2o_engine.get_best_model(aml)
                live = h2o_engine.get_model_metrics(
                    best_model, frame, result.get("ml_task", ""), target_column or None
                )
                all_metrics.update(live)
            except Exception:
                pass

    if not all_metrics and best:
        all_metrics = {k: v for k, v in best.metrics.items() if k != "model_id"}

    cleaned_metrics = {}
    for k, v in all_metrics.items():
        if v is None:
            continue
        try:
            fv = float(v)
            if math.isnan(fv) or math.isinf(fv):
                continue
            cleaned_metrics[k] = round(fv, 6)
        except (ValueError, TypeError):
            continue

    feature_importance = result.get("feature_importance", [])

    return {
        "run_id": run_id,
        "best_model": best.model_dump() if best else None,
        "all_metrics": cleaned_metrics,
        "feature_importance": feature_importance[:10],
        "ml_task": result.get("ml_task", ""),
        "target_column": target_column,
        "feature_columns": feature_columns,
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

    # Try live H2O (only works if the run is still in memory)
    if run:
        aml = run.get("aml")
        frame = run.get("frame")
        if aml and frame:
            best = h2o_engine.get_best_model(aml)
            cm_data = h2o_engine.get_confusion_matrix(best, frame)
            if cm_data:
                return ConfusionMatrixResponse(
                    run_id=run_id,
                    model_id=best.model_id,
                    labels=cm_data["labels"],
                    matrix=cm_data["matrix"],
                )

    # Try persisted confusion matrix
    if "confusion_matrix" in result:
        cm = result["confusion_matrix"]
        return ConfusionMatrixResponse(
            run_id=run_id,
            model_id=result.get("best_model_id", "unknown"),
            labels=cm.get("labels", []),
            matrix=cm.get("matrix", []),
        )

    raise HTTPException(404, "Confusion matrix data not available")


async def get_residuals(run_id: str) -> ResidualsResponse:
    run, result = _get_result_or_load(run_id)
    if not result:
        raise HTTPException(404, "Results not found")
    if result.get("ml_task") != "regression":
        raise HTTPException(400, "Residuals only available for regression tasks")

    # Try live H2O
    if run:
        aml = run.get("aml")
        frame = run.get("frame")
        if aml and frame:
            best = h2o_engine.get_best_model(aml)
            preds = best.predict(frame)
            pred_col = preds.as_data_frame().iloc[:, 0].tolist()
            req = run["request"]
            actual_col = frame[req.target_column].as_data_frame().iloc[:, 0].tolist()
            return ResidualsResponse(
                run_id=run_id,
                model_id=best.model_id,
                actual=actual_col[:500],
                predicted=pred_col[:500],
            )

    # Try persisted residuals
    if "residuals" in result:
        res = result["residuals"]
        return ResidualsResponse(
            run_id=run_id,
            model_id=result.get("best_model_id", "unknown"),
            actual=res.get("actual", []),
            predicted=res.get("predicted", []),
        )

    raise HTTPException(404, "Residual data not available")


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
        return GainsLiftResponse(run_id=run_id, rows=[])

    # Gains/lift needs live H2O AutoML + frame; absent after restart or storage-only load.
    if not run:
        return GainsLiftResponse(run_id=run_id, rows=[])

    aml = run.get("aml")
    frame = run.get("frame")
    if not aml or not frame:
        return GainsLiftResponse(run_id=run_id, rows=[])

    best = h2o_engine.get_best_model(aml)
    loop = asyncio.get_event_loop()
    rows_data = await loop.run_in_executor(None, lambda: h2o_engine.get_gains_lift(best, frame))
    rows = [GainsLiftRow(**r) for r in rows_data]
    return GainsLiftResponse(run_id=run_id, rows=rows)


async def generate_ai_summary(run_id: str) -> AISummaryResponse:
    run, result = _get_result_or_load(run_id)
    if not result:
        raise HTTPException(404, "Results not found")

    lb = result.get("leaderboard", [])
    models = _models_from_persisted(lb) if lb and isinstance(lb[0], dict) else lb
    best = next((m for m in models if m.is_best), None) if models else None
    ml_task = result.get("ml_task", "")

    tc = ""
    if run and run.get("request"):
        tc = run["request"].target_column
    elif result.get("config"):
        tc = result["config"].get("target_column", "")

    all_metrics = {}
    # Try live H2O
    if run:
        aml = run.get("aml")
        frame = run.get("frame")
        if aml and frame:
            try:
                best_model = h2o_engine.get_best_model(aml)
                raw_metrics = h2o_engine.get_model_metrics(best_model, frame, ml_task, tc or None)
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

    if not all_metrics and best:
        for k, v in best.metrics.items():
            if k != "model_id" and v is not None:
                try:
                    fv = float(v)
                    if not (math.isnan(fv) or math.isinf(fv)):
                        all_metrics[k] = fv
                except (ValueError, TypeError):
                    pass

    # Also try persisted all_metrics
    if not all_metrics and "all_metrics" in result:
        all_metrics = result["all_metrics"]

    best_algo = best.algorithm if best else "Unknown"
    best_id = best.model_id if best else "Unknown"
    target = tc

    num_models = len(lb)
    metrics_str = ", ".join(f"{k}: {v}" for k, v in (all_metrics or {}).items() if v is not None)

    ai = ai_service.get_ai_service()
    try:
        summary = await ai.generate_results_summary(
            best_algo=best_algo,
            best_id=best_id,
            target=target,
            ml_task=ml_task,
            metrics=all_metrics,
            num_models=num_models,
        )
        return summary
    except Exception as e:
        logger.warning(f"AI summary generation failed: {e}")
        return _rule_based_summary(best_algo, best_id, target, ml_task, all_metrics, num_models)


def _rule_based_summary(best_algo, best_id, target, ml_task, metrics, num_models):


    def _safe(v):
        if v is None:
            return None
        try:
            fv = float(v)
            return None if (math.isnan(fv) or math.isinf(fv)) else fv
        except (ValueError, TypeError):
            return None

    auc = _safe(metrics.get("auc"))
    acc = _safe(metrics.get("accuracy"))
    rmse = _safe(metrics.get("rmse"))
    r2 = _safe(metrics.get("r2"))

    if ml_task == "classification":
        perf_desc = f"an AUC of {auc:.4f}" if auc else "the best available metrics"
        acc_desc = f"and an accuracy of {acc*100:.1f}%" if acc else ""
        exec_summary = (
            f"The H2O AutoML results indicate that the best-performing model for the "
            f"{ml_task} task of predicting '{target}' is {best_id}, "
            f"achieving {perf_desc} {acc_desc}."
        )
    else:
        perf_desc = f"an RMSE of {rmse:.4f}" if rmse else "the best available metrics"
        r2_desc = f"and R² of {r2:.4f}" if r2 else ""
        exec_summary = (
            f"The H2O AutoML results indicate that the best-performing model for the "
            f"{ml_task} task of predicting '{target}' is {best_id}, "
            f"achieving {perf_desc} {r2_desc}."
        )

    insights = [
        f"The best model, {best_id}, achieved the highest performance among {num_models} trained models.",
    ]
    if ml_task == "classification":
        if acc and acc < 0.75:
            insights.append(f"The accuracy of {acc*100:.1f}% is reasonable but suggests potential for further optimization.")
        elif acc and acc >= 0.75:
            insights.append(f"The accuracy of {acc*100:.1f}% indicates strong predictive performance.")
        if auc and auc < 0.7:
            insights.append(f"The AUC of {auc:.4f} suggests moderate discriminatory power with room for improvement.")
    else:
        if r2 is not None:
            insights.append(f"The R² score of {r2:.4f} explains {r2*100:.1f}% of variance in the target variable.")

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
        run["progress"] = pct
        run["message"] = msg

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

        progress_events: list[tuple[int, str]] = []

        def progress_cb(pct, msg):
            progress_events.append((pct, msg))

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

        for pct, msg in progress_events:
            await _clustering_broadcast(run_id, pct, msg)

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
        )
    try:
        st = TrainingStatus(run["status"])
    except ValueError:
        st = TrainingStatus.TRAINING
    return TrainingStatusResponse(
        run_id=run_id,
        status=st,
        progress_percent=run["progress"],
        current_stage=run["status"],
        message=run["message"],
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


async def list_training_history() -> TrainingHistoryResponse:
    """Return all completed training/clustering runs from the DB."""
    rows = team_db.list_completed_runs()
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
