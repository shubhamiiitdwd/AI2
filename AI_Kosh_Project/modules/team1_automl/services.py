"""
Core orchestrator for the AutoML wizard.
All business logic lives here; router.py delegates to these functions.
"""
import uuid
import asyncio
import logging
import json
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
)
from .enums import MLTask, TrainingStatus
from . import data_processor
from . import storage_service
from . import ai_service
from . import h2o_engine
from . import team_db

logger = logging.getLogger(__name__)

_active_runs: dict[str, dict] = {}
_websocket_connections: dict[str, list[WebSocket]] = {}


# ── Dataset Management ─────────────────────────────────────────────────────

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

    suggestions = _generate_usecase_suggestions(columns_info.columns, ds.filename)
    return UseCaseSuggestionsResponse(dataset_id=dataset_id, suggestions=suggestions)


def _generate_usecase_suggestions(columns: list[ColumnInfo], filename: str) -> list[UseCaseSuggestion]:
    suggestions = []
    col_names = [c.name.lower() for c in columns]
    col_map = {c.name.lower(): c for c in columns}

    categorical_cols = [c for c in columns if c.dtype == "object" and 2 <= c.unique_count <= 20]
    numeric_cols = [c for c in columns if c.dtype in ("float64", "int64")]
    binary_cols = [c for c in columns if c.unique_count == 2]

    for col in categorical_cols:
        suggestions.append(UseCaseSuggestion(
            use_case=f"Predict {col.name} based on other features",
            ml_task="classification",
            target_hint=col.name,
        ))

    for col in binary_cols:
        if col not in categorical_cols:
            suggestions.append(UseCaseSuggestion(
                use_case=f"Classify whether {col.name} is true or false",
                ml_task="classification",
                target_hint=col.name,
            ))

    for col in numeric_cols:
        name_lower = col.name.lower()
        if any(kw in name_lower for kw in ("price", "cost", "salary", "revenue", "amount", "value", "sales")):
            suggestions.append(UseCaseSuggestion(
                use_case=f"Predict {col.name} using regression",
                ml_task="regression",
                target_hint=col.name,
            ))

    if numeric_cols and not suggestions:
        target = numeric_cols[-1]
        if target.unique_count > 10:
            suggestions.append(UseCaseSuggestion(
                use_case=f"Predict {target.name} as a continuous value",
                ml_task="regression",
                target_hint=target.name,
            ))

    if any(kw in filename.lower() for kw in ("iris", "flower", "species")):
        species_col = next((c.name for c in columns if "species" in c.name.lower()), None)
        if species_col:
            suggestions.insert(0, UseCaseSuggestion(
                use_case=f"Classify flower species using measurements",
                ml_task="classification",
                target_hint=species_col,
            ))

    if any(kw in filename.lower() for kw in ("hous", "real_estate", "property")):
        price_col = next((c.name for c in columns if "price" in c.name.lower()), None)
        if price_col:
            suggestions.insert(0, UseCaseSuggestion(
                use_case=f"Predict house price based on property features",
                ml_task="regression",
                target_hint=price_col,
            ))

    if any(kw in " ".join(col_names) for kw in ("churn", "cancel", "leave", "attrition")):
        churn_col = next((c.name for c in columns if any(kw in c.name.lower() for kw in ("churn", "cancel", "leave", "attrition"))), None)
        if churn_col:
            suggestions.insert(0, UseCaseSuggestion(
                use_case=f"Predict customer churn / attrition",
                ml_task="classification",
                target_hint=churn_col,
            ))

    if numeric_cols and len(suggestions) < 3:
        for col in numeric_cols:
            if col.unique_count > 10 and not any(s.target_hint == col.name for s in suggestions):
                suggestions.append(UseCaseSuggestion(
                    use_case=f"Predict {col.name} value using regression analysis",
                    ml_task="regression",
                    target_hint=col.name,
                ))
                break

    if categorical_cols and len(suggestions) < 3:
        for col in categorical_cols:
            if not any(s.target_hint == col.name for s in suggestions):
                suggestions.append(UseCaseSuggestion(
                    use_case=f"Build a classifier to determine {col.name}",
                    ml_task="classification",
                    target_hint=col.name,
                ))
                break

    seen = set()
    unique = []
    for s in suggestions:
        key = (s.target_hint, s.ml_task)
        if key not in seen:
            seen.add(key)
            unique.append(s)
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
    import concurrent.futures

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

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            h2o_engine.train_automl, aml, req.feature_columns, req.target_column, frame
        )

        while not future.done():
            await asyncio.sleep(2)

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

async def get_leaderboard(run_id: str) -> LeaderboardResponse:
    run = _active_runs.get(run_id)
    if not run or not run.get("result"):
        raise HTTPException(404, "Results not found. Training may still be in progress.")

    result = run["result"]
    primary_metric = "auc" if result["ml_task"] == "classification" else "rmse"

    return LeaderboardResponse(
        run_id=run_id,
        ml_task=result["ml_task"],
        primary_metric=primary_metric,
        models=result["leaderboard"],
    )


async def get_best_model(run_id: str):
    import math

    run = _active_runs.get(run_id)
    if not run or not run.get("result"):
        raise HTTPException(404, "Results not found")

    result = run["result"]
    best = next((m for m in result["leaderboard"] if m.is_best), None)

    all_metrics = {}
    aml = run.get("aml")
    frame = run.get("frame")
    if aml and frame:
        try:
            best_model = h2o_engine.get_best_model(aml)
            all_metrics = h2o_engine.get_model_metrics(best_model, frame, result["ml_task"])
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

    req = run["request"]
    return {
        "run_id": run_id,
        "best_model": best.model_dump() if best else None,
        "all_metrics": cleaned_metrics,
        "feature_importance": result["feature_importance"][:10],
        "ml_task": result["ml_task"],
        "target_column": req.target_column,
        "feature_columns": req.feature_columns,
    }


async def get_feature_importance(run_id: str) -> FeatureImportanceResponse:
    run = _active_runs.get(run_id)
    if not run or not run.get("result"):
        raise HTTPException(404, "Results not found")

    result = run["result"]
    best = next((m for m in result["leaderboard"] if m.is_best), None)
    return FeatureImportanceResponse(
        run_id=run_id,
        model_id=best.model_id if best else "unknown",
        features=result["feature_importance"][:15],
    )


async def get_confusion_matrix(run_id: str) -> ConfusionMatrixResponse:
    run = _active_runs.get(run_id)
    if not run or not run.get("result"):
        raise HTTPException(404, "Results not found")
    if run["result"]["ml_task"] != "classification":
        raise HTTPException(400, "Confusion matrix only available for classification tasks")

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
    raise HTTPException(404, "Confusion matrix data not available")


async def get_residuals(run_id: str) -> ResidualsResponse:
    run = _active_runs.get(run_id)
    if not run or not run.get("result"):
        raise HTTPException(404, "Results not found")
    if run["result"]["ml_task"] != "regression":
        raise HTTPException(400, "Residuals only available for regression tasks")

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
    raise HTTPException(404, "Residual data not available")


async def export_results(run_id: str, format: str = "csv"):
    run = _active_runs.get(run_id)
    if not run or not run.get("result"):
        raise HTTPException(404, "Results not found")

    result = run["result"]
    export_dir = PROCESSED_DATA_DIR / run_id
    export_dir.mkdir(parents=True, exist_ok=True)

    if format == "json":
        import math

        export_path = export_dir / "results.json"
        data = {
            "run_id": run_id,
            "ml_task": result["ml_task"],
            "leaderboard": [m.model_dump() for m in result["leaderboard"]],
            "feature_importance": result["feature_importance"][:15],
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
        for m in result["leaderboard"]:
            row = {"rank": m.rank, "model_id": m.model_id, "algorithm": m.algorithm, "is_best": m.is_best}
            row.update(m.metrics)
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
    run = _active_runs.get(run_id)
    if not run or not run.get("result"):
        raise HTTPException(404, "Results not found")
    if run["result"]["ml_task"] != "classification":
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
    import math

    run = _active_runs.get(run_id)
    if not run or not run.get("result"):
        raise HTTPException(404, "Results not found")

    result = run["result"]
    req = run["request"]
    best = next((m for m in result["leaderboard"] if m.is_best), None)
    ml_task = result["ml_task"]

    all_metrics = {}
    aml = run.get("aml")
    frame = run.get("frame")
    if aml and frame:
        try:
            best_model = h2o_engine.get_best_model(aml)
            raw_metrics = h2o_engine.get_model_metrics(best_model, frame, ml_task)
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

    best_algo = best.algorithm if best else "Unknown"
    best_id = best.model_id if best else "Unknown"
    target = req.target_column
    num_models = len(result["leaderboard"])
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
    import math

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
    )
