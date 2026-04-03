import logging
import threading
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

_h2o = None
_h2o_lock = threading.Lock()
_h2o_available = None


def _check_h2o():
    global _h2o, _h2o_available
    if _h2o_available is not None:
        return _h2o_available
    try:
        import h2o as _h2o_mod
        _h2o = _h2o_mod
        _h2o_available = True
    except ImportError:
        logger.warning("h2o package not installed. Run: pip install h2o")
        _h2o_available = False
    return _h2o_available


def init_h2o(max_mem_size: str = "2G") -> bool:
    if not _check_h2o():
        return False
    with _h2o_lock:
        try:
            _h2o.init(max_mem_size=max_mem_size, nthreads=-1)
            logger.info("H2O cluster initialized successfully")
            return True
        except Exception as e:
            logger.error(f"H2O init failed (Java 17+ required): {e}")
            return False


def shutdown_h2o():
    if _h2o and _h2o_available:
        try:
            _h2o.cluster().shutdown()
        except Exception:
            pass


def load_dataset(file_path: str):
    if not _h2o:
        raise RuntimeError("H2O not initialized")
    return _h2o.import_file(file_path)


def setup_automl(
    frame,
    target: str,
    ml_task: str,
    include_algos: list[str] = None,
    max_models: int = 20,
    max_runtime_secs: int = 300,
    nfolds: int = 5,
    seed: int = 42,
):
    """Prepare frame types and create AutoML object (does not start training)."""
    from h2o.automl import H2OAutoML

    if ml_task == "classification":
        col_type = frame[target].types[target]
        if col_type == "real":
            frame[target] = frame[target].round().ascharacter().asfactor()
        elif col_type == "int":
            frame[target] = frame[target].asfactor()
        elif col_type == "string" or col_type == "enum":
            frame[target] = frame[target].asfactor()
        else:
            frame[target] = frame[target].ascharacter().asfactor()

    algo_map = {
        "DRF": "DRF", "GLM": "GLM", "XGBoost": "XGBoost",
        "GBM": "GBM", "DeepLearning": "DeepLearning", "StackedEnsemble": "StackedEnsemble",
    }
    algos = [algo_map[a] for a in (include_algos or []) if a in algo_map] or None

    aml = H2OAutoML(
        max_models=max_models,
        max_runtime_secs=max_runtime_secs,
        seed=seed,
        nfolds=nfolds,
        include_algos=algos,
        sort_metric="AUTO",
    )
    return aml


def train_automl(aml, features, target, frame):
    """Blocking call that runs AutoML training. Call from a worker thread."""
    aml.train(x=features, y=target, training_frame=frame)
    return aml


def poll_leaderboard(aml) -> list[dict]:
    """Get a snapshot of the current leaderboard during or after training."""
    import math
    try:
        lb = aml.leaderboard
        if lb is None or lb.nrows == 0:
            return []
        df = lb.as_data_frame()
        records = df.to_dict(orient="records")
        for rec in records:
            for k, v in list(rec.items()):
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    rec[k] = None
        return records
    except Exception:
        return []


def get_event_log(aml) -> list[dict]:
    """Get the AutoML event log after training completes."""
    try:
        el = aml.event_log
        if el is None:
            return []
        df = el.as_data_frame()
        events = []
        for _, row in df.iterrows():
            name = str(row.get("name", ""))
            value = str(row.get("value", ""))
            if name and value:
                events.append({"name": name, "value": value})
        return events
    except Exception:
        return []


def get_training_events_from_leaderboard(aml) -> list[str]:
    """Parse model training events from the leaderboard after training."""
    try:
        lb = aml.leaderboard
        if lb is None or lb.nrows == 0:
            return []
        df = lb.as_data_frame()
        events = []
        metric_cols = [c for c in df.columns if c != "model_id"]
        primary_metric = metric_cols[0] if metric_cols else None

        for i, row in df.iterrows():
            model_id = row["model_id"]
            events.append(f"AutoML: starting {model_id} model training")
            if primary_metric and i == 0:
                val = row[primary_metric]
                import math
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    events.append(f"New leader: {model_id}, {primary_metric}: {val}")

        seen_leaders = set()
        best_val = None
        for i, row in df.iterrows():
            model_id = row["model_id"]
            if primary_metric:
                val = row[primary_metric]
                import math
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    if best_val is None or val <= best_val:
                        best_val = val
                        if model_id not in seen_leaders:
                            seen_leaders.add(model_id)
        return events
    except Exception:
        return []


def run_automl(
    frame,
    target: str,
    features: list[str],
    ml_task: str,
    include_algos: list[str] = None,
    max_models: int = 20,
    max_runtime_secs: int = 300,
    nfolds: int = 5,
    seed: int = 42,
    progress_callback=None,
):
    """Legacy single-call interface (still works)."""
    aml = setup_automl(frame, target, ml_task, include_algos, max_models, max_runtime_secs, nfolds, seed)
    aml.train(x=features, y=target, training_frame=frame)
    return aml


def get_leaderboard(aml, extra_columns: list[str] = None) -> list[dict]:
    import math
    lb = aml.leaderboard
    if extra_columns:
        lb = _h2o.automl.get_leaderboard(aml, extra_columns=extra_columns)
    df = lb.as_data_frame()
    records = df.to_dict(orient="records")
    for rec in records:
        for k, v in list(rec.items()):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                rec[k] = None
    return records


def get_best_model(aml):
    return aml.leader


def get_variable_importance(model) -> list[dict]:
    try:
        varimp = model.varimp(use_pandas=True)
        if varimp is not None and not varimp.empty:
            return varimp.to_dict(orient="records")
    except Exception:
        pass
    return []


def get_confusion_matrix(model, frame) -> Optional[dict]:
    perf = None
    for getter in [
        lambda: model.model_performance(xval=True),
        lambda: model.model_performance(frame),
    ]:
        try:
            perf = getter()
            if perf:
                break
        except Exception:
            continue
    if not perf:
        return None
    try:
        cm = perf.confusion_matrix()
        if cm is not None:
            cm_df = cm.to_list()
            labels = cm.col_header[:-1] if hasattr(cm, 'col_header') else []
            return {"matrix": cm_df, "labels": labels}
    except Exception:
        pass
    return None


def get_model_metrics(model, frame, ml_task: str) -> dict:
    metrics = {}
    try:
        perf = model.model_performance(xval=True)
    except Exception:
        try:
            perf = model.model_performance(frame)
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return _metrics_from_leaderboard(model, ml_task)

    try:
        if ml_task == "classification":
            metrics["accuracy"] = _safe_metric(lambda: 1 - perf.mean_per_class_error())
            metrics["mean_per_class_error"] = _safe_metric(perf.mean_per_class_error)
            metrics["logloss"] = _safe_metric(perf.logloss)
            metrics["rmse"] = _safe_metric(perf.rmse)
            metrics["mse"] = _safe_metric(perf.mse)
            try:
                metrics["auc"] = _safe_metric(perf.auc)
            except Exception:
                pass
        else:
            metrics["rmse"] = _safe_metric(perf.rmse)
            metrics["mae"] = _safe_metric(perf.mae)
            metrics["r2"] = _safe_metric(perf.r2)
            metrics["mse"] = _safe_metric(perf.mse)
            metrics["rmsle"] = _safe_metric(perf.rmsle)
    except Exception as e:
        logger.error(f"Error extracting individual metrics: {e}")

    return {k: v for k, v in metrics.items() if v is not None}


def _metrics_from_leaderboard(model, ml_task: str) -> dict:
    """Fallback: extract metrics from the model's built-in summary."""
    metrics = {}
    try:
        summary = model._model_json.get("output", {}).get("cross_validation_metrics_summary")
        if summary:
            df = summary.as_data_frame()
            for _, row in df.iterrows():
                name = row.iloc[0].lower()
                val = _safe_metric(lambda: float(row.iloc[1]))
                if val is not None:
                    metrics[name] = val
    except Exception:
        pass
    return metrics


def predict_single(model, row_frame, ml_task: str) -> dict:
    """Predict on a pre-built single-row H2O frame."""
    import math
    preds = model.predict(row_frame)
    pred_df = preds.as_data_frame()

    result = {"prediction": str(pred_df.iloc[0, 0])}
    if ml_task == "classification" and pred_df.shape[1] > 1:
        class_probs = {}
        for col in pred_df.columns[1:]:
            val = float(pred_df[col].iloc[0])
            class_probs[str(col)] = round(val, 4) if not math.isnan(val) else 0.0
        result["class_probabilities"] = class_probs
    return result


def predict_all_models(aml, feature_values: dict, ml_task: str, training_frame) -> list[dict]:
    """Run prediction on all leaderboard models using properly typed frame."""
    import pandas as pd

    pdf = pd.DataFrame([feature_values])
    row_frame = _h2o.H2OFrame(pdf)

    for col in row_frame.columns:
        if col in training_frame.columns:
            train_type = training_frame[col].types[col]
            if train_type == "enum":
                row_frame[col] = row_frame[col].ascharacter().asfactor()
            elif train_type == "real":
                row_frame[col] = row_frame[col].asnumeric()
            elif train_type == "int":
                row_frame[col] = row_frame[col].asnumeric()

    results = []
    lb = aml.leaderboard.as_data_frame()
    for model_id in lb["model_id"].tolist():
        try:
            model = _h2o.get_model(model_id)
            pred = predict_single(model, row_frame, ml_task)
            results.append({"model_id": model_id, **pred})
        except Exception as e:
            err_msg = str(e)
            if len(err_msg) > 150:
                err_msg = err_msg[:150] + "..."
            results.append({"model_id": model_id, "prediction": None, "error": err_msg})
    return results


def get_gains_lift(model, frame) -> list[dict]:
    """Extract gains/lift table from model performance (tries xval first)."""
    perf = None
    for getter in [
        lambda: model.model_performance(xval=True),
        lambda: model.model_performance(frame),
    ]:
        try:
            perf = getter()
            if perf:
                break
        except Exception:
            continue

    if not perf:
        return []

    try:
        gl = perf.gains_lift()
        if gl is None:
            return []
        gl_df = gl.as_data_frame() if hasattr(gl, 'as_data_frame') else gl
        rows = []
        for i, row in gl_df.iterrows():
            rows.append({
                "group": int(row.get("group", i + 1)) if "group" in row else i + 1,
                "cumulative_data_pct": round(float(row.get("cumulative_data_fraction", 0)) * 100, 1),
                "lift": round(float(row.get("cumulative_lift", 0)), 2),
                "gain_pct": round(float(row.get("cumulative_gain", 0)) * 100, 1),
            })
        return rows
    except Exception as e:
        logger.debug(f"Gains/lift not available: {e}")
    return []


def get_random_row(frame, target: str, features: list[str]) -> dict:
    """Get a random row from the frame as feature values."""
    import random
    df = frame.as_data_frame()
    row = df[features].iloc[random.randint(0, len(df) - 1)]
    return {col: row[col] for col in features}


def save_model(model, path: str) -> str:
    return _h2o.save_model(model=model, path=path, force=True)


def load_model(path: str):
    return _h2o.load_model(path)


def _safe_metric(fn):
    import math
    try:
        val = fn() if callable(fn) else fn
        if val is None:
            return None
        fval = float(val)
        if math.isnan(fval) or math.isinf(fval):
            return None
        return round(fval, 6)
    except Exception:
        return None
