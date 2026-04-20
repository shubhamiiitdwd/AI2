import logging
import math
import random
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


def is_xgboost_available() -> bool:
    """Check whether XGBoost extension is available in current H2O runtime."""
    if not _h2o or not _h2o_available:
        return False
    try:
        cluster = _h2o.cluster()
        if hasattr(cluster, "list_all_extensions"):
            exts = cluster.list_all_extensions() or []
            for ext in exts:
                name = str(ext.get("name", "")).lower()
                enabled = bool(ext.get("enabled", False))
                if "xgboost" in name:
                    return enabled
        # If extension list is unavailable, do not assume availability.
        return False
    except Exception:
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


def split_train_holdout(frame, train_ratio: float, seed: int = 42):
    """Split into training and holdout/test frames. train_ratio is the fraction kept for training (e.g. 0.8 = 80%)."""
    if not _h2o or frame is None:
        raise RuntimeError("H2O not initialized")
    n = int(frame.nrows)
    if n < 2:
        return frame, frame
    r = max(0.05, min(0.95, float(train_ratio)))
    parts = frame.split_frame(ratios=[r], seed=seed)
    train_f = parts[0]
    test_f = parts[1]
    try:
        if test_f.nrows == 0 and n >= 2:
            parts = frame.split_frame(ratios=[0.9], seed=seed)
            train_f, test_f = parts[0], parts[1]
    except Exception:
        pass
    return train_f, test_f


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
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    events.append(f"New leader: {model_id}, {primary_metric}: {val}")

        seen_leaders = set()
        best_val = None
        for i, row in df.iterrows():
            model_id = row["model_id"]
            if primary_metric:
                val = row[primary_metric]
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


def get_confusion_matrix_on_frame(model, frame) -> Optional[dict]:
    """Confusion matrix for a specific frame (e.g. holdout test set), not cross-validation."""
    if frame is None:
        return None
    try:
        perf = model.model_performance(frame)
        if not perf:
            return None
        cm = perf.confusion_matrix()
        if cm is not None:
            cm_df = cm.to_list()
            labels = cm.col_header[:-1] if hasattr(cm, "col_header") else []
            return {"matrix": cm_df, "labels": labels}
    except Exception:
        pass
    return None


def build_classification_holdout_rows(model, test_frame, target: str, max_rows: int = 200) -> list[dict]:
    """Per-row actual, predicted label, and confidence (max predicted class probability)."""
    if test_frame is None or int(test_frame.nrows) == 0:
        return []
    try:
        preds = model.predict(test_frame)
        pdf = preds.as_data_frame()
        pred_col = "predict" if "predict" in pdf.columns else pdf.columns[0]
        prob_cols = [c for c in pdf.columns if c != pred_col]
        n = min(int(test_frame.nrows), len(pdf), max_rows)
        actual_series = test_frame[target].as_data_frame().iloc[:, 0].astype(str)
        rows = []
        for i in range(n):
            conf = 1.0
            if prob_cols:
                vals = []
                for c in prob_cols:
                    try:
                        v = float(pdf.iloc[i][c])
                        if not (math.isnan(v) or math.isinf(v)):
                            vals.append(v)
                    except (TypeError, ValueError):
                        continue
                if vals:
                    conf = max(vals)
            rows.append({
                "actual": str(actual_series.iloc[i]),
                "predicted": str(pdf.iloc[i][pred_col]),
                "confidence": round(float(conf), 6),
            })
        return rows
    except Exception as e:
        logger.debug("build_classification_holdout_rows: %s", e)
        return []


def build_regression_holdout_rows(model, test_frame, target: str, max_rows: int = 200) -> list[dict]:
    """Per-row actual, predicted, and error (predicted - actual)."""
    if test_frame is None or int(test_frame.nrows) == 0:
        return []
    try:
        preds = model.predict(test_frame)
        pred_vals = preds.as_data_frame().iloc[:, 0].tolist()
        actual_vals = test_frame[target].as_data_frame().iloc[:, 0].tolist()
        n = min(len(pred_vals), len(actual_vals), max_rows)
        rows = []
        for i in range(n):
            try:
                a = float(actual_vals[i])
                p = float(pred_vals[i])
                if math.isnan(a) or math.isnan(p):
                    continue
                rows.append({"actual": a, "predicted": p, "error": round(a - p, 8)})
            except (TypeError, ValueError):
                continue
        return rows
    except Exception as e:
        logger.debug("build_regression_holdout_rows: %s", e)
        return []


def _json_safe_scalar(val):
    try:
        if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            return None
        if isinstance(val, (bool, int)):
            return val
        if isinstance(val, float):
            return round(val, 8)
        return str(val)
    except Exception:
        return str(val)


def evaluate_classification_frame(
    model,
    frame,
    target_column: str,
    max_detail_rows: int = 200,
    feature_columns: Optional[list[str]] = None,
) -> tuple[dict, Optional[dict], list[dict]]:
    """Holdout validation only: sklearn precision/recall/f1 + CM + rows (same predict). No H2O leaderboard metrics."""
    empty: tuple[dict, Optional[dict], list] = ({}, None, [])
    if frame is None or int(frame.nrows) == 0 or not target_column:
        return empty
    try:
        from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

        preds = model.predict(frame)
        pdf = preds.as_data_frame()
        pred_col = "predict" if "predict" in pdf.columns else pdf.columns[0]
        prob_cols = [c for c in pdf.columns if c != pred_col]
        y_true_s = frame[target_column].as_data_frame().iloc[:, 0].astype(str)
        y_pred_s = pdf[pred_col].astype(str)
        n = min(len(y_true_s), len(y_pred_s))
        if n == 0:
            return empty
        y_true = [str(y_true_s.iloc[i]) for i in range(n)]
        y_pred = [str(y_pred_s.iloc[i]) for i in range(n)]

        pr, rc, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        metrics: dict[str, float] = {
            "precision": round(float(pr), 6),
            "recall": round(float(rc), 6),
            "f1": round(float(f1), 6),
        }

        labels = sorted(set(y_true) | set(y_pred))
        cm_arr = confusion_matrix(y_true, y_pred, labels=labels)
        cm_data = {
            "labels": [str(x) for x in labels],
            "matrix": [[int(x) for x in row] for row in cm_arr.tolist()],
        }

        names = list(getattr(frame, "names", []) or [])
        feat_use = [c for c in (feature_columns or []) if c in names and c != target_column][:80]
        feat_pdf = None
        if feat_use:
            try:
                feat_pdf = frame[feat_use].as_data_frame()
            except Exception:
                feat_pdf = None

        rows: list[dict] = []
        if max_detail_rows > 0:
            lim = min(n, max_detail_rows)
            for i in range(lim):
                pred_lab = y_pred[i]
                conf = 1.0
                matched = False
                probs_row: dict[str, float] = {}
                for c in prob_cols:
                    try:
                        pv = float(pdf.iloc[i][c])
                        if not (math.isnan(pv) or math.isinf(pv)):
                            probs_row[str(c).strip()] = round(pv, 6)
                    except (TypeError, ValueError):
                        continue
                for c in prob_cols:
                    if str(c).strip() == str(pred_lab).strip():
                        try:
                            v = float(pdf.iloc[i][c])
                            if not (math.isnan(v) or math.isinf(v)):
                                conf = v
                                matched = True
                        except (TypeError, ValueError):
                            pass
                        break
                if not matched and prob_cols:
                    vals = []
                    for c in prob_cols:
                        try:
                            v = float(pdf.iloc[i][c])
                            if not (math.isnan(v) or math.isinf(v)):
                                vals.append(v)
                        except (TypeError, ValueError):
                            continue
                    if vals:
                        conf = max(vals)
                entry: dict = {
                    "actual": y_true[i],
                    "predicted": y_pred[i],
                    "confidence": round(float(conf), 6),
                    "probabilities": probs_row,
                }
                if feat_pdf is not None and i < len(feat_pdf):
                    fd: dict[str, object] = {}
                    for c in feat_use:
                        try:
                            fd[c] = _json_safe_scalar(feat_pdf.iloc[i][c])
                        except Exception:
                            fd[c] = None
                    entry["features"] = fd
                rows.append(entry)

        return metrics, cm_data, rows
    except Exception as e:
        logger.debug("evaluate_classification_frame: %s", e)
        return empty


def write_classification_holdout_csv(
    model,
    frame,
    target_column: str,
    feature_columns: list[str],
    out_path: str,
) -> bool:
    """Export full holdout (or eval frame) rows: features, actual, predicted, class probabilities."""
    if frame is None or int(frame.nrows) == 0:
        return False
    try:
        import pandas as pd

        preds = model.predict(frame)
        pred_pdf = preds.as_data_frame()
        names = list(getattr(frame, "names", []) or [])
        feat_cols = [c for c in feature_columns if c in names and c != target_column]
        parts = []
        if feat_cols:
            parts.append(frame[feat_cols].as_data_frame())
        parts.append(frame[target_column].as_data_frame().rename(columns={target_column: "actual"}))
        parts.append(pred_pdf)
        merged = pd.concat(parts, axis=1)
        merged.to_csv(out_path, index=False)
        return True
    except Exception as e:
        logger.warning("write_classification_holdout_csv: %s", e)
        return False


def write_regression_holdout_csv(
    model,
    frame,
    target_column: str,
    feature_columns: list[str],
    out_path: str,
) -> bool:
    """Export full validation/holdout regression rows: features, actual, predicted, error."""
    if frame is None or int(frame.nrows) == 0:
        return False
    try:
        import pandas as pd

        preds = model.predict(frame)
        pred_df = preds.as_data_frame()
        pc0 = pred_df.columns[0]
        pred_pdf = pred_df.rename(columns={pc0: "predicted"})
        names = list(getattr(frame, "names", []) or [])
        feat_cols = [c for c in feature_columns if c in names and c != target_column]
        parts = []
        if feat_cols:
            parts.append(frame[feat_cols].as_data_frame())
        act = frame[target_column].as_data_frame().rename(columns={target_column: "actual"})
        parts.append(act)
        parts.append(pred_pdf)
        merged = pd.concat(parts, axis=1)
        if "predicted" in merged.columns and "actual" in merged.columns:
            merged["error"] = merged["actual"] - merged["predicted"]
        merged.to_csv(out_path, index=False)
        return True
    except Exception as e:
        logger.warning("write_regression_holdout_csv: %s", e)
        return False


def evaluate_regression_frame(
    model,
    frame,
    target_column: str,
    feature_columns: Optional[list[str]] = None,
    max_detail_rows: int = 200,
) -> tuple[dict, list[dict]]:
    """One predict pass: sklearn core metrics + per-row actual/predicted/error/optional features."""
    empty: tuple[dict, list] = ({}, [])
    if frame is None or int(frame.nrows) == 0 or not target_column:
        return empty
    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        preds = model.predict(frame)
        pred_df = preds.as_data_frame()
        pred_col = pred_df.columns[0]
        y_pred = pred_df[pred_col].tolist()
        actual_series = frame[target_column].as_data_frame().iloc[:, 0]
        y_true = actual_series.tolist()
        n = min(len(y_pred), len(y_true))
        if n == 0:
            return empty

        xs: list[float] = []
        ys: list[float] = []
        for i in range(n):
            try:
                a = float(y_true[i])
                p = float(y_pred[i])
                if math.isnan(a) or math.isnan(p):
                    continue
                xs.append(a)
                ys.append(p)
            except (TypeError, ValueError):
                continue
        if not xs:
            return empty

        mse = float(mean_squared_error(xs, ys))
        metrics = {
            "mse": round(mse, 6),
            "rmse": round(math.sqrt(mse), 6),
            "mae": round(float(mean_absolute_error(xs, ys)), 6),
        }
        r2 = _safe_metric(lambda: float(r2_score(xs, ys)))
        if r2 is not None:
            metrics["r2"] = r2

        try:
            perf = model.model_performance(frame)
            if perf:
                rmsle = _safe_metric(getattr(perf, "rmsle", None))
                if rmsle is not None:
                    metrics["rmsle"] = rmsle
        except Exception:
            pass

        names = list(getattr(frame, "names", []) or [])
        feat_use = [c for c in (feature_columns or []) if c in names and c != target_column][:80]
        feat_pdf = None
        if feat_use:
            try:
                feat_pdf = frame[feat_use].as_data_frame()
            except Exception:
                feat_pdf = None

        rows: list[dict] = []
        if max_detail_rows > 0:
            lim = min(n, max_detail_rows)
            for i in range(lim):
                try:
                    a = float(y_true[i])
                    p = float(y_pred[i])
                    if math.isnan(a) or math.isnan(p):
                        continue
                except (TypeError, ValueError):
                    continue
                entry: dict = {
                    "actual": a,
                    "predicted": p,
                    "error": round(a - p, 8),
                }
                if feat_pdf is not None and i < len(feat_pdf):
                    fd: dict[str, object] = {}
                    for c in feat_use:
                        try:
                            fd[c] = _json_safe_scalar(feat_pdf.iloc[i][c])
                        except Exception:
                            fd[c] = None
                    entry["features"] = fd
                rows.append(entry)

        return metrics, rows
    except Exception as e:
        logger.debug("evaluate_regression_frame: %s", e)
        return empty


def _mean_if_sequence(val) -> float | None:
    """H2O sometimes returns a list of per-class metrics for multinomial models."""
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        nums = []
        for x in val:
            try:
                fx = float(x)
                if not (math.isnan(fx) or math.isinf(fx)):
                    nums.append(fx)
            except (TypeError, ValueError):
                continue
        if not nums:
            return None
        return round(sum(nums) / len(nums), 6)
    try:
        fx = float(val)
        if math.isnan(fx) or math.isinf(fx):
            return None
        return round(fx, 6)
    except (TypeError, ValueError):
        return None


def _classification_prf_from_perf(perf) -> dict[str, float]:
    """Precision, recall, and F1 from H2O model metrics (binomial or multinomial)."""
    out: dict[str, float] = {}
    f1_candidates = ("F1", "f1")
    for key in ("precision", "recall"):
        try:
            fn = getattr(perf, key, None)
            if fn is None:
                continue
            raw = fn() if callable(fn) else fn
            v = _mean_if_sequence(raw)
            if v is not None:
                out[key] = v
        except Exception:
            continue
    for name in f1_candidates:
        try:
            fn = getattr(perf, name, None)
            if fn is None:
                continue
            raw = fn() if callable(fn) else fn
            v = _mean_if_sequence(raw)
            if v is not None:
                out["f1"] = v
                break
        except Exception:
            continue
    return out


def _macro_prf_from_confusion(perf) -> dict[str, float]:
    """Macro precision, recall, F1 from the confusion matrix.

    H2OMultinomialModelMetrics often does not expose .precision()/.recall()/.F1();
    this path works for binomial and multinomial when a confusion matrix exists.
    """
    out: dict[str, float] = {}
    try:
        cm = perf.confusion_matrix()
        if cm is None:
            return out
        table = getattr(cm, "table", None)
        if table is None:
            return out
        try:
            df = table.as_data_frame(use_pandas=True)
        except TypeError:
            df = table.as_data_frame()
        if df is None or getattr(df, "empty", True):
            return out
    except Exception as e:
        logger.debug("confusion matrix unavailable for PRF: %s", e)
        return out

    try:
        label_col = df.columns[0]
        df = df.set_index(label_col)
        # Drop helper columns (Error, Rate, etc.)
        skip = {"error", "err", "rate", "total"}
        keep_cols = []
        for c in df.columns:
            cl = str(c).strip().lower()
            if cl in skip or "err" in cl:
                continue
            keep_cols.append(c)
        df = df[keep_cols]
        df2 = df.copy()
        df2.index = [str(x).strip() for x in df2.index]
        df2.columns = [str(x).strip() for x in df2.columns]
        labels = sorted(set(df2.index) & set(df2.columns))
        labels = [x for x in labels if x.lower() not in skip]
        if len(labels) < 2:
            return out
        sub = df2.loc[labels, labels]
        arr = sub.values.astype(float)
        n = arr.shape[0]
        if n < 2 or arr.shape[0] != arr.shape[1]:
            return out
        precisions = []
        recalls = []
        f1s = []
        for i in range(n):
            col_sum = float(arr[:, i].sum())
            row_sum = float(arr[i, :].sum())
            tp = float(arr[i, i])
            p = tp / col_sum if col_sum > 0 else 0.0
            r = tp / row_sum if row_sum > 0 else 0.0
            precisions.append(p)
            recalls.append(r)
            f1s.append((2 * p * r / (p + r)) if (p + r) > 0 else 0.0)
        out["precision"] = round(sum(precisions) / n, 6)
        out["recall"] = round(sum(recalls) / n, 6)
        out["f1"] = round(sum(f1s) / n, 6)
    except Exception as e:
        logger.debug("macro PRF from confusion matrix: %s", e)
    return out


def _merge_classification_prf(perf) -> dict[str, float]:
    """Prefer direct H2O metric methods; fill gaps from confusion matrix (multinomial)."""
    direct = _classification_prf_from_perf(perf)
    from_cm = _macro_prf_from_confusion(perf)
    merged = {**from_cm, **direct}
    return {k: v for k, v in merged.items() if k in ("precision", "recall", "f1") and v is not None}


def _model_response_column(model) -> Optional[str]:
    """Resolve the training target column name from an H2O model."""
    try:
        p = getattr(model, "actual_params", None) or {}
        rc = p.get("response_column")
        if isinstance(rc, str):
            return rc
        if rc is not None and hasattr(rc, "name"):
            return str(rc.name)
    except Exception:
        pass
    try:
        p = getattr(model, "params", None) or {}
        rc = p.get("response_column")
        if isinstance(rc, dict):
            av = rc.get("actual_value")
            if isinstance(av, str):
                return av
            if av is not None and hasattr(av, "name"):
                return str(av.name)
    except Exception:
        pass
    try:
        j = getattr(model, "_model_json", None) or {}
        resp = j.get("response_column_name")
        if isinstance(resp, str):
            return resp
        out = j.get("output", {})
        if isinstance(out, dict):
            col = out.get("response_column") or out.get("responseColumnName")
            if isinstance(col, str):
                return col
    except Exception:
        pass
    return None


def _macro_prf_sklearn(model, frame, target_column: Optional[str] = None) -> dict[str, float]:
    """Macro precision/recall/F1 from predictions vs actuals (robust for multinomial)."""
    out: dict[str, float] = {}
    try:
        from sklearn.metrics import precision_recall_fscore_support

        target = target_column or _model_response_column(model)
        if not target:
            return out
        names = list(getattr(frame, "names", []) or getattr(frame, "columns", []))
        if target not in names:
            return out

        pred = model.predict(frame)
        y_true = frame[target].asfactor().as_data_frame().iloc[:, 0].astype(str)
        p_df = pred.as_data_frame()
        pred_col = "predict" if "predict" in p_df.columns else p_df.columns[0]
        y_pred = p_df[pred_col].astype(str)
        if len(y_true) != len(y_pred):
            return out
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        out["precision"] = round(float(p), 6)
        out["recall"] = round(float(r), 6)
        out["f1"] = round(float(f1), 6)
    except Exception as e:
        logger.debug("sklearn macro PRF fallback: %s", e)
    return out


def get_model_metrics(
    model,
    frame,
    ml_task: str,
    target_column: Optional[str] = None,
    eval_frame=None,
    feature_columns: Optional[list[str]] = None,
    max_detail_rows: int = 0,
) -> dict:
    """Metrics on a single scoring frame (prefer holdout via eval_frame). Classification/regression use sklearn core + H2O extras on same frame."""
    score_frame = eval_frame if eval_frame is not None else frame
    target = target_column or _model_response_column(model)
    if score_frame is None or int(getattr(score_frame, "nrows", 0) or 0) == 0:
        return _metrics_from_leaderboard(model, ml_task)

    try:
        if ml_task == "classification" and target:
            metrics, _, _ = evaluate_classification_frame(
                model,
                score_frame,
                target,
                max_detail_rows=max_detail_rows,
                feature_columns=feature_columns,
            )
            return {k: v for k, v in metrics.items() if v is not None}
        if ml_task == "regression" and target:
            metrics, _ = evaluate_regression_frame(
                model,
                score_frame,
                target,
                feature_columns=feature_columns,
                max_detail_rows=max_detail_rows,
            )
            return {k: v for k, v in metrics.items() if v is not None}
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")

    return _metrics_from_leaderboard(model, ml_task)


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


def model_is_binomial(model) -> bool:
    """H2O gains/lift is only defined for binary classification."""
    try:
        mc = getattr(model, "model_category", None)
        if mc is None:
            return True
        s = str(mc).lower()
        if "multinomial" in s or "ordinal" in s:
            return False
        return "binomial" in s
    except Exception:
        return True


def sample_prediction_check(model, frame) -> tuple[bool, str]:
    """Verify the leader model can score at least one row (surfaces DL / backend issues in API)."""
    try:
        if frame is None or int(frame.nrows) < 1:
            return True, ""
        sample = frame.head(1)
        pred = model.predict(sample)
        if pred is None or int(pred.nrows) < 1:
            return False, "Model.predict returned no rows for a sample scoring pass."
        return True, ""
    except Exception as e:
        msg = str(e).strip()
        if len(msg) > 220:
            msg = msg[:220] + "…"
        return False, msg


def get_gains_lift(model, frame) -> list[dict]:
    """Extract gains/lift table from model performance (training frame first for small data / stable metrics)."""
    perf = None
    for getter in [
        lambda: model.model_performance(frame),
        lambda: model.model_performance(xval=True),
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
        gl_df = gl.as_data_frame() if hasattr(gl, "as_data_frame") else gl
        rows = []
        cols = {str(c).lower().replace(" ", "_"): c for c in gl_df.columns}

        def col(*names: str):
            for n in names:
                if n in cols:
                    return cols[n]
                for k, v in cols.items():
                    if n in k:
                        return v
            return None

        cdf_c = col("cumulative_data_fraction", "cumulative_data_pct")
        lift_c = col("cumulative_lift", "lift")
        gain_c = col("cumulative_gain", "cumulative_capture")

        for i, row in gl_df.iterrows():
            rdict = row.to_dict() if hasattr(row, "to_dict") else dict(row)
            cdf = 0.0
            if cdf_c and cdf_c in rdict:
                cdf = float(rdict[cdf_c])
            elif "cumulative_data_fraction" in rdict:
                cdf = float(rdict["cumulative_data_fraction"])
            lift_v = 0.0
            if lift_c and lift_c in rdict:
                lift_v = float(rdict[lift_c])
            elif "cumulative_lift" in rdict:
                lift_v = float(rdict["cumulative_lift"])
            gain_v = 0.0
            if gain_c and gain_c in rdict:
                gain_v = float(rdict[gain_c])
            elif "cumulative_gain" in rdict:
                gain_v = float(rdict["cumulative_gain"])
            grp = int(rdict.get("group", i + 1)) if "group" in rdict else i + 1
            rows.append({
                "group": grp,
                "cumulative_data_pct": round(cdf * 100, 1) if cdf <= 1.0 else round(cdf, 1),
                "lift": round(lift_v, 2),
                "gain_pct": round(gain_v * 100, 1) if gain_v <= 1.0 else round(gain_v, 1),
            })
        return rows
    except Exception as e:
        logger.debug("Gains/lift not available: %s", e)
    return []


def _json_friendly_cell(val):
    """Convert pandas / numpy scalars so FastAPI can JSON-encode the random-row response."""
    import math

    import numpy as np
    import pandas as pd

    if val is None:
        return None
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        fv = float(val)
        if math.isnan(fv) or math.isinf(fv):
            return None
        return fv
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, pd.Timestamp):
        return val.isoformat()
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    return val


def get_random_row(frame, target: str, features: list[str]) -> dict:
    """Get a random row from the frame as feature values (JSON-serializable Python scalars)."""
    df = frame.as_data_frame()
    row = df[features].iloc[random.randint(0, len(df) - 1)]
    return {col: _json_friendly_cell(row[col]) for col in features}


def save_model(model, path: str) -> str:
    return _h2o.save_model(model=model, path=path, force=True)


def load_model(path: str):
    return _h2o.load_model(path)


def _safe_metric(fn):
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
