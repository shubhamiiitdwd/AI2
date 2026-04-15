"""
Clustering AutoML Engine — multi-algorithm grid search with normalized scoring.

Tests KMeans, GMM, DBSCAN across hyperparameter grids, evaluates with
Silhouette / Calinski-Harabasz / Davies-Bouldin, then ranks by composite score.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score,
)
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

# Columns that must not be used as clustering inputs (labels / leakage from prior runs).
_LEAKAGE_SUBSTRINGS = ("cluster_label", "cluster_id", "prediction_cluster")


def sanitize_clustering_feature_columns(feature_cols: list[str]) -> list[str]:
    """Drop label-like columns so clustering and F-importance stay meaningful."""
    out = []
    for c in feature_cols:
        cl = c.lower().strip()
        if cl in _LEAKAGE_SUBSTRINGS or cl.endswith("_cluster"):
            continue
        out.append(c)
    return out


# ── Feature preparation ────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame, feature_cols: list[str]):
    """Scale numerics, one-hot encode categoricals. Returns (X_scaled, feature_names)."""
    numeric_cols = [c for c in feature_cols if df[c].dtype in ("float64", "int64", "float32", "int32")]
    cat_cols = [c for c in feature_cols if c not in numeric_cols]

    parts = []
    names = []

    if numeric_cols:
        num_data = df[numeric_cols].fillna(0).values
        scaler = StandardScaler()
        parts.append(scaler.fit_transform(num_data))
        names.extend(numeric_cols)

    if cat_cols:
        try:
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            cat_data = enc.fit_transform(df[cat_cols].fillna("_missing_"))
            parts.append(cat_data)
            names.extend(enc.get_feature_names_out(cat_cols).tolist())
        except Exception as e:
            logger.warning(f"One-hot encoding failed for {cat_cols}: {e}. Dropping categoricals.")

    if not parts:
        raise ValueError("No usable features after preparation")

    X = np.hstack(parts)
    return X, names


# ── Algorithm shortlisting ─────────────────────────────────────────────────

def shortlist_algorithms(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    n_rows = len(df)
    candidates = ["kmeans", "gmm"]
    if n_rows >= 500:
        candidates.append("dbscan")
    return candidates


# ── Hyperparameter grid ────────────────────────────────────────────────────

def build_param_grid(
    algorithms: list[str],
    user_algorithm: str | None = None,
    user_n_clusters: int | None = None,
    user_eps: float | None = None,
    user_min_samples: int | None = None,
) -> dict[str, list[dict]]:
    grid: dict[str, list[dict]] = {}

    if user_algorithm:
        algorithms = [user_algorithm]

    for algo in algorithms:
        if algo == "kmeans":
            if user_n_clusters:
                grid["kmeans"] = [{"n_clusters": user_n_clusters}]
            else:
                grid["kmeans"] = [{"n_clusters": k} for k in range(2, 11)]

        elif algo == "gmm":
            if user_n_clusters:
                grid["gmm"] = [{"n_components": user_n_clusters, "covariance_type": "full"}]
            else:
                grid["gmm"] = [
                    {"n_components": k, "covariance_type": cov}
                    for k in range(2, 11)
                    for cov in ["full", "tied", "diag"]
                ]

        elif algo == "dbscan":
            if user_eps and user_min_samples:
                grid["dbscan"] = [{"eps": user_eps, "min_samples": user_min_samples}]
            else:
                grid["dbscan"] = [
                    {"eps": eps, "min_samples": ms}
                    for eps in [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
                    for ms in [3, 5, 10, 15]
                ]

    return grid


# ── Model fitting ──────────────────────────────────────────────────────────

def fit_single_model(X: np.ndarray, algorithm: str, params: dict) -> np.ndarray:
    if algorithm == "kmeans":
        model = KMeans(
            n_clusters=params["n_clusters"],
            random_state=params.get("random_state", 42),
            n_init=10,
        )
        return model.fit_predict(X)

    elif algorithm == "gmm":
        model = GaussianMixture(
            n_components=params["n_components"],
            covariance_type=params.get("covariance_type", "full"),
            random_state=params.get("random_state", 42),
            n_init=3,
        )
        return model.fit_predict(X)

    elif algorithm == "dbscan":
        model = DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
        return model.fit_predict(X)

    raise ValueError(f"Unknown algorithm: {algorithm}")


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate_model(X: np.ndarray, labels: np.ndarray) -> dict | None:
    n_labels = len(set(labels) - {-1})
    if n_labels < 2:
        return None

    mask = labels != -1
    if mask.sum() < 2:
        return None

    X_valid = X[mask]
    labels_valid = labels[mask]

    try:
        return {
            "silhouette": float(silhouette_score(X_valid, labels_valid)),
            "calinski_harabasz": float(calinski_harabasz_score(X_valid, labels_valid)),
            "davies_bouldin": float(davies_bouldin_score(X_valid, labels_valid)),
            "n_clusters": n_labels,
            "n_noise": int((~mask).sum()),
        }
    except Exception as e:
        logger.warning(f"Metric computation failed: {e}")
        return None


# ── Normalized composite scoring ───────────────────────────────────────────

def compute_composite_score(results: list[dict]) -> list[dict]:
    if not results:
        return results

    def _min_max(vals):
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return [0.5] * len(vals)
        return [(v - mn) / (mx - mn) for v in vals]

    sil_norm = _min_max([r["silhouette"] for r in results])
    ch_norm = _min_max([r["calinski_harabasz"] for r in results])
    db_norm = _min_max([r["davies_bouldin"] for r in results])

    for i, r in enumerate(results):
        r["composite_score"] = round(
            0.40 * sil_norm[i] + 0.35 * ch_norm[i] - 0.25 * db_norm[i], 4
        )

    return sorted(results, key=lambda x: x["composite_score"], reverse=True)


def select_best_model(scored: list[dict]) -> dict:
    return scored[0]


# ── Stability check ───────────────────────────────────────────────────────

def stability_check(X: np.ndarray, algorithm: str, params: dict, n_runs: int = 5) -> dict:
    if algorithm == "dbscan":
        return {"avg_ari": 1.0, "is_stable": True, "n_runs": 1}

    all_labels = []
    for seed in range(n_runs):
        labels = fit_single_model(X, algorithm, {**params, "random_state": seed})
        all_labels.append(labels)

    ari_scores = []
    for i in range(len(all_labels)):
        for j in range(i + 1, len(all_labels)):
            ari_scores.append(adjusted_rand_score(all_labels[i], all_labels[j]))

    avg_ari = sum(ari_scores) / len(ari_scores) if ari_scores else 0.0
    return {
        "avg_ari": round(float(avg_ari), 4),
        "is_stable": avg_ari > 0.8,
        "n_runs": n_runs,
    }


# ── Cluster summaries & feature importance ─────────────────────────────────

def get_cluster_summary(
    df: pd.DataFrame, labels: np.ndarray, feature_cols: list[str],
) -> list[dict]:
    df_copy = df[feature_cols].copy()
    df_copy["_cluster"] = labels
    clean = df_copy[df_copy["_cluster"] != -1]
    total = len(clean)
    summaries = []
    for cid in sorted(clean["_cluster"].unique()):
        group = clean[clean["_cluster"] == cid]
        centroid = {}
        for col in feature_cols:
            if group[col].dtype in ("float64", "int64", "float32", "int32"):
                centroid[col] = round(float(group[col].mean()), 4)
        summaries.append({
            "cluster_id": int(cid),
            "size": len(group),
            "percentage": round(len(group) / total * 100, 1) if total else 0,
            "centroid": centroid,
        })
    return summaries


def get_feature_importance_per_cluster(
    df: pd.DataFrame, labels: np.ndarray, feature_cols: list[str],
) -> list[dict]:
    from sklearn.feature_selection import f_classif

    feature_cols = sanitize_clustering_feature_columns(feature_cols)
    numeric_cols = [c for c in feature_cols if df[c].dtype in ("float64", "int64", "float32", "int32")]
    if not numeric_cols:
        return []

    mask = labels != -1
    X_valid = df.loc[mask, numeric_cols].fillna(0).values
    y_valid = labels[mask]

    if len(set(y_valid)) < 2:
        return []

    try:
        f_scores, _ = f_classif(X_valid, y_valid)
        f_scores = np.asarray(f_scores, dtype=float)
        f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)
        total = float(np.nansum(f_scores))
        if total <= 0 or not np.isfinite(total):
            total = 1.0
        result = []
        for col, f_val in zip(numeric_cols, f_scores):
            imp = float(f_val / total) if total > 0 else 0.0
            if not np.isfinite(imp):
                imp = 0.0
            result.append({"feature": col, "importance": round(imp, 6)})
        return sorted(result, key=lambda x: x["importance"], reverse=True)
    except Exception as e:
        logger.warning(f"Feature importance failed: {e}")
        return []


# ── Elbow analysis ─────────────────────────────────────────────────────────

def get_elbow_data(X: np.ndarray, max_k: int = 10) -> tuple[list[dict], int]:
    """
    KMeans-only sweep for visualization: pick K in [2, max_k] with best silhouette.
    Independent of the multi-algo leaderboard (GMM/DBSCAN and composite score).
    """
    data = []
    best_k, best_sil = 2, -1.0
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        sil = float(silhouette_score(X, labels))
        data.append({
            "k": k,
            "inertia": round(float(model.inertia_), 2),
            "silhouette": round(sil, 4),
        })
        if sil > best_sil:
            best_sil = sil
            best_k = k
    return data, best_k


# ── Dimensionality reduction ──────────────────────────────────────────────

def reduce_dimensions_pca(X: np.ndarray, labels: np.ndarray) -> list[dict]:
    n_components = min(2, X.shape[1])
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(X)
    points = []
    for i in range(len(labels)):
        points.append({
            "x": round(float(coords[i, 0]), 4),
            "y": round(float(coords[i, 1]), 4) if n_components > 1 else 0.0,
            "cluster": int(labels[i]),
        })
    return points


# ── Full pipeline (synchronous, called from executor) ─────────────────────

def run_full_pipeline(
    df: pd.DataFrame,
    feature_cols: list[str],
    user_algorithm: str | None = None,
    user_n_clusters: int | None = None,
    user_eps: float | None = None,
    user_min_samples: int | None = None,
    run_stability: bool = True,
    progress_callback=None,
) -> dict:
    """
    Full clustering AutoML: prepare → shortlist → grid → train all → score → best.
    progress_callback(pct, msg) is called at each stage for WebSocket updates.
    """
    def _progress(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    _progress(10, "Scaling & encoding features...")
    feature_cols = sanitize_clustering_feature_columns(feature_cols)
    if not feature_cols:
        raise ValueError("No valid feature columns after removing label/leakage columns.")
    X, feature_names = prepare_features(df, feature_cols)

    _progress(15, "Shortlisting candidate algorithms...")
    candidates = shortlist_algorithms(df, feature_cols)
    param_grid = build_param_grid(candidates, user_algorithm, user_n_clusters, user_eps, user_min_samples)
    total_combos = sum(len(v) for v in param_grid.values())

    all_results: list[dict] = []
    done = 0
    for algo, params_list in param_grid.items():
        for params in params_list:
            done += 1
            pct = 20 + int(50 * done / total_combos)
            _progress(min(pct, 70), f"Training {algo} ({params}) [{done}/{total_combos}]")

            try:
                labels = fit_single_model(X, algo, params)
                metrics = evaluate_model(X, labels)
                if metrics:
                    all_results.append({
                        "algorithm": algo,
                        "params": params,
                        "labels": labels,
                        **metrics,
                    })
            except Exception as e:
                logger.warning(f"Model {algo}({params}) failed: {e}")

    if not all_results:
        raise ValueError("All clustering candidates failed. Check your features.")

    _progress(75, f"Scoring {len(all_results)} models...")
    scored = compute_composite_score(all_results)
    best = select_best_model(scored)

    stability = None
    if run_stability:
        _progress(80, "Running stability check...")
        stability = stability_check(X, best["algorithm"], best["params"])

    _progress(85, "Generating visualizations...")
    pca_points = reduce_dimensions_pca(X, best["labels"])
    summaries = get_cluster_summary(df, best["labels"], feature_cols)
    feat_imp = get_feature_importance_per_cluster(df, best["labels"], feature_cols)

    _progress(88, "Computing elbow analysis...")
    elbow_data, recommended_k = get_elbow_data(X)

    leaderboard = []
    for rank, r in enumerate(scored, 1):
        leaderboard.append({
            "rank": rank,
            "algorithm": r["algorithm"],
            "params": r["params"],
            "n_clusters": r["n_clusters"],
            "n_noise_points": r.get("n_noise", 0),
            "silhouette": round(r["silhouette"], 4),
            "calinski_harabasz": round(r["calinski_harabasz"], 2),
            "davies_bouldin": round(r["davies_bouldin"], 4),
            "composite_score": round(r["composite_score"], 4),
            "is_best": rank == 1,
        })

    return {
        "best_algorithm": best["algorithm"],
        "best_params": best["params"],
        "best_labels": best["labels"],
        "best_metrics": {
            "silhouette_score": round(best["silhouette"], 4),
            "calinski_harabasz": round(best["calinski_harabasz"], 2),
            "davies_bouldin": round(best["davies_bouldin"], 4),
            "composite_score": round(best["composite_score"], 4),
            "n_clusters": best["n_clusters"],
            "n_noise_points": best.get("n_noise", 0),
        },
        "stability": stability,
        "cluster_summaries": summaries,
        "leaderboard": leaderboard,
        "feature_importance": feat_imp,
        "feature_columns": feature_cols,
        "total_candidates_tested": total_combos,
        "pca_points": pca_points,
        "elbow_data": elbow_data,
        "recommended_k": recommended_k,
    }
