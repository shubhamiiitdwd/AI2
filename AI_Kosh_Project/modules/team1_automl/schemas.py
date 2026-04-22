from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional
from .enums import MLTask, ModelType, TrainingStatus


class DatasetMetadata(BaseModel):
    id: str
    filename: str
    total_rows: int
    total_columns: int
    size_bytes: int
    category: str = "Uploaded Dataset"
    description: str = ""


class DataLibraryFileRef(BaseModel):
    name: str
    size_bytes: int = 0


class DataLibraryFolderInfo(BaseModel):
    folder: str
    files: list[DataLibraryFileRef]


class DataLibraryIndexResponse(BaseModel):
    source: str
    folders: list[DataLibraryFolderInfo]


class DataLibraryImportRequest(BaseModel):
    folder: str
    filename: str


class DataLibraryImportResponse(BaseModel):
    accepted: bool
    dataset: Optional[DatasetMetadata] = None
    message: str = ""
    insight: Optional[DatasetWorkflowInsightResponse] = None


class ColumnInfo(BaseModel):
    name: str
    dtype: str
    null_count: int
    unique_count: int
    sample_values: list = []


class DatasetColumnsResponse(BaseModel):
    dataset_id: str
    columns: list[ColumnInfo]


class DatasetPreviewResponse(BaseModel):
    dataset_id: str
    columns: list[str]
    rows: list[dict]
    total_rows: int


class AIRecommendRequest(BaseModel):
    dataset_id: str
    use_case: str


class AIRecommendResponse(BaseModel):
    target_column: str
    features: list[str]
    confidence: str = "high confidence"
    reasoning: str = ""
    source: str = "rule-based"


class UseCaseSuggestion(BaseModel):
    use_case: str
    ml_task: str
    target_hint: str


class UseCaseSuggestionsResponse(BaseModel):
    dataset_id: str
    suggestions: list[UseCaseSuggestion]
    # Optional: filled when Azure (or future LLM) returns structured rationale
    reasoning: str = ""
    confidence: str = ""
    recommended_task: str = ""


class ValidateConfigRequest(BaseModel):
    dataset_id: str
    target_column: str
    feature_columns: list[str]
    ml_task: MLTask


class ValidateConfigResponse(BaseModel):
    valid: bool
    message: str = ""
    suggested_task: Optional[MLTask] = None


class TrainingStartRequest(BaseModel):
    dataset_id: str
    target_column: str
    feature_columns: list[str]
    ml_task: MLTask
    models: list[ModelType] = []
    auto_mode: bool = False
    train_test_split: float = 0.8
    nfolds: int = 5
    max_models: int = 20
    max_runtime_secs: int = 300


class TrainingStartResponse(BaseModel):
    run_id: str
    status: TrainingStatus
    message: str


class TrainingStatusResponse(BaseModel):
    run_id: str
    status: TrainingStatus
    progress_percent: int = 0
    current_stage: str = ""
    message: str = ""
    # Detailed lines for clustering / training progress when exposed by the backend
    logs: list[str] = Field(default_factory=list)


class ModelResult(BaseModel):
    model_id: str
    algorithm: str
    metrics: dict
    rank: int
    is_best: bool = False


class LeaderboardResponse(BaseModel):
    run_id: str
    ml_task: str
    primary_metric: str
    models: list[ModelResult]


class FeatureImportanceResponse(BaseModel):
    run_id: str
    model_id: str
    features: list[dict]


class ConfusionMatrixResponse(BaseModel):
    run_id: str
    model_id: str
    labels: list[str]
    matrix: list[list[int]]


class ResidualsResponse(BaseModel):
    run_id: str
    model_id: str
    actual: list[float]
    predicted: list[float]
    errors: list[float] = Field(default_factory=list)


class ClassificationHoldoutRow(BaseModel):
    actual: str
    predicted: str
    confidence: float
    probabilities: dict = Field(default_factory=dict)
    features: dict = Field(default_factory=dict)


class RegressionHoldoutRow(BaseModel):
    actual: float
    predicted: float
    error: float
    features: dict = Field(default_factory=dict)


class HoldoutEvaluationResponse(BaseModel):
    run_id: str
    model_id: str
    train_ratio_config: float
    test_ratio_config: float
    train_rows: int
    test_rows: int
    train_fraction_actual: float = 0.0
    test_fraction_actual: float = 0.0
    classification_rows: list[ClassificationHoldoutRow] = Field(default_factory=list)
    regression_rows: list[RegressionHoldoutRow] = Field(default_factory=list)


class ExportResponse(BaseModel):
    run_id: str
    download_url: str
    format: str


class PredictRequest(BaseModel):
    feature_values: dict


class PredictionModelResult(BaseModel):
    model_id: str
    prediction: Optional[str] = None
    class_probabilities: Optional[dict] = None
    error: Optional[str] = None


class PredictResponse(BaseModel):
    run_id: str
    predictions: list[PredictionModelResult]


class GainsLiftRow(BaseModel):
    group: int
    cumulative_data_pct: float
    lift: float
    gain_pct: float


class GainsLiftResponse(BaseModel):
    run_id: str
    rows: list[GainsLiftRow]
    note: str = ""


class BestModelSummary(BaseModel):
    model_id: str
    algorithm: str
    metrics: dict
    all_metrics: dict


class AISummaryResponse(BaseModel):
    executive_summary: str
    key_insights: list[str]
    recommendations: list[str]
    real_world_example: str
    source: str = "rule-based"


class AutoDetectTaskResponse(BaseModel):
    task: str
    confidence: str
    reasoning: str
    suggestions: list["UseCaseSuggestion"] = []
    source: str = "rule-based"


class HFDatasetInfo(BaseModel):
    hf_id: str
    hf_url: str
    name: str
    filename: str
    task: str
    rows: int
    cols: int
    size_kb: int
    description: str
    target_hint: str


class HFImportRequest(BaseModel):
    hf_id: str


# ── Clustering AutoML Schemas ──────────────────────────────────────────────

class ClusteringStartRequest(BaseModel):
    dataset_id: str
    feature_columns: list[str]
    algorithm: Optional[str] = None
    n_clusters: Optional[int] = None
    eps: Optional[float] = None
    min_samples: Optional[int] = None
    run_stability_check: bool = True


class ClusteringStartResponse(BaseModel):
    run_id: str
    status: str
    message: str


class CandidateModelResult(BaseModel):
    rank: int
    algorithm: str
    params: dict
    n_clusters: int
    n_noise_points: int = 0
    silhouette: float
    calinski_harabasz: float
    davies_bouldin: float
    composite_score: float
    is_best: bool = False


class StabilityResult(BaseModel):
    avg_ari: float
    is_stable: bool
    n_runs: int


class ClusterMetrics(BaseModel):
    silhouette_score: float
    calinski_harabasz: float
    davies_bouldin: float
    composite_score: float
    n_clusters: int
    n_noise_points: int = 0


class ClusterSummary(BaseModel):
    cluster_id: int
    size: int
    percentage: float
    centroid: dict


class ClusterFeatureImportance(BaseModel):
    feature: str
    importance: float = 0.0


class DimensionReductionPoint(BaseModel):
    x: float
    y: float
    cluster: int


class ClusteringResultResponse(BaseModel):
    run_id: str
    best_algorithm: str
    best_params: dict
    best_metrics: ClusterMetrics
    stability: Optional[StabilityResult] = None
    cluster_summaries: list[ClusterSummary]
    leaderboard: list[CandidateModelResult]
    feature_importance: list[ClusterFeatureImportance]
    feature_columns: list[str]
    total_candidates_tested: int
    pca_points: Optional[list[DimensionReductionPoint]] = None


class ElbowDataPoint(BaseModel):
    k: int
    inertia: float
    silhouette: float


class ElbowResponse(BaseModel):
    run_id: str
    data: list[ElbowDataPoint]
    recommended_k: int


# ── Training History Schemas (Azure persistence) ──────────────────────────

class TrainingRunSummary(BaseModel):
    run_id: str
    dataset_id: str
    dataset_name: str = ""
    ml_task: str = ""
    target_column: str = ""
    best_model_id: str = ""
    best_algorithm: str = ""
    primary_metric: str = ""
    best_metric_value: float = 0.0
    model_count: int = 0
    status: str = "complete"
    run_type: str = "training"
    created_at: str = ""


class TrainingHistoryResponse(BaseModel):
    runs: list[TrainingRunSummary]


class DatasetWorkflowInsightResponse(BaseModel):
    """Whether the CSV looks tabular for H2O AutoML / clustering vs raw text needing prep."""

    is_structured_tabular: bool
    needs_data_exchange: bool
    suggest_automl: bool
    headline: str
    detail: str
    source: str = "rules"
    data_characteristics: str = ""
    preprocessing_guidance: str = ""
    feature_engineering_guidance: str = ""


class ClusteringLabeledPreviewResponse(BaseModel):
    run_id: str
    columns: list[str]
    rows: list[dict]


class TextInsightResponse(BaseModel):
    text: str
    source: str = "rules"

