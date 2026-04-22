from fastapi import APIRouter, UploadFile, File, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional

from .schemas import (
    DatasetMetadata, DatasetColumnsResponse, DatasetPreviewResponse,
    AIRecommendRequest, AIRecommendResponse,
    ValidateConfigRequest, ValidateConfigResponse,
    TrainingStartRequest, TrainingStartResponse, TrainingStatusResponse,
    LeaderboardResponse, FeatureImportanceResponse,
    ConfusionMatrixResponse, ResidualsResponse, HoldoutEvaluationResponse, ExportResponse,
    UseCaseSuggestionsResponse, AutoDetectTaskResponse,
    PredictRequest, PredictResponse, GainsLiftResponse, AISummaryResponse,
    HFDatasetInfo, HFImportRequest,
    ClusteringStartRequest, ClusteringStartResponse,
    ClusteringResultResponse, ElbowResponse,
    TrainingHistoryResponse,
    DatasetWorkflowInsightResponse,
    ClusteringLabeledPreviewResponse,
    TextInsightResponse,
    DataLibraryIndexResponse, DataLibraryImportRequest, DataLibraryImportResponse,
)
from . import services
from . import hf_datasets
from . import data_gov_catalog

router = APIRouter(prefix="/team1", tags=["Team 1 - AutoML"])
router.include_router(data_gov_catalog.router)


# ── Dataset Management ─────────────────────────────────────────────────────

@router.post("/datasets/upload", response_model=DatasetMetadata)
async def upload_dataset(file: UploadFile = File(...)):
    return await services.upload_dataset(file)


@router.get("/datasets", response_model=list[DatasetMetadata])
async def list_datasets():
    return await services.list_datasets()


# HuggingFace routes must come before {dataset_id} wildcard routes
@router.get("/datasets/huggingface/browse")
async def browse_hf_datasets(task: Optional[str] = Query(default=None)):
    return hf_datasets.get_curated_list(task)


@router.post("/datasets/huggingface/import", response_model=DatasetMetadata)
async def import_hf_dataset(req: HFImportRequest):
    result = await hf_datasets.import_hf_dataset(req.hf_id)
    return result


@router.get("/datasets/data-library", response_model=DataLibraryIndexResponse)
def list_data_library():
    return services.list_data_library_index()


@router.post(
    "/datasets/data-library",
    response_model=DataLibraryIndexResponse,
    summary="List module data library (POST = same as GET; fixes HTTP 405 if a client or proxy issues POST).",
)
def list_data_library_post():
    return services.list_data_library_index()


@router.post("/datasets/data-library/import", response_model=DataLibraryImportResponse)
async def import_data_library(req: DataLibraryImportRequest):
    return await services.import_data_library_dataset(req)


@router.get("/datasets/{dataset_id}/preview", response_model=DatasetPreviewResponse)
async def preview_dataset(dataset_id: str, rows: int = Query(default=10, le=100)):
    return await services.preview_dataset(dataset_id, rows)


@router.get("/datasets/{dataset_id}/columns", response_model=DatasetColumnsResponse)
async def get_columns(dataset_id: str):
    return await services.get_dataset_columns(dataset_id)


@router.get("/datasets/{dataset_id}/workflow-insight", response_model=DatasetWorkflowInsightResponse)
async def dataset_workflow_insight(dataset_id: str):
    return await services.get_dataset_workflow_insight(dataset_id)


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    return await services.delete_dataset(dataset_id)


# ── Column Configuration ───────────────────────────────────────────────────

@router.get("/configure/suggest-usecases/{dataset_id}", response_model=UseCaseSuggestionsResponse)
async def suggest_usecases(dataset_id: str):
    return await services.suggest_usecases(dataset_id)


@router.post("/configure/auto-detect-task/{dataset_id}", response_model=AutoDetectTaskResponse)
async def auto_detect_task(dataset_id: str):
    return await services.auto_detect_task(dataset_id)


@router.post("/configure/ai-recommend", response_model=AIRecommendResponse)
async def ai_recommend(req: AIRecommendRequest):
    return await services.ai_recommend(req)


@router.post("/configure/validate", response_model=ValidateConfigResponse)
async def validate_config(req: ValidateConfigRequest):
    return await services.validate_config(req)


# ── Training ───────────────────────────────────────────────────────────────

@router.post("/training/start", response_model=TrainingStartResponse)
async def start_training(req: TrainingStartRequest):
    return await services.start_training(req)


# History routes must come before {run_id} wildcard routes
@router.get("/training/history", response_model=TrainingHistoryResponse)
async def list_training_history(limit: Optional[int] = Query(default=None, le=5000)):
    return await services.list_training_history(limit=limit)


@router.get("/training/history/{dataset_id}", response_model=TrainingHistoryResponse)
async def get_dataset_training_history(dataset_id: str):
    return await services.get_dataset_training_history(dataset_id)


@router.get("/training/{run_id}/status", response_model=TrainingStatusResponse)
async def training_status(run_id: str):
    return await services.get_training_status(run_id)


@router.post("/training/{run_id}/stop")
async def stop_training(run_id: str):
    return await services.stop_training(run_id)


@router.websocket("/ws/training/{run_id}")
async def training_ws(websocket: WebSocket, run_id: str):
    await services.training_websocket(websocket, run_id)


# ── Results ────────────────────────────────────────────────────────────────

@router.get("/results/{run_id}/leaderboard", response_model=LeaderboardResponse)
async def get_leaderboard(run_id: str):
    return await services.get_leaderboard(run_id)


@router.get("/results/{run_id}/best-model")
async def get_best_model(run_id: str):
    return await services.get_best_model(run_id)


@router.get("/results/{run_id}/feature-importance", response_model=FeatureImportanceResponse)
async def get_feature_importance(run_id: str):
    return await services.get_feature_importance(run_id)


@router.get("/results/{run_id}/confusion-matrix", response_model=ConfusionMatrixResponse)
async def get_confusion_matrix(run_id: str):
    return await services.get_confusion_matrix(run_id)


@router.get("/results/{run_id}/residuals", response_model=ResidualsResponse)
async def get_residuals(run_id: str):
    return await services.get_residuals(run_id)


@router.get("/results/{run_id}/holdout-evaluation", response_model=HoldoutEvaluationResponse)
async def get_holdout_evaluation(run_id: str):
    return await services.get_holdout_evaluation(run_id)


@router.get("/results/{run_id}/holdout-predictions.csv")
async def export_holdout_predictions_csv(run_id: str):
    return await services.export_holdout_predictions_csv(run_id)


@router.get("/results/{run_id}/holdout-regression-predictions.csv")
async def export_holdout_regression_predictions_csv(run_id: str):
    return await services.export_holdout_regression_predictions_csv(run_id)


@router.get("/results/{run_id}/export")
async def export_results(run_id: str, format: str = Query(default="csv")):
    return await services.export_results(run_id, format)


@router.post("/results/{run_id}/predict", response_model=PredictResponse)
async def predict(run_id: str, req: PredictRequest):
    return await services.predict(run_id, req)


@router.get("/results/{run_id}/random-row")
async def random_row(run_id: str):
    return await services.get_random_row(run_id)


@router.get("/results/{run_id}/gains-lift", response_model=GainsLiftResponse)
async def get_gains_lift(run_id: str):
    return await services.get_gains_lift(run_id)


@router.post("/results/{run_id}/ai-summary", response_model=AISummaryResponse)
async def generate_ai_summary(run_id: str):
    return await services.generate_ai_summary(run_id)


# ── Clustering ─────────────────────────────────────────────────────────────

@router.post("/clustering/start", response_model=ClusteringStartResponse)
async def start_clustering(req: ClusteringStartRequest):
    return await services.start_clustering(req)


@router.get("/clustering/{run_id}/status")
async def clustering_status(run_id: str):
    return await services.get_clustering_status(run_id)


@router.get("/clustering/{run_id}/result", response_model=ClusteringResultResponse)
async def clustering_result(run_id: str):
    return await services.get_clustering_result(run_id)


@router.get("/clustering/{run_id}/elbow", response_model=ElbowResponse)
async def clustering_elbow(run_id: str):
    return await services.get_elbow_analysis(run_id)


@router.get("/clustering/{run_id}/elbow-insight", response_model=TextInsightResponse)
async def clustering_elbow_insight(run_id: str):
    return await services.get_clustering_elbow_insight(run_id)


@router.get("/clustering/{run_id}/labeled-preview", response_model=ClusteringLabeledPreviewResponse)
async def clustering_labeled_preview(run_id: str, rows: int = Query(default=10, le=50), cols: int = Query(default=10, le=50)):
    return await services.get_clustering_labeled_preview(run_id, max_rows=rows, max_cols=cols)


@router.get("/clustering/{run_id}/labeled-export.csv")
async def clustering_labeled_export_csv(run_id: str):
    return await services.export_clustering_labeled_csv(run_id)


@router.websocket("/ws/clustering/{run_id}")
async def clustering_ws(websocket: WebSocket, run_id: str):
    await services.clustering_websocket(websocket, run_id)


# ── Load Persisted Results ─────────────────────────────────────────────────

@router.post("/results/{run_id}/load")
async def load_persisted_result(run_id: str):
    return await services.load_persisted_result(run_id)
