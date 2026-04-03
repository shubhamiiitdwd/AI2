from fastapi import APIRouter, UploadFile, File, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional

from .schemas import (
    DatasetMetadata, DatasetColumnsResponse, DatasetPreviewResponse,
    AIRecommendRequest, AIRecommendResponse,
    ValidateConfigRequest, ValidateConfigResponse,
    TrainingStartRequest, TrainingStartResponse, TrainingStatusResponse,
    LeaderboardResponse, FeatureImportanceResponse,
    ConfusionMatrixResponse, ResidualsResponse, ExportResponse,
    UseCaseSuggestionsResponse,
    PredictRequest, PredictResponse, GainsLiftResponse, AISummaryResponse,
    HFDatasetInfo, HFImportRequest,
)
from . import services
from . import hf_datasets

router = APIRouter(prefix="/team1", tags=["Team 1 - AutoML"])


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


@router.get("/datasets/{dataset_id}/preview", response_model=DatasetPreviewResponse)
async def preview_dataset(dataset_id: str, rows: int = Query(default=10, le=100)):
    return await services.preview_dataset(dataset_id, rows)


@router.get("/datasets/{dataset_id}/columns", response_model=DatasetColumnsResponse)
async def get_columns(dataset_id: str):
    return await services.get_dataset_columns(dataset_id)


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    return await services.delete_dataset(dataset_id)


# ── Column Configuration ───────────────────────────────────────────────────

@router.get("/configure/suggest-usecases/{dataset_id}", response_model=UseCaseSuggestionsResponse)
async def suggest_usecases(dataset_id: str):
    return await services.suggest_usecases(dataset_id)


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
