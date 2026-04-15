import axios from 'axios';
import type {
  DatasetMetadata, DatasetColumnsResponse, DatasetPreviewResponse,
  AIRecommendResponse, TrainingStartRequest, TrainingStartResponse,
  TrainingStatusResponse, TrainingHistoryResponse, LeaderboardResponse, FeatureImportanceResponse,
  ConfusionMatrixResponse, ResidualsResponse, UseCaseSuggestionsResponse,
  HFDatasetInfo, AISummaryResponse, AutoDetectTaskResponse,
  ClusteringStartRequest, ClusteringStartResponse,
  ClusteringResultResponse, ElbowResponse,
} from './types';

const DEFAULT_BACKEND_PORT = '8099';

// Dev: empty baseURL → requests go to Vite; vite.config.ts proxies /team1 to FastAPI (no CORS).
// Prod: set VITE_API_URL or fall back to localhost API (port must match VITE_BACKEND_PORT / backend).
const BASE =
  import.meta.env.VITE_API_URL ||
  (import.meta.env.DEV
    ? ''
    : `http://localhost:${import.meta.env.VITE_BACKEND_PORT || DEFAULT_BACKEND_PORT}`);
const api = axios.create({ baseURL: BASE });

export const uploadDataset = async (file: File): Promise<DatasetMetadata> => {
  const form = new FormData();
  form.append('file', file);
  const { data } = await api.post('/team1/datasets/upload', form);
  return data;
};

export const listDatasets = async (): Promise<DatasetMetadata[]> => {
  const { data } = await api.get('/team1/datasets');
  return data;
};

export const getDatasetPreview = async (id: string, rows = 10): Promise<DatasetPreviewResponse> => {
  const { data } = await api.get(`/team1/datasets/${id}/preview`, { params: { rows } });
  return data;
};

export const getDatasetColumns = async (id: string): Promise<DatasetColumnsResponse> => {
  const { data } = await api.get(`/team1/datasets/${id}/columns`);
  return data;
};

export const deleteDataset = async (id: string): Promise<void> => {
  await api.delete(`/team1/datasets/${id}`);
};

export const suggestUseCases = async (datasetId: string): Promise<UseCaseSuggestionsResponse> => {
  const { data } = await api.get(`/team1/configure/suggest-usecases/${datasetId}`);
  return data;
};

export const autoDetectTask = async (datasetId: string): Promise<AutoDetectTaskResponse> => {
  const { data } = await api.post(`/team1/configure/auto-detect-task/${datasetId}`);
  return data;
};

export const aiRecommend = async (datasetId: string, useCase: string): Promise<AIRecommendResponse> => {
  const { data } = await api.post('/team1/configure/ai-recommend', { dataset_id: datasetId, use_case: useCase });
  return data;
};

export const validateConfig = async (body: { dataset_id: string; target_column: string; feature_columns: string[]; ml_task: string }) => {
  const { data } = await api.post('/team1/configure/validate', body);
  return data;
};

export const startTraining = async (req: TrainingStartRequest): Promise<TrainingStartResponse> => {
  const { data } = await api.post('/team1/training/start', req);
  return data;
};

export const getTrainingStatus = async (runId: string): Promise<TrainingStatusResponse> => {
  const { data } = await api.get(`/team1/training/${runId}/status`);
  return data;
};

export const stopTraining = async (runId: string): Promise<void> => {
  await api.post(`/team1/training/${runId}/stop`);
};

export const getTrainingHistory = async (): Promise<TrainingHistoryResponse> => {
  const { data } = await api.get('/team1/training/history');
  return data;
};

export const loadPersistedResult = async (runId: string): Promise<Record<string, unknown>> => {
  const { data } = await api.post(`/team1/results/${runId}/load`);
  return data;
};

export const getLeaderboard = async (runId: string): Promise<LeaderboardResponse> => {
  const { data } = await api.get(`/team1/results/${runId}/leaderboard`);
  return data;
};

export const getBestModel = async (runId: string) => {
  const { data } = await api.get(`/team1/results/${runId}/best-model`);
  return data;
};

export const getFeatureImportance = async (runId: string): Promise<FeatureImportanceResponse> => {
  const { data } = await api.get(`/team1/results/${runId}/feature-importance`);
  return data;
};

export const getConfusionMatrix = async (runId: string): Promise<ConfusionMatrixResponse> => {
  const { data } = await api.get(`/team1/results/${runId}/confusion-matrix`);
  return data;
};

export const getResiduals = async (runId: string): Promise<ResidualsResponse> => {
  const { data } = await api.get(`/team1/results/${runId}/residuals`);
  return data;
};

export const getExportUrl = (runId: string, format: string = 'csv') =>
  `${BASE}/team1/results/${runId}/export?format=${format}`;

export const predict = async (runId: string, featureValues: Record<string, unknown>) => {
  const { data } = await api.post(`/team1/results/${runId}/predict`, { feature_values: featureValues });
  return data;
};

export const getRandomRow = async (runId: string) => {
  const { data } = await api.get(`/team1/results/${runId}/random-row`);
  return data;
};

export const getGainsLift = async (runId: string) => {
  const { data } = await api.get(`/team1/results/${runId}/gains-lift`);
  return data;
};

export const generateAISummary = async (runId: string): Promise<AISummaryResponse> => {
  const { data } = await api.post(`/team1/results/${runId}/ai-summary`);
  return data;
};

export const browseHFDatasets = async (task?: string): Promise<HFDatasetInfo[]> => {
  const params = task ? { task } : {};
  const { data } = await api.get('/team1/datasets/huggingface/browse', { params });
  return data;
};

export const importHFDataset = async (hfId: string): Promise<DatasetMetadata> => {
  const { data } = await api.post('/team1/datasets/huggingface/import', { hf_id: hfId });
  return data;
};

// ── Clustering API ──────────────────────────────────────────────────────

export const startClustering = async (req: ClusteringStartRequest): Promise<ClusteringStartResponse> => {
  const { data } = await api.post('/team1/clustering/start', req);
  return data;
};

export const getClusteringStatus = async (runId: string): Promise<TrainingStatusResponse> => {
  const { data } = await api.get(`/team1/clustering/${runId}/status`);
  return data;
};

export const getClusteringResult = async (runId: string): Promise<ClusteringResultResponse> => {
  const { data } = await api.get(`/team1/clustering/${runId}/result`);
  return data;
};

export const getElbowAnalysis = async (runId: string): Promise<ElbowResponse> => {
  const { data } = await api.get(`/team1/clustering/${runId}/elbow`);
  return data;
};

export const getWsUrl = (runId: string) => {
  if (BASE) {
    const wsBase = BASE.replace(/^http/, 'ws');
    return `${wsBase}/team1/ws/training/${runId}`;
  }
  // Do not send training WebSockets through the Vite dev proxy — it often logs
  // "ws proxy error: write ECONNABORTED" when FastAPI closes the socket or uvicorn reloads.
  // HTTP still uses the Vite proxy (same-origin). WS connects straight to the API port.
  const host = typeof window !== 'undefined' ? window.location.hostname : '127.0.0.1';
  const port = import.meta.env.VITE_BACKEND_PORT || DEFAULT_BACKEND_PORT;
  return `ws://${host}:${port}/team1/ws/training/${runId}`;
};
