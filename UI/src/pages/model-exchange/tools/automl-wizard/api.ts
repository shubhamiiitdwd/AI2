import axios, { isAxiosError } from 'axios';
import type {
  DatasetMetadata, DatasetColumnsResponse, DatasetPreviewResponse,
  AIRecommendResponse, TrainingStartRequest, TrainingStartResponse,
  TrainingStatusResponse, TrainingHistoryResponse, LeaderboardResponse, FeatureImportanceResponse,
  ConfusionMatrixResponse, ResidualsResponse, UseCaseSuggestionsResponse,
  HFDatasetInfo, AISummaryResponse, AutoDetectTaskResponse,
  ClusteringStartRequest, ClusteringStartResponse,
  ClusteringResultResponse, ElbowResponse,
  HoldoutEvaluationResponse,
  DatasetWorkflowInsightResponse,
  ClusteringLabeledPreviewResponse,
  TextInsightResponse,
  DataLibraryIndexResponse,
  DataLibraryImportResponse,
} from './types';

const DEFAULT_BACKEND_PORT = '8099';

/**
 * Team1 AutoML API origin (port 8099 by default). Never use VITE_API_BASE_URL here — that is often
 * Team2 / another service (e.g. :8000). Module library must call the same host as run_local.
 */
function getTeam1ApiOrigin(): string {
  const a = (import.meta.env.VITE_API_URL as string | undefined)?.trim();
  const b = (import.meta.env.VITE_TEAM1_API_URL as string | undefined)?.trim();
  if (a) return a.replace(/\/+$/, '');
  if (b) return b.replace(/\/+$/, '');
  const port = (import.meta.env.VITE_BACKEND_PORT as string | undefined) || DEFAULT_BACKEND_PORT;
  return `http://127.0.0.1:${port}`;
}

// Default client: dev = same-origin + Vite proxy to FastAPI; prod = full API URL (or VITE_API_URL).
const BASE =
  import.meta.env.VITE_API_URL ||
  (import.meta.env.DEV
    ? ''
    : `http://localhost:${import.meta.env.VITE_BACKEND_PORT || DEFAULT_BACKEND_PORT}`);
const api = axios.create({ baseURL: BASE });

/**
 * Module data library (Azure) only: always call Team1 on getTeam1ApiOrigin() — same pattern as
 * connecting WebSockets directly to the API port (avoids HTTP 405 when /team1 hits the Vite port).
 */
const dataLibraryApi = axios.create({
  baseURL: getTeam1ApiOrigin(),
});

/** User-visible message for failed /team1 calls (FastAPI `detail` or network). */
export function formatTeam1AxiosError(err: unknown, fallback: string): string {
  if (!isAxiosError(err)) return fallback;
  if (err.code === 'ERR_NETWORK' || err.message === 'Network Error') {
    return (
      'Cannot reach the API. Start the Team1 backend from the AI_Kosh_Project folder ' +
      '(e.g. python -m modules.team1_automl.run_local) and keep the default port 8099 if you use npm run dev for the UI.'
    );
  }
  const d = err.response?.data;
  if (d && typeof d === 'object' && 'detail' in d) {
    const det = (d as { detail?: unknown }).detail;
    if (typeof det === 'string' && det.trim()) return det.trim();
    if (Array.isArray(det)) {
      const parts = det.map((x) => {
        if (typeof x === 'object' && x !== null && 'msg' in x) return String((x as { msg: unknown }).msg);
        return String(x);
      });
      if (parts.length) return parts.join(' ');
    }
  }
  if (err.response?.status) return `${fallback} (HTTP ${err.response.status})`;
  return fallback;
}

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

export const getDatasetWorkflowInsight = async (id: string): Promise<DatasetWorkflowInsightResponse> => {
  const { data } = await api.get(`/team1/datasets/${id}/workflow-insight`);
  return data;
};

export const deleteDataset = async (id: string): Promise<void> => {
  await api.delete(`/team1/datasets/${id}`);
};

export const listDataLibrary = async (): Promise<DataLibraryIndexResponse> => {
  const { data } = await dataLibraryApi.get('/team1/datasets/data-library');
  return data;
};

export const importDataLibrary = async (folder: string, filename: string): Promise<DataLibraryImportResponse> => {
  const { data } = await dataLibraryApi.post('/team1/datasets/data-library/import', { folder, filename });
  return data;
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

export const getTrainingHistory = async (limit?: number): Promise<TrainingHistoryResponse> => {
  const { data } = await api.get('/team1/training/history', {
    params: limit != null ? { limit } : undefined,
  });
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

export const getHoldoutEvaluation = async (runId: string): Promise<HoldoutEvaluationResponse> => {
  const { data } = await api.get(`/team1/results/${runId}/holdout-evaluation`);
  return data;
};

export const getExportUrl = (runId: string, format: string = 'csv') =>
  `${BASE}/team1/results/${runId}/export?format=${format}`;

export const getHoldoutPredictionsCsvUrl = (runId: string) =>
  `${BASE}/team1/results/${runId}/holdout-predictions.csv`;

export const getHoldoutRegressionPredictionsCsvUrl = (runId: string) =>
  `${BASE}/team1/results/${runId}/holdout-regression-predictions.csv`;

export const predict = async (runId: string, featureValues: Record<string, unknown>) => {
  const { data } = await api.post(`/team1/results/${runId}/predict`, { feature_values: featureValues });
  return data;
};

export const getRandomRow = async (runId: string) => {
  const { data } = await api.get(`/team1/results/${runId}/random-row`);
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

export const getClusteringElbowInsight = async (runId: string): Promise<TextInsightResponse> => {
  const { data } = await api.get(`/team1/clustering/${runId}/elbow-insight`);
  return data;
};

export const getClusteringLabeledPreview = async (
  runId: string,
  rows = 10,
  cols = 10,
): Promise<ClusteringLabeledPreviewResponse> => {
  const { data } = await api.get(`/team1/clustering/${runId}/labeled-preview`, {
    params: { rows, cols },
  });
  return data;
};

export const getClusteringLabeledCsvUrl = (runId: string) =>
  `${BASE}/team1/clustering/${runId}/labeled-export.csv`;

export const getWsUrl = (runId: string) => {
  if (BASE) {
    const wsBase = BASE.replace(/^http/, 'ws');
    return `${wsBase}/team1/ws/training/${runId}`;
  }
  const host = typeof window !== 'undefined' ? window.location.hostname : '127.0.0.1';
  const port = import.meta.env.VITE_BACKEND_PORT || DEFAULT_BACKEND_PORT;
  return `ws://${host}:${port}/team1/ws/training/${runId}`;
};
