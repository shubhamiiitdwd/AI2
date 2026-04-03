import axios from 'axios';
import type {
  DatasetMetadata, DatasetColumnsResponse, DatasetPreviewResponse,
  AIRecommendResponse, TrainingStartRequest, TrainingStartResponse,
  TrainingStatusResponse, LeaderboardResponse, FeatureImportanceResponse,
  ConfusionMatrixResponse, ResidualsResponse, UseCaseSuggestionsResponse,
  HFDatasetInfo,
} from './types';

const BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8001';
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

export const generateAISummary = async (runId: string) => {
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

export const getWsUrl = (runId: string) => {
  const wsBase = BASE.replace(/^http/, 'ws');
  return `${wsBase}/team1/ws/training/${runId}`;
};
