export interface DatasetMetadata {
  id: string;
  filename: string;
  total_rows: number;
  total_columns: number;
  size_bytes: number;
  category: string;
  description: string;
}

export interface ColumnInfo {
  name: string;
  dtype: string;
  null_count: number;
  unique_count: number;
  sample_values: string[];
}

export interface DatasetColumnsResponse {
  dataset_id: string;
  columns: ColumnInfo[];
}

export interface DatasetPreviewResponse {
  dataset_id: string;
  columns: string[];
  rows: Record<string, unknown>[];
  total_rows: number;
}

export interface AIRecommendResponse {
  target_column: string;
  features: string[];
  confidence: string;
  reasoning: string;
}

export interface UseCaseSuggestion {
  use_case: string;
  ml_task: string;
  target_hint: string;
}

export interface UseCaseSuggestionsResponse {
  dataset_id: string;
  suggestions: UseCaseSuggestion[];
}

export interface TrainingStartRequest {
  dataset_id: string;
  target_column: string;
  feature_columns: string[];
  ml_task: MLTask;
  models: ModelType[];
  auto_mode: boolean;
  train_test_split: number;
  nfolds: number;
  max_models: number;
  max_runtime_secs: number;
}

export interface TrainingStartResponse {
  run_id: string;
  status: string;
  message: string;
}

export interface TrainingStatusResponse {
  run_id: string;
  status: string;
  progress_percent: number;
  current_stage: string;
  message: string;
}

export interface ModelResult {
  model_id: string;
  algorithm: string;
  metrics: Record<string, number | null>;
  rank: number;
  is_best: boolean;
}

export interface LeaderboardResponse {
  run_id: string;
  ml_task: string;
  primary_metric: string;
  models: ModelResult[];
}

export interface FeatureImportanceResponse {
  run_id: string;
  model_id: string;
  features: Record<string, unknown>[];
}

export interface ConfusionMatrixResponse {
  run_id: string;
  model_id: string;
  labels: string[];
  matrix: number[][];
}

export interface ResidualsResponse {
  run_id: string;
  model_id: string;
  actual: number[];
  predicted: number[];
}

export type MLTask = 'classification' | 'regression' | 'clustering';
export type ModelType = 'DRF' | 'GLM' | 'XGBoost' | 'GBM' | 'DeepLearning' | 'StackedEnsemble';

export type WizardStep = 0 | 1 | 2 | 3 | 4;

export interface WizardState {
  currentStep: WizardStep;
  dataset: DatasetMetadata | null;
  columns: ColumnInfo[];
  targetColumn: string;
  featureColumns: string[];
  mlTask: MLTask;
  selectedModels: ModelType[];
  autoMode: boolean;
  trainTestSplit: number;
  nfolds: number;
  maxModels: number;
  maxRuntimeSecs: number;
  runId: string | null;
  trainingStatus: TrainingStatusResponse | null;
  leaderboard: LeaderboardResponse | null;
}

export interface HFDatasetInfo {
  hf_id: string;
  hf_url: string;
  name: string;
  filename: string;
  task: string;
  rows: number;
  cols: number;
  size_kb: number;
  description: string;
  target_hint: string;
}

export const MODEL_INFO: Record<ModelType, { name: string; speed: string; description: string }> = {
  DRF: { name: 'Distributed Random Forest', speed: 'Medium', description: 'Ensemble of random trees' },
  GLM: { name: 'Generalized Linear Model', speed: 'Fast', description: 'Logistic regression variant' },
  XGBoost: { name: 'XGBoost', speed: 'Medium', description: 'Gradient boosting framework' },
  GBM: { name: 'Gradient Boosting Machine', speed: 'Medium', description: "H2O's GBM implementation" },
  DeepLearning: { name: 'Deep Learning', speed: 'Slow', description: 'Neural network models' },
  StackedEnsemble: { name: 'Stacked Ensemble', speed: 'Slow', description: 'Meta-learner combining models' },
};
