export interface DatasetMetadata {
  id: string;
  filename: string;
  total_rows: number;
  total_columns: number;
  size_bytes: number;
  category: string;
  description: string;
}

/** Shared module datasets (Azure blob or local DATA_LIBRARY). */
export interface DataLibraryFileRef {
  name: string;
  size_bytes: number;
}

export interface DataLibraryFolderInfo {
  folder: string;
  files: DataLibraryFileRef[];
}

export interface DataLibraryIndexResponse {
  source: string;
  folders: DataLibraryFolderInfo[];
}

export interface DataLibraryImportResponse {
  accepted: boolean;
  dataset?: DatasetMetadata;
  message: string;
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
  source?: string;
}

export interface UseCaseSuggestion {
  use_case: string;
  ml_task: string;
  target_hint: string;
}

export interface UseCaseSuggestionsResponse {
  dataset_id: string;
  suggestions: UseCaseSuggestion[];
  reasoning?: string;
  confidence?: string;
  recommended_task?: string;
}

export interface AISummaryResponse {
  executive_summary: string;
  key_insights: string[];
  recommendations: string[];
  real_world_example: string;
  source?: string;
}

export interface AutoDetectTaskResponse {
  task: 'classification' | 'regression' | 'clustering';
  confidence: string;
  reasoning: string;
  suggestions: UseCaseSuggestion[];
  source: string;
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
  /** Backend may return incremental training / clustering log lines */
  logs?: string[];
}

export interface TrainingRunSummary {
  run_id: string;
  dataset_id: string;
  dataset_name: string;
  ml_task: string;
  target_column: string;
  best_model_id: string;
  best_algorithm: string;
  primary_metric: string;
  best_metric_value: number;
  model_count: number;
  status: string;
  run_type: string;
  created_at: string;
}

export interface TrainingHistoryResponse {
  runs: TrainingRunSummary[];
}

export interface DatasetWorkflowInsightResponse {
  is_structured_tabular: boolean;
  needs_data_exchange: boolean;
  suggest_automl: boolean;
  headline: string;
  detail: string;
  source?: string;
  data_characteristics?: string;
  preprocessing_guidance?: string;
  feature_engineering_guidance?: string;
}

export interface ClusteringLabeledPreviewResponse {
  run_id: string;
  columns: string[];
  rows: Record<string, unknown>[];
}

export interface TextInsightResponse {
  text: string;
  source?: string;
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
  errors?: number[];
}

export interface ClassificationHoldoutRow {
  actual: string;
  predicted: string;
  confidence: number;
  probabilities?: Record<string, number>;
  features?: Record<string, unknown>;
}

export interface RegressionHoldoutRow {
  actual: number;
  predicted: number;
  error: number;
  features?: Record<string, unknown>;
}

export interface HoldoutEvaluationResponse {
  run_id: string;
  model_id: string;
  train_ratio_config: number;
  test_ratio_config: number;
  train_rows: number;
  test_rows: number;
  train_fraction_actual: number;
  test_fraction_actual: number;
  classification_rows: ClassificationHoldoutRow[];
  regression_rows: RegressionHoldoutRow[];
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
  clusteringResult: ClusteringResultResponse | null;
}

// ── Clustering Types ──────────────────────────────────────────────────────

export interface ClusteringStartRequest {
  dataset_id: string;
  feature_columns: string[];
  algorithm?: string;
  n_clusters?: number;
  eps?: number;
  min_samples?: number;
  run_stability_check: boolean;
}

export interface ClusteringStartResponse {
  run_id: string;
  status: string;
  message: string;
}

export interface CandidateModelResult {
  rank: number;
  algorithm: string;
  params: Record<string, unknown>;
  n_clusters: number;
  n_noise_points: number;
  silhouette: number;
  calinski_harabasz: number;
  davies_bouldin: number;
  composite_score: number;
  is_best: boolean;
}

export interface StabilityResult {
  avg_ari: number;
  is_stable: boolean;
  n_runs: number;
}

export interface ClusterMetrics {
  silhouette_score: number;
  calinski_harabasz: number;
  davies_bouldin: number;
  composite_score: number;
  n_clusters: number;
  n_noise_points: number;
}

export interface ClusterSummary {
  cluster_id: number;
  size: number;
  percentage: number;
  centroid: Record<string, number>;
}

export interface ClusterFeatureImportance {
  feature: string;
  importance: number;
}

export interface DimensionReductionPoint {
  x: number;
  y: number;
  cluster: number;
}

export interface ClusteringResultResponse {
  run_id: string;
  best_algorithm: string;
  best_params: Record<string, unknown>;
  best_metrics: ClusterMetrics;
  stability: StabilityResult | null;
  cluster_summaries: ClusterSummary[];
  leaderboard: CandidateModelResult[];
  feature_importance: ClusterFeatureImportance[];
  feature_columns: string[];
  total_candidates_tested: number;
  pca_points: DimensionReductionPoint[] | null;
}

export interface ElbowDataPoint {
  k: number;
  inertia: number;
  silhouette: number;
}

export interface ElbowResponse {
  run_id: string;
  data: ElbowDataPoint[];
  recommended_k: number;
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
