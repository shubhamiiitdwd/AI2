import { useState, useEffect } from 'react';
import type {
  ColumnInfo, DatasetMetadata, AIRecommendResponse, DatasetPreviewResponse, UseCaseSuggestion, MLTask, AutoDetectTaskResponse,
  ClusteringStartRequest,
} from '../types';
import * as api from '../api';
import { aiSourceDisplay } from '../aiSource';

interface Props {
  datasetId: string;
  dataset: DatasetMetadata | null;
  columns: ColumnInfo[];
  targetColumn: string;
  featureColumns: string[];
  onTargetChange: (col: string) => void;
  onFeaturesChange: (cols: string[]) => void;
  onTaskSuggest: (task: MLTask) => void;
  onContinue: () => void;
  onClusteringDetected?: () => void;
  onBack?: () => void;
  backLabel?: string;
  /** Unsupervised: no target column; Continue enabled when at least one feature is selected. */
  clusteringMode?: boolean;
  /** When set, shows algorithm options + Start Clustering instead of Continue (step stays on configuration until the run starts). */
  onClusteringStart?: (runId: string) => void | Promise<void>;
}

export default function StepConfigureData({
  datasetId, dataset, columns, targetColumn, featureColumns,
  onTargetChange, onFeaturesChange, onTaskSuggest, onContinue,
  onClusteringDetected, onBack, backLabel, clusteringMode = false, onClusteringStart,
}: Props) {
  const [activeTab, setActiveTab] = useState<'config' | 'preview'>('config');
  /** Clustering: default tab shows only algorithm & search; features are on a separate tab. */
  const [clusteringPanel, setClusteringPanel] = useState<'algorithm' | 'features'>('algorithm');
  const [useCase, setUseCase] = useState('');
  const [aiResult, setAiResult] = useState<AIRecommendResponse | null>(null);
  const [aiLoading, setAiLoading] = useState(false);
  const [preview, setPreview] = useState<DatasetPreviewResponse | null>(null);
  const [suggestions, setSuggestions] = useState<UseCaseSuggestion[]>([]);
  const [suggestionsLoading, setSuggestionsLoading] = useState(false);
  const [taskFilter, setTaskFilter] = useState<MLTask | null>(null);

  const [detectResult, setDetectResult] = useState<AutoDetectTaskResponse | null>(null);
  const [detecting, setDetecting] = useState(false);

  const [clAlgorithm, setClAlgorithm] = useState('');
  const [clNClusters, setClNClusters] = useState<number | undefined>(undefined);
  const [clEps, setClEps] = useState<number | undefined>(undefined);
  const [clMinSamples, setClMinSamples] = useState<number | undefined>(undefined);
  const [clStability, setClStability] = useState(true);
  const [clStarting, setClStarting] = useState(false);

  useEffect(() => {
    if (clusteringMode) return;
    setSuggestionsLoading(true);
    api.suggestUseCases(datasetId)
      .then((res) => setSuggestions(res.suggestions))
      .catch(() => {})
      .finally(() => setSuggestionsLoading(false));
  }, [datasetId, clusteringMode]);

  const handleAutoDetect = async () => {
    setDetecting(true);
    try {
      const result = await api.autoDetectTask(datasetId);
      setDetectResult(result);
      if (result.suggestions.length > 0) {
        setSuggestions(result.suggestions);
      }
      const detectedTask = result.task as MLTask;
      setTaskFilter(detectedTask);
      onTaskSuggest(detectedTask);
      if (detectedTask === 'clustering' && onClusteringDetected) {
        onClusteringDetected();
      }
    } catch { /* ignore */ }
    finally { setDetecting(false); }
  };

  const handleTaskSelect = (task: MLTask) => {
    setTaskFilter(task);
    onTaskSuggest(task);
    setDetectResult(null);
  };

  const handleAIRecommend = async () => {
    if (!useCase.trim()) return;
    setAiLoading(true);
    try {
      const result = await api.aiRecommend(datasetId, useCase);
      setAiResult(result);
      onTargetChange(result.target_column);
      onFeaturesChange(result.features);
      if (taskFilter) onTaskSuggest(taskFilter);
    } catch { /* ignore */ }
    finally { setAiLoading(false); }
  };

  const handleFeatureToggle = (col: string) => {
    if (col === targetColumn) return;
    const newFeatures = featureColumns.includes(col)
      ? featureColumns.filter((c) => c !== col)
      : [...featureColumns, col];
    onFeaturesChange(newFeatures);
  };

  const loadPreview = async () => {
    if (!preview) {
      try {
        const data = await api.getDatasetPreview(datasetId, 10);
        setPreview(data);
      } catch { /* ignore */ }
    }
    setActiveTab('preview');
  };

  const DTYPE_COLORS: Record<string, string> = {
    'object': '#e67e22',
    'float64': '#2ecc71',
    'int64': '#3498db',
    'bool': '#9b59b6',
  };

  const visibleSuggestions = taskFilter
    ? suggestions.filter((s) => s.ml_task === taskFilter)
    : [];

  const numericCols = columns.filter((c) =>
    c.dtype === 'int64' || c.dtype === 'float64' || c.dtype === 'int32' || c.dtype === 'float32'
  );

  const handleClusteringStart = async () => {
    if (!onClusteringStart || featureColumns.length === 0) return;
    setClStarting(true);
    try {
      const req: ClusteringStartRequest = {
        dataset_id: datasetId,
        feature_columns: featureColumns,
        algorithm: clAlgorithm || undefined,
        n_clusters: clNClusters,
        eps: clAlgorithm === 'dbscan' ? clEps : undefined,
        min_samples: clAlgorithm === 'dbscan' ? clMinSamples : undefined,
        run_stability_check: clStability,
      };
      const resp = await api.startClustering(req);

      // Cache hit — backend returned previous results instantly
      if (resp.status === 'complete') {
        // Fetch and display results directly, skipping the execution step
        try {
          const [result, elbow] = await Promise.all([
            api.getClusteringResult(resp.run_id),
            api.getElbowAnalysis(resp.run_id).catch(() => null),
          ]);
          // Set the result on the parent so the results step renders
          await onClusteringStart(resp.run_id);
          // Import: the parent's onComplete will be called when StepClustering fetches results
        } catch {
          // Fallback: proceed normally so polling will retry
          await onClusteringStart(resp.run_id);
        }
      } else {
        await onClusteringStart(resp.run_id);
      }
    } catch {
      /* ignore */
    } finally {
      setClStarting(false);
    }
  };

  return (
    <div className="aw-step-content">
      <div className={`aw-step-main ${clusteringMode ? 'aw-step-main--wide' : ''}`}>
        {onBack && (
          <button className="aw-back-btn" onClick={onBack}>{backLabel || '← Back to Select Dataset'}</button>
        )}
        <div className="aw-tab-bar">
          {clusteringMode ? (
            <>
              <button
                type="button"
                className={`aw-tab ${activeTab === 'config' && clusteringPanel === 'algorithm' ? 'aw-tab--active' : ''}`}
                onClick={() => { setActiveTab('config'); setClusteringPanel('algorithm'); }}
              >Algorithm &amp; search</button>
              <button
                type="button"
                className={`aw-tab ${activeTab === 'config' && clusteringPanel === 'features' ? 'aw-tab--active' : ''}`}
                onClick={() => { setActiveTab('config'); setClusteringPanel('features'); }}
              >Features</button>
              <button
                type="button"
                className={`aw-tab ${activeTab === 'preview' ? 'aw-tab--active' : ''}`}
                onClick={loadPreview}
              >Dataset preview</button>
            </>
          ) : (
            <>
              <button
                type="button"
                className={`aw-tab ${activeTab === 'config' ? 'aw-tab--active' : ''}`}
                onClick={() => setActiveTab('config')}
              >Column Configuration</button>
              <button
                type="button"
                className={`aw-tab ${activeTab === 'preview' ? 'aw-tab--active' : ''}`}
                onClick={loadPreview}
              >Dataset Preview</button>
            </>
          )}
        </div>

        {activeTab === 'config' && (
          <>
            <div className="aw-config-header">
              <h3>
                {clusteringMode
                  ? clusteringPanel === 'algorithm'
                    ? 'Algorithm & search'
                    : 'Feature columns'
                  : 'Column Configuration'}
              </h3>
              <p>
                {clusteringMode
                  ? clusteringPanel === 'algorithm'
                    ? 'Set how models are searched. Pick or adjust feature columns on the Features tab.'
                    : 'Choose which columns to cluster on (exclude IDs and prior cluster labels).'
                  : 'Select target variable and feature columns'}
              </p>
            </div>
            {(!clusteringMode || clusteringPanel === 'features') && (
            <div className="aw-columns-table">
              <div className="aw-col-header">
                <span className="aw-col-check">Feature</span>
                <span className="aw-col-name">Column Name</span>
                <span className="aw-col-type">Type</span>
                <span className="aw-col-null">Nulls</span>
                <span className="aw-col-unique">Unique</span>
                {!clusteringMode && <span className="aw-col-target">Target</span>}
              </div>
              {columns.map((col) => (
                <div key={col.name} className={`aw-col-row ${!clusteringMode && col.name === targetColumn ? 'aw-col-row--target' : ''}`}>
                  <span className="aw-col-check">
                    <input
                      type="checkbox"
                      checked={featureColumns.includes(col.name)}
                      onChange={() => handleFeatureToggle(col.name)}
                      disabled={!clusteringMode && col.name === targetColumn}
                    />
                  </span>
                  <span className="aw-col-name">{col.name}</span>
                  <span className="aw-col-type">
                    <span className="aw-dtype-badge" style={{ backgroundColor: DTYPE_COLORS[col.dtype] || '#95a5a6' }}>
                      {col.dtype}
                    </span>
                  </span>
                  <span className="aw-col-null">{col.null_count}</span>
                  <span className="aw-col-unique">{col.unique_count}</span>
                  {!clusteringMode && (
                    <span className="aw-col-target">
                      <input
                        type="radio"
                        name="target"
                        checked={col.name === targetColumn}
                        onChange={() => {
                          onTargetChange(col.name);
                          onFeaturesChange(columns.filter((c) => c.name !== col.name).map((c) => c.name));
                        }}
                      />
                    </span>
                  )}
                </div>
              ))}
            </div>
            )}

            {clusteringMode && onClusteringStart && clusteringPanel === 'algorithm' && (
              <div className="cl-config-section cl-config-section--embedded">
                <div className="cl-config-section">
                  <label className="cl-label">Algorithm (leave empty for Auto)</label>
                  <select className="cl-select" value={clAlgorithm} onChange={(e) => setClAlgorithm(e.target.value)}>
                    <option value="">Auto (try all)</option>
                    <option value="kmeans">KMeans</option>
                    <option value="gmm">GMM (Gaussian Mixture)</option>
                    <option value="dbscan">DBSCAN</option>
                  </select>
                </div>
                {(clAlgorithm === 'kmeans' || clAlgorithm === 'gmm' || clAlgorithm === '') && (
                  <div className="cl-config-section">
                    <label className="cl-label">Number of Clusters (optional, 2–20)</label>
                    <input
                      className="cl-input"
                      type="number" min={2} max={20}
                      value={clNClusters ?? ''}
                      onChange={(e) => setClNClusters(e.target.value ? Number(e.target.value) : undefined)}
                      placeholder="Auto (grid search 2–10)"
                    />
                  </div>
                )}
                {clAlgorithm === 'dbscan' && (
                  <>
                    <div className="cl-config-section">
                      <label className="cl-label">DBSCAN eps (optional)</label>
                      <input
                        className="cl-input"
                        type="number" step="0.1" min={0.01}
                        value={clEps ?? ''}
                        onChange={(e) => setClEps(e.target.value ? Number(e.target.value) : undefined)}
                        placeholder="Auto grid search"
                      />
                    </div>
                    <div className="cl-config-section">
                      <label className="cl-label">DBSCAN min_samples (optional)</label>
                      <input
                        className="cl-input"
                        type="number" min={1}
                        value={clMinSamples ?? ''}
                        onChange={(e) => setClMinSamples(e.target.value ? Number(e.target.value) : undefined)}
                        placeholder="Auto grid search"
                      />
                    </div>
                  </>
                )}
                <div className="cl-config-section">
                  <label className="cl-toggle-row">
                    <input type="checkbox" checked={clStability} onChange={(e) => setClStability(e.target.checked)} />
                    <span>Run Stability Check (5 runs, compare ARI)</span>
                  </label>
                </div>
              </div>
            )}

            {clusteringMode && onClusteringStart ? (
              clusteringPanel === 'algorithm' ? (
                <>
                  <button
                    className="aw-btn aw-btn--primary aw-btn--full"
                    type="button"
                    onClick={handleClusteringStart}
                    disabled={featureColumns.length === 0 || clStarting}
                  >
                    {clStarting ? 'Starting…' : 'Start clustering →'}
                  </button>
                  <div className="cl-info-note" style={{ marginTop: 12 }}>
                    <strong>Features selected:</strong> {featureColumns.length} columns.
                    Numeric: {numericCols.filter((c) => featureColumns.includes(c.name)).length}. Categorical will be one-hot encoded.
                    {featureColumns.length === 0 && (
                      <span> Select at least one column on the <button type="button" className="cl-inline-link" onClick={() => setClusteringPanel('features')}>Features</button> tab.</span>
                    )}
                  </div>
                </>
              ) : (
                <p className="cl-info-note" style={{ marginTop: 12 }}>
                  When ready, open <button type="button" className="cl-inline-link" onClick={() => setClusteringPanel('algorithm')}>Algorithm &amp; search</button> to start the run.
                </p>
              )
            ) : (
              <button
                className="aw-btn aw-btn--primary aw-btn--full"
                onClick={onContinue}
                disabled={clusteringMode ? featureColumns.length === 0 : !targetColumn}
              >
                Continue →
              </button>
            )}
          </>
        )}

        {activeTab === 'preview' && preview && (
          <div className="aw-preview-table-wrap">
            {dataset && (
              <div className="aw-preview-summary">
                <div className="aw-preview-stat">
                  <span className="aw-preview-stat-label">Total Rows</span>
                  <span className="aw-preview-stat-value">{dataset.total_rows.toLocaleString()}</span>
                </div>
                <div className="aw-preview-stat">
                  <span className="aw-preview-stat-label">Total Columns</span>
                  <span className="aw-preview-stat-value">{dataset.total_columns}</span>
                </div>
                <div className="aw-preview-stat">
                  <span className="aw-preview-stat-label">Size</span>
                  <span className="aw-preview-stat-value">
                    {dataset.size_bytes < 1024
                      ? `${dataset.size_bytes} B`
                      : dataset.size_bytes < 1024 * 1024
                        ? `${(dataset.size_bytes / 1024).toFixed(1)} KB`
                        : `${(dataset.size_bytes / (1024 * 1024)).toFixed(1)} MB`}
                  </span>
                </div>
              </div>
            )}
            <p className="aw-preview-hint">Showing top {preview.rows.length} rows of {preview.total_rows.toLocaleString()}</p>
            <table className="aw-preview-table">
              <thead>
                <tr>{preview.columns.map((c) => <th key={c}>{c}</th>)}</tr>
              </thead>
              <tbody>
                {preview.rows.map((row, i) => (
                  <tr key={i}>{preview.columns.map((c) => <td key={c}>{String(row[c] ?? '')}</td>)}</tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {!clusteringMode && (
      <div className="aw-step-sidebar">
        <div className="aw-ai-panel">
          <h4>🤖 AI Assistant</h4>
          <p className="aw-ai-desc">Describe your use case and let AI recommend the optimal configuration</p>

          {/* Auto-Detect Task */}
          <div className="aw-auto-detect-section">
            <button
              className="aw-btn aw-btn--detect aw-btn--full"
              onClick={handleAutoDetect}
              disabled={detecting}
            >
              {detecting ? 'Analyzing dataset...' : '🔍 Auto-Detect Task (AI)'}
            </button>
            {detectResult && (
              <div className="aw-detect-result">
                <div className="aw-detect-task-badge">
                  Detected: <strong>{detectResult.task.toUpperCase()}</strong>
                  <span className={`aw-badge aw-badge--${detectResult.confidence === 'high' ? 'green' : detectResult.confidence === 'medium' ? 'orange' : 'red'}`}>
                    {detectResult.confidence}
                  </span>
                </div>
                <p className="aw-detect-reasoning">{detectResult.reasoning}</p>
                <div className="aw-detect-source">
                  <span className={`aw-badge ${aiSourceDisplay(detectResult.source).badgeClass}`}>
                    {aiSourceDisplay(detectResult.source).label}
                  </span>
                </div>
                {detectResult.task === 'clustering' && onClusteringDetected && (
                  <button className="aw-btn aw-btn--primary aw-btn--full" onClick={onClusteringDetected}>
                    Start Clustering Pipeline →
                  </button>
                )}
              </div>
            )}
          </div>

          <hr className="aw-ai-divider" />

          {/* Suggested Use Cases — Classification & Regression only */}
          {suggestions.length > 0 && (
            <div className="aw-ai-suggestions">
              <label className="aw-ai-label">💡 Suggested Use Cases</label>
              <div className="aw-suggestion-filter">
                {(['classification', 'regression'] as const).map((task) => (
                  <button
                    key={task}
                    className={`aw-suggestion-filter-btn ${taskFilter === task ? 'aw-suggestion-filter-btn--active' : ''}`}
                    onClick={() => handleTaskSelect(task)}
                  >
                    {task}
                  </button>
                ))}
              </div>
              <div className="aw-suggestion-chips">
                {!taskFilter && (
                  <div className="aw-suggestion-empty">
                    Select a task above to view suggested use cases.
                  </div>
                )}
                {visibleSuggestions.map((s, i) => (
                  <button
                    key={i}
                    className={`aw-suggestion-chip ${useCase === s.use_case ? 'aw-suggestion-chip--active' : ''}`}
                    onClick={() => {
                      setUseCase(s.use_case);
                      if (s.ml_task === 'classification' || s.ml_task === 'regression') {
                        const t = s.ml_task as MLTask;
                        setTaskFilter(t);
                        onTaskSuggest(t);
                      }
                    }}
                  >
                    <span className="aw-suggestion-text">{s.use_case}</span>
                    <span className={`aw-suggestion-task aw-suggestion-task--${s.ml_task}`}>{s.ml_task}</span>
                  </button>
                ))}
                {taskFilter && visibleSuggestions.length === 0 && (
                  <div className="aw-suggestion-empty">
                    No suggestions for {taskFilter}. Try another filter.
                  </div>
                )}
              </div>
            </div>
          )}
          {suggestionsLoading && <p className="aw-ai-desc">Analyzing dataset for suggestions...</p>}

          <label className="aw-ai-label">Your Use Case</label>
          <textarea
            className="aw-ai-input"
            placeholder="e.g., I want to classify iris species"
            value={useCase}
            onChange={(e) => setUseCase(e.target.value)}
            rows={3}
          />
          <button className="aw-btn aw-btn--primary aw-btn--full" onClick={handleAIRecommend} disabled={aiLoading || !useCase.trim()}>
            {aiLoading ? 'Generating...' : '🤖 Generate Configuration'}
          </button>

          {aiResult && (
            <div className="aw-ai-result">
              <div className="aw-ai-result-header">
                <span>✨ AI Recommendation</span>
                <span className="aw-badge aw-badge--green">{aiResult.confidence}</span>
              </div>
              {aiResult.source ? (
                <div className="aw-ai-field">
                  <label>AI Source</label>
                  <span className={`aw-badge ${aiSourceDisplay(aiResult.source).badgeClass}`}>
                    {aiSourceDisplay(aiResult.source).label}
                  </span>
                </div>
              ) : null}
              <div className="aw-ai-field">
                <label>Target Column</label>
                <span className="aw-badge aw-badge--orange">{aiResult.target_column}</span>
              </div>
              <div className="aw-ai-field">
                <label>Features Selected</label>
                <span className="aw-ai-features-count">{aiResult.features.length} features</span>
              </div>
              <div className="aw-ai-reasoning">
                <h5>📋 Reasoning</h5>
                <p>{aiResult.reasoning}</p>
              </div>
              <div className="aw-ai-applied-msg">
                ℹ️ Configuration applied. Review the table and adjust if needed.
              </div>
            </div>
          )}
        </div>
      </div>
      )}
    </div>
  );
}
