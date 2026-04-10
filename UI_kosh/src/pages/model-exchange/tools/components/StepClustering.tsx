import { useState, useEffect, useRef, useCallback } from 'react';
import {
  ScatterChart, Scatter, XAxis, YAxis, Tooltip, CartesianGrid,
  ResponsiveContainer, BarChart, Bar, Cell, LineChart, Line,
  Legend, PieChart, Pie,
} from 'recharts';
import * as api from '../api';
import type {
  ClusteringResultResponse, ClusteringStartRequest, ElbowResponse,
  ColumnInfo,
} from '../types';

const CLUSTER_COLORS = [
  '#6366f1', '#f59e0b', '#10b981', '#ef4444', '#3b82f6',
  '#ec4899', '#8b5cf6', '#14b8a6', '#f97316', '#06b6d4',
];

interface Props {
  datasetId: string;
  featureColumns: string[];
  columns: ColumnInfo[];
  onComplete: (result: ClusteringResultResponse) => void;
  onContinueToSupervisedML?: () => void;
  onBack?: () => void;
  existingResult?: ClusteringResultResponse | null;
}

type Phase = 'config' | 'running' | 'done' | 'results';

interface ConfigUsed {
  algorithm: string;
  nClusters?: number;
  eps?: number;
  minSamples?: number;
  stability: boolean;
}

export default function StepClustering({ datasetId, featureColumns, columns, onComplete, onContinueToSupervisedML, onBack, existingResult }: Props) {
  const [phase, setPhase] = useState<Phase>(existingResult ? 'results' : 'config');
  const [algorithm, setAlgorithm] = useState<string>('');
  const [nClusters, setNClusters] = useState<number | undefined>(undefined);
  const [eps, setEps] = useState<number | undefined>(undefined);
  const [minSamples, setMinSamples] = useState<number | undefined>(undefined);
  const [runStability, setRunStability] = useState(true);

  const [runId, setRunId] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [logMessages, setLogMessages] = useState<string[]>([]);
  const [configUsed, setConfigUsed] = useState<ConfigUsed | null>(null);

  const [result, setResult] = useState<ClusteringResultResponse | null>(existingResult || null);
  const [elbow, setElbow] = useState<ElbowResponse | null>(null);

  useEffect(() => {
    if (existingResult && !elbow) {
      api.getElbowAnalysis(existingResult.run_id).then(setElbow).catch(() => {});
    }
  }, [existingResult]);
  const [activeTab, setActiveTab] = useState<'overview' | 'scatter' | 'leaderboard' | 'elbow' | 'importance'>('overview');
  const [fetchError, setFetchError] = useState<string | null>(null);
  const retryCountRef = useRef(0);

  const wsRef = useRef<WebSocket | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);

  const scrollToLogEnd = useCallback(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(scrollToLogEnd, [logMessages, scrollToLogEnd]);

  const handleStart = async () => {
    setPhase('running');
    setLogMessages([]);
    setProgress(0);
    setConfigUsed({
      algorithm: algorithm || 'auto',
      nClusters,
      eps: algorithm === 'dbscan' ? eps : undefined,
      minSamples: algorithm === 'dbscan' ? minSamples : undefined,
      stability: runStability,
    });

    try {
      const req: ClusteringStartRequest = {
        dataset_id: datasetId,
        feature_columns: featureColumns,
        algorithm: algorithm || undefined,
        n_clusters: nClusters,
        eps: algorithm === 'dbscan' ? eps : undefined,
        min_samples: algorithm === 'dbscan' ? minSamples : undefined,
        run_stability_check: runStability,
      };
      const resp = await api.startClustering(req);
      setRunId(resp.run_id);
      connectWs(resp.run_id);
    } catch (e) {
      setLogMessages((prev) => [...prev, `Error: ${e}`]);
      setPhase('config');
    }
  };

  const connectWs = (rid: string) => {
    const url = api.getClusteringWsUrl(rid);
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        if (msg.progress !== undefined) setProgress(msg.progress);
        if (msg.message) {
          const ts = msg.timestamp ? new Date(msg.timestamp).toLocaleTimeString() : '';
          setLogMessages((prev) => [...prev, `[${ts}] ${msg.message}`]);
        }
        if (msg.progress === 100) {
          fetchResults(rid);
        }
      } catch { /* ignore */ }
    };

    ws.onerror = () => {
      setLogMessages((prev) => [...prev, '[WS] Connection error']);
    };
    ws.onclose = () => {
      if (progress < 100) {
        setTimeout(() => fetchResults(rid), 2000);
      }
    };
  };

  const fetchResults = async (rid: string) => {
    setFetchError(null);
    try {
      const [res, elb] = await Promise.all([
        api.getClusteringResult(rid),
        api.getElbowAnalysis(rid),
      ]);
      retryCountRef.current = 0;
      setResult(res);
      setElbow(elb);
      setPhase('done');
      onComplete(res);
    } catch (err) {
      retryCountRef.current += 1;
      if (retryCountRef.current < 5) {
        setTimeout(() => fetchResults(rid), 3000);
      } else {
        setFetchError(
          'Could not load clustering results. The backend may have restarted and lost the in-memory results. Please re-run clustering.'
        );
      }
    }
  };

  useEffect(() => {
    return () => { wsRef.current?.close(); };
  }, []);

  const numericCols = columns.filter((c) =>
    c.dtype === 'int64' || c.dtype === 'float64' || c.dtype === 'int32' || c.dtype === 'float32'
  );

  // ── CONFIG PHASE ──
  if (phase === 'config') {
    return (
      <div className="aw-step-content">
        <div className="aw-step-main">
          {onBack && (
            <button className="aw-back-btn" onClick={onBack}>← Back to Configure Data</button>
          )}
          <h3 className="aw-section-title">Clustering Configuration</h3>

          <div className="cl-config-section">
            <label className="cl-label">Algorithm (leave empty for Auto)</label>
            <select className="cl-select" value={algorithm} onChange={(e) => setAlgorithm(e.target.value)}>
              <option value="">Auto (try all)</option>
              <option value="kmeans">KMeans</option>
              <option value="gmm">GMM (Gaussian Mixture)</option>
              <option value="dbscan">DBSCAN</option>
            </select>
          </div>

          {(algorithm === 'kmeans' || algorithm === 'gmm' || algorithm === '') && (
            <div className="cl-config-section">
              <label className="cl-label">Number of Clusters (optional, 2-20)</label>
              <input
                className="cl-input"
                type="number" min={2} max={20}
                value={nClusters ?? ''}
                onChange={(e) => setNClusters(e.target.value ? Number(e.target.value) : undefined)}
                placeholder="Auto (grid search 2-10)"
              />
            </div>
          )}

          {algorithm === 'dbscan' && (
            <>
              <div className="cl-config-section">
                <label className="cl-label">DBSCAN eps (optional)</label>
                <input
                  className="cl-input"
                  type="number" step="0.1" min={0.01}
                  value={eps ?? ''}
                  onChange={(e) => setEps(e.target.value ? Number(e.target.value) : undefined)}
                  placeholder="Auto grid search"
                />
              </div>
              <div className="cl-config-section">
                <label className="cl-label">DBSCAN min_samples (optional)</label>
                <input
                  className="cl-input"
                  type="number" min={1}
                  value={minSamples ?? ''}
                  onChange={(e) => setMinSamples(e.target.value ? Number(e.target.value) : undefined)}
                  placeholder="Auto grid search"
                />
              </div>
            </>
          )}

          <div className="cl-config-section">
            <label className="cl-toggle-row">
              <input type="checkbox" checked={runStability} onChange={(e) => setRunStability(e.target.checked)} />
              <span>Run Stability Check (5 runs, compare ARI)</span>
            </label>
          </div>

          <button className="aw-btn aw-btn--primary aw-btn--full" onClick={handleStart}>
            Start Clustering →
          </button>

          <div className="cl-info-note">
            <strong>Features selected:</strong> {featureColumns.length} columns.
            Numeric: {numericCols.length}. Categorical will be one-hot encoded.
          </div>
        </div>

        <div className="aw-step-sidebar">
          <h4>Clustering Pipeline:</h4>
          <ul className="aw-workflow-list">
            <li className="aw-workflow-active">Configure algorithm & params</li>
            <li>Scale & encode features</li>
            <li>Grid search all candidates</li>
            <li>Score & rank models</li>
            <li>Stability check</li>
            <li>View results & visualizations</li>
          </ul>
        </div>
      </div>
    );
  }

  // ── RUNNING PHASE ──
  if (phase === 'running') {
    return (
      <div className="aw-step-content">
        <div className="aw-step-main">
          <h3 className="aw-section-title">
            {progress >= 100 ? 'Clustering Complete! Loading results...' : 'Clustering in Progress...'}
          </h3>

          <div className="cl-progress-bar-container">
            <div className="cl-progress-bar" style={{ width: `${progress}%` }} />
            <span className="cl-progress-label">{progress}%</span>
          </div>

          <div className="cl-live-log">
            {logMessages.map((msg, i) => (
              <div key={i} className="cl-log-line">{msg}</div>
            ))}
            <div ref={logEndRef} />
          </div>

          {progress >= 100 && runId && !fetchError && (
            <button
              className="aw-btn aw-btn--primary aw-btn--full"
              style={{ marginTop: 16 }}
              onClick={() => { retryCountRef.current = 0; fetchResults(runId); }}
            >
              View Results →
            </button>
          )}

          {fetchError && (
            <div style={{ marginTop: 16 }}>
              <div className="aw-review-banner" style={{ background: '#fef2f2', border: '1px solid #fecaca', color: '#dc2626' }}>
                {fetchError}
              </div>
              <div style={{ display: 'flex', gap: 12, marginTop: 12 }}>
                <button
                  className="aw-btn aw-btn--primary"
                  onClick={() => { retryCountRef.current = 0; fetchResults(runId!); }}
                >
                  Retry Loading
                </button>
                <button
                  className="aw-btn aw-btn--outline"
                  onClick={() => { setPhase('config'); setFetchError(null); setProgress(0); }}
                >
                  Re-run Clustering
                </button>
              </div>
            </div>
          )}
        </div>

        <div className="aw-step-sidebar">
          <h4>Clustering Pipeline:</h4>
          <ul className="aw-workflow-list">
            <li>Configure algorithm & params</li>
            <li className="aw-workflow-active">Training & evaluating models...</li>
            <li>Score & rank models</li>
            <li>Stability check</li>
            <li>View results & visualizations</li>
          </ul>
        </div>
      </div>
    );
  }

  // ── DONE PHASE (clustering finished, user reviews logs before viewing results) ──
  if (phase === 'done') {
    return (
      <div className="aw-step-content">
        <div className="aw-step-main">
          <h3 className="aw-section-title">Clustering Complete!</h3>

          <div className="cl-progress-bar-container">
            <div className="cl-progress-bar" style={{ width: '100%' }} />
            <span className="cl-progress-label">100%</span>
          </div>

          <div style={{ marginBottom: 16 }}>
            <div className="aw-review-banner">
              <span>Clustering finished successfully. {result?.total_candidates_tested} model configurations tested.</span>
            </div>
          </div>

          <div className="cl-live-log">
            {logMessages.map((msg, i) => (
              <div key={i} className="cl-log-line">{msg}</div>
            ))}
            <div ref={logEndRef} />
          </div>

          <button
            className="aw-btn aw-btn--primary aw-btn--full"
            style={{ marginTop: 16 }}
            onClick={() => setPhase('results')}
          >
            View Clustering Results →
          </button>
        </div>

        <div className="aw-step-sidebar">
          <h4>Clustering Pipeline:</h4>
          <ul className="aw-workflow-list">
            <li>Configure algorithm & params</li>
            <li>Scale & encode features</li>
            <li>Grid search all candidates</li>
            <li>Score & rank models</li>
            <li>Stability check</li>
            <li className="aw-workflow-active">View results & visualizations</li>
          </ul>
        </div>
      </div>
    );
  }

  // ── RESULTS PHASE ──
  if (!result) return <div className="aw-loading">Loading results...</div>;

  const bestMetrics = result.best_metrics;
  const pieData = result.cluster_summaries.map((s) => ({
    name: `Cluster ${s.cluster_id}`,
    value: s.size,
    percentage: s.percentage,
  }));

  return (
    <div className="aw-step-content cl-results-page">
      <div className="aw-step-main cl-results-main">
        <h3 className="aw-section-title">Clustering Results</h3>

        {/* Best model banner */}
        <div className="cl-best-banner">
          <div className="cl-best-algo">
            <span className="cl-best-badge">Best</span>
            <strong>{result.best_algorithm.toUpperCase()}</strong>
            <span className="cl-best-params">
              {Object.entries(result.best_params).map(([k, v]) => `${k}=${v}`).join(', ')}
            </span>
          </div>
          <div className="cl-metrics-row">
            <div className="cl-metric-card">
              <span className="cl-metric-value">{bestMetrics.silhouette_score.toFixed(4)}</span>
              <span className="cl-metric-label">Silhouette</span>
            </div>
            <div className="cl-metric-card">
              <span className="cl-metric-value">{bestMetrics.calinski_harabasz.toFixed(1)}</span>
              <span className="cl-metric-label">Calinski-Harabasz</span>
            </div>
            <div className="cl-metric-card">
              <span className="cl-metric-value">{bestMetrics.davies_bouldin.toFixed(4)}</span>
              <span className="cl-metric-label">Davies-Bouldin</span>
            </div>
            <div className="cl-metric-card">
              <span className="cl-metric-value">{bestMetrics.composite_score.toFixed(4)}</span>
              <span className="cl-metric-label">Composite</span>
            </div>
            <div className="cl-metric-card">
              <span className="cl-metric-value">{bestMetrics.n_clusters}</span>
              <span className="cl-metric-label">Clusters</span>
            </div>
          </div>
          {result.stability && (
            <div className={`cl-stability-badge ${result.stability.is_stable ? 'cl-stable' : 'cl-unstable'}`}>
              {result.stability.is_stable ? '✔ Stable' : '⚠ Unstable'} (ARI: {result.stability.avg_ari.toFixed(3)}, {result.stability.n_runs} runs)
            </div>
          )}
        </div>

        {/* Navigation */}
        {logMessages.length > 0 && (
          <button className="aw-back-btn" style={{ marginBottom: 12 }} onClick={() => setPhase('done')}>
            ← Back to Logs
          </button>
        )}

        {/* Tabs */}
        <div className="cl-tabs">
          {(['overview', 'scatter', 'leaderboard', 'elbow', 'importance'] as const).map((tab) => (
            <button
              key={tab}
              className={`cl-tab ${activeTab === tab ? 'cl-tab--active' : ''}`}
              onClick={() => setActiveTab(tab)}
            >
              {tab === 'overview' && 'Overview'}
              {tab === 'scatter' && 'Scatter Plot'}
              {tab === 'leaderboard' && `Leaderboard (${result.leaderboard.length})`}
              {tab === 'elbow' && 'Elbow Analysis'}
              {tab === 'importance' && 'Feature Importance'}
            </button>
          ))}
        </div>

        {/* Overview tab */}
        {activeTab === 'overview' && (
          <div className="cl-overview">
            <div className="cl-overview-grid">
              <div className="cl-overview-card">
                <h4>Cluster Distribution</h4>
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={pieData}
                      dataKey="value"
                      nameKey="name"
                      cx="50%" cy="50%"
                      outerRadius={90}
                      label={({ name, percentage }) => `${name} (${percentage}%)`}
                    >
                      {pieData.map((_entry, index) => (
                        <Cell key={index} fill={CLUSTER_COLORS[index % CLUSTER_COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              <div className="cl-overview-card">
                <h4>Cluster Details</h4>
                <table className="cl-summary-table">
                  <thead>
                    <tr>
                      <th>Cluster</th>
                      <th>Size</th>
                      <th>%</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.cluster_summaries.map((s) => (
                      <tr key={s.cluster_id}>
                        <td>
                          <span className="cl-dot" style={{ background: CLUSTER_COLORS[s.cluster_id % CLUSTER_COLORS.length] }} />
                          Cluster {s.cluster_id}
                        </td>
                        <td>{s.size}</td>
                        <td>{s.percentage}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="cl-info-bar">
              <span>Tested <strong>{result.total_candidates_tested}</strong> model configurations</span>
              {bestMetrics.n_noise_points > 0 && (
                <span> | <strong>{bestMetrics.n_noise_points}</strong> noise points (DBSCAN)</span>
              )}
            </div>

            {onContinueToSupervisedML && (
              <button className="aw-btn aw-btn--primary aw-btn--full" style={{ marginTop: 16 }} onClick={onContinueToSupervisedML}>
                Continue to Supervised ML (with cluster labels) →
              </button>
            )}
          </div>
        )}

        {/* Scatter plot tab */}
        {activeTab === 'scatter' && result.pca_points && (
          <div className="cl-scatter-container">
            <h4>PCA 2D Projection</h4>
            <ResponsiveContainer width="100%" height={400}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="x" name="PC1" type="number" />
                <YAxis dataKey="y" name="PC2" type="number" />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Legend />
                {Array.from(new Set(result.pca_points.map((p) => p.cluster))).sort().map((cid) => (
                  <Scatter
                    key={cid}
                    name={cid === -1 ? 'Noise' : `Cluster ${cid}`}
                    data={result.pca_points!.filter((p) => p.cluster === cid)}
                    fill={cid === -1 ? '#999' : CLUSTER_COLORS[cid % CLUSTER_COLORS.length]}
                  />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Leaderboard tab */}
        {activeTab === 'leaderboard' && (
          <div className="cl-leaderboard-container">
            <table className="cl-leaderboard-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Algorithm</th>
                  <th>Params</th>
                  <th>Clusters</th>
                  <th>Silhouette</th>
                  <th>CH</th>
                  <th>DB</th>
                  <th>Score</th>
                </tr>
              </thead>
              <tbody>
                {result.leaderboard.map((m) => (
                  <tr key={m.rank} className={m.is_best ? 'cl-row-best' : ''}>
                    <td>{m.is_best ? '🏆' : `#${m.rank}`}</td>
                    <td>{m.algorithm.toUpperCase()}</td>
                    <td className="cl-params-cell">
                      {Object.entries(m.params).map(([k, v]) => `${k}=${v}`).join(', ')}
                    </td>
                    <td>{m.n_clusters}</td>
                    <td>{m.silhouette.toFixed(4)}</td>
                    <td>{m.calinski_harabasz.toFixed(1)}</td>
                    <td>{m.davies_bouldin.toFixed(4)}</td>
                    <td><strong>{m.composite_score.toFixed(4)}</strong></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Elbow tab */}
        {activeTab === 'elbow' && elbow && (
          <div className="cl-elbow-container">
            <h4>Elbow Analysis (KMeans)</h4>
            <p className="cl-elbow-hint">Recommended K = <strong>{elbow.recommended_k}</strong> (highest silhouette)</p>
            <div className="cl-elbow-charts">
              <div className="cl-elbow-chart">
                <h5>Inertia vs K</h5>
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={elbow.data}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="k" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="inertia" stroke="#6366f1" strokeWidth={2} dot />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="cl-elbow-chart">
                <h5>Silhouette vs K</h5>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={elbow.data}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="k" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="silhouette" fill="#10b981">
                      {elbow.data.map((entry, index) => (
                        <Cell
                          key={index}
                          fill={entry.k === elbow.recommended_k ? '#f59e0b' : '#10b981'}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}

        {/* Feature importance tab */}
        {activeTab === 'importance' && (
          <div className="cl-importance-container">
            <h4>Feature Importance for Clustering</h4>
            {result.feature_importance.length > 0 ? (
              <ResponsiveContainer width="100%" height={Math.max(250, result.feature_importance.length * 35)}>
                <BarChart data={result.feature_importance} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis dataKey="feature" type="category" width={150} />
                  <Tooltip />
                  <Bar dataKey="importance" fill="#6366f1" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p className="cl-no-data">No feature importance data available (may require numeric features).</p>
            )}
          </div>
        )}

      </div>

      <div className="aw-step-sidebar">
        <h4>Configuration Used</h4>
        <div className="cl-config-review">
          {configUsed ? (
            <>
              <div className="cl-config-review-row">
                <span className="cl-config-review-label">Algorithm</span>
                <span className="cl-config-review-value">
                  {configUsed.algorithm === 'auto' ? 'Auto (all)' : configUsed.algorithm.toUpperCase()}
                </span>
              </div>
              {configUsed.nClusters && (
                <div className="cl-config-review-row">
                  <span className="cl-config-review-label">Requested Clusters</span>
                  <span className="cl-config-review-value">{configUsed.nClusters}</span>
                </div>
              )}
              {configUsed.eps && (
                <div className="cl-config-review-row">
                  <span className="cl-config-review-label">DBSCAN eps</span>
                  <span className="cl-config-review-value">{configUsed.eps}</span>
                </div>
              )}
              {configUsed.minSamples && (
                <div className="cl-config-review-row">
                  <span className="cl-config-review-label">DBSCAN min_samples</span>
                  <span className="cl-config-review-value">{configUsed.minSamples}</span>
                </div>
              )}
              <div className="cl-config-review-row">
                <span className="cl-config-review-label">Stability Check</span>
                <span className="cl-config-review-value">{configUsed.stability ? 'Enabled' : 'Disabled'}</span>
              </div>
            </>
          ) : (
            <div className="cl-config-review-row">
              <span className="cl-config-review-label">Algorithm</span>
              <span className="cl-config-review-value">{result.best_algorithm.toUpperCase()}</span>
            </div>
          )}
          <hr style={{ margin: '12px 0', border: 'none', borderTop: '1px solid #e5e7eb' }} />
          <div className="cl-config-review-row">
            <span className="cl-config-review-label">Best Model</span>
            <span className="cl-config-review-value">{result.best_algorithm.toUpperCase()}</span>
          </div>
          <div className="cl-config-review-row">
            <span className="cl-config-review-label">Best Params</span>
            <span className="cl-config-review-value" style={{ fontSize: '0.82rem' }}>
              {Object.entries(result.best_params).map(([k, v]) => `${k}=${v}`).join(', ')}
            </span>
          </div>
          <div className="cl-config-review-row">
            <span className="cl-config-review-label">Models Tested</span>
            <span className="cl-config-review-value">{result.total_candidates_tested}</span>
          </div>
          <div className="cl-config-review-row">
            <span className="cl-config-review-label">Features Used</span>
            <span className="cl-config-review-value">{featureColumns.length}</span>
          </div>
        </div>

        <h4 style={{ marginTop: 20 }}>Clustering Pipeline:</h4>
        <ul className="aw-workflow-list">
          <li>Configure algorithm & params</li>
          <li>Scale & encode features</li>
          <li>Grid search all candidates</li>
          <li>Score & rank models</li>
          <li>Stability check</li>
          <li className="aw-workflow-active">View results & visualizations</li>
        </ul>
      </div>
    </div>
  );
}
