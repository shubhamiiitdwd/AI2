import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { isAxiosError } from 'axios';
import {
  ScatterChart, Scatter, XAxis, YAxis, Tooltip, CartesianGrid,
  ResponsiveContainer, BarChart, Bar, Cell, LineChart, Line,
  Legend, PieChart, Pie,
} from 'recharts';
import * as api from '../api';
import type {
  ClusteringResultResponse, ElbowResponse,
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
  onBack?: () => void;
  /** Live run + logs vs full results dashboard (parent wizard steps 2 vs 3). */
  wizardView: 'execution' | 'results';
  /** Set when the run is started from Clustering configuration; progress via HTTP polling. */
  runId: string | null;
  /** Filled after the run completes; used when wizardView is results. */
  clusteringResult: ClusteringResultResponse | null;
  onViewResults: () => void;
  /** From results: go back to logs (parent sets wizard step to model search). */
  onBackToLogs: () => void;
}

interface ConfigUsed {
  algorithm: string;
  nClusters?: number;
  eps?: number;
  minSamples?: number;
  stability: boolean;
}

type ExecutionPhase = 'running' | 'done';

export default function StepClustering({
  datasetId,
  featureColumns,
  columns,
  onComplete,
  onBack,
  wizardView,
  runId,
  clusteringResult: clusteringResultProp,
  onViewResults,
  onBackToLogs,
}: Props) {
  const [executionPhase, setExecutionPhase] = useState<ExecutionPhase>('running');
  const [progress, setProgress] = useState(0);
  const [logMessages, setLogMessages] = useState<string[]>([]);
  const [configUsed, setConfigUsed] = useState<ConfigUsed | null>(null);

  const [fetchedResult, setFetchedResult] = useState<ClusteringResultResponse | null>(clusteringResultProp);
  const [elbow, setElbow] = useState<ElbowResponse | null>(null);

  const importanceChartData = useMemo(() => {
    const src = clusteringResultProp ?? fetchedResult;
    if (!src?.feature_importance?.length) return [];
    const raw = src.feature_importance.map((f) => ({
      feature: f.feature,
      importance: Math.max(0, Number(f.importance) || 0),
    }));
    const max = Math.max(...raw.map((r) => r.importance), 1e-12);
    return raw.map((r) => ({
      ...r,
      importancePct: (r.importance / max) * 100,
    }));
  }, [clusteringResultProp, fetchedResult]);

  useEffect(() => {
    setFetchedResult(clusteringResultProp);
  }, [clusteringResultProp]);

  useEffect(() => {
    const rid = clusteringResultProp?.run_id;
    if (rid && !elbow) {
      api.getElbowAnalysis(rid).then(setElbow).catch(() => {});
    }
  }, [clusteringResultProp, elbow]);

  useEffect(() => {
    if (wizardView === 'execution' && clusteringResultProp) {
      setExecutionPhase('done');
    }
  }, [wizardView, clusteringResultProp]);
  const [activeTab, setActiveTab] = useState<'overview' | 'scatter' | 'leaderboard' | 'elbow' | 'importance'>('overview');
  /** Overview pie: visual scale (radius). Scatter: domain zoom (1 = full data range). */
  const [pieVisualZoom, setPieVisualZoom] = useState(1);
  const [scatterDomainZoom, setScatterDomainZoom] = useState(1);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const retryCountRef = useRef(0);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const progressRef = useRef(0);
  const logEndRef = useRef<HTMLDivElement>(null);
  /** Avoid resetting log lines when the tracking effect re-runs for the same run (e.g. step 2 ↔ 3). */
  const trackingRunIdRef = useRef<string | null>(null);

  useEffect(() => {
    progressRef.current = progress;
  }, [progress]);

  const scrollToLogEnd = useCallback(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(scrollToLogEnd, [logMessages, scrollToLogEnd]);

  const clusteringFetchDoneRef = useRef<string | null>(null);
  const clusteringFetchInFlightRef = useRef(false);

  const fetchResults = useCallback(async (rid: string) => {
    if (clusteringFetchDoneRef.current === rid) return;
    if (clusteringFetchInFlightRef.current) return;
    clusteringFetchInFlightRef.current = true;
    setFetchError(null);
    try {
      const [res, elb] = await Promise.all([
        api.getClusteringResult(rid),
        api.getElbowAnalysis(rid),
      ]);
      clusteringFetchDoneRef.current = rid;
      retryCountRef.current = 0;
      setFetchedResult(res);
      setElbow(elb);
      setExecutionPhase('done');
      onComplete(res);
    } catch (err) {
      const is404 = isAxiosError(err) && err.response?.status === 404;
      // 404 = result/elbow not available yet (race right after 100%) or run unknown. Retry a few times for races only.
      retryCountRef.current += 1;
      const maxRetries = is404 ? 10 : 4;
      const delayMs = is404 ? 2000 : 3000;
      if (retryCountRef.current < maxRetries) {
        setTimeout(() => { fetchResults(rid); }, delayMs);
      } else {
        setFetchError(
          'Could not load clustering results. The backend may have restarted and lost the in-memory results. Please re-run clustering.'
        );
      }
    } finally {
      clusteringFetchInFlightRef.current = false;
    }
  }, [onComplete]);

  /** Always poll HTTP status so logs work even when WebSocket fails (proxy, Strict Mode remount, etc.). */
  const startHttpPolling = useCallback(
    (rid: string) => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
      pollRef.current = setInterval(async () => {
        try {
          const st = await api.getClusteringStatus(rid);
          const pct = st.progress_percent ?? 0;
          progressRef.current = pct;
          setProgress(pct);
          if (st.message) {
            const line = `[status] ${st.message}`;
            setLogMessages((prev) => (prev[prev.length - 1] === line ? prev : [...prev, line]));
          }
          if (pct >= 100) {
            if (pollRef.current) {
              clearInterval(pollRef.current);
              pollRef.current = null;
            }
            fetchResults(rid);
          }
        } catch {
          /* run may not be registered for a tick right after start */
        }
      }, 1200);
    },
    [fetchResults],
  );

  const clusteringResultRunId = clusteringResultProp?.run_id ?? null;

  useEffect(() => {
    if (!runId) return;
    if (wizardView !== 'execution') return;

    if (clusteringResultRunId === runId && clusteringResultProp) {
      setExecutionPhase('done');
      setFetchedResult(clusteringResultProp);
      setProgress(100);
      progressRef.current = 100;
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
      return;
    }

    const isNewRun = trackingRunIdRef.current !== runId;
    if (isNewRun) {
      trackingRunIdRef.current = runId;
      setExecutionPhase('running');
      setLogMessages(['[status] Tracking job (HTTP polling).']);
      setProgress(0);
      progressRef.current = 0;
      setFetchError(null);
      clusteringFetchDoneRef.current = null;
      retryCountRef.current = 0;
    }

    startHttpPolling(runId);

    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [wizardView, runId, clusteringResultRunId, clusteringResultProp, startHttpPolling]);

  const displayResult = clusteringResultProp ?? fetchedResult;

  const pcaScatterDomains = useMemo(() => {
    const pts = displayResult?.pca_points;
    if (!pts?.length) return null;
    const xs = pts.map((p) => p.x);
    const ys = pts.map((p) => p.y);
    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs);
    const yMin = Math.min(...ys);
    const yMax = Math.max(...ys);
    const xMid = (xMin + xMax) / 2;
    const yMid = (yMin + yMax) / 2;
    const xHalf = Math.max(xMax - xMin, 1e-9) / 2;
    const yHalf = Math.max(yMax - yMin, 1e-9) / 2;
    const z = Math.max(0.35, Math.min(4, scatterDomainZoom));
    return {
      x: [xMid - xHalf / z, xMid + xHalf / z] as [number, number],
      y: [yMid - yHalf / z, yMid + yHalf / z] as [number, number],
    };
  }, [displayResult?.pca_points, scatterDomainZoom]);

  // ── EXECUTION: RUNNING ──
  if (wizardView === 'execution' && executionPhase === 'running') {
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
              <div className="aw-review-banner aw-review-banner--error">
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
                  onClick={() => {
                    setFetchError(null);
                    onBack?.();
                  }}
                >
                  Re-run Clustering
                </button>
              </div>
            </div>
          )}
        </div>

        <aside className="aw-step-sidebar cl-exec-sidebar">
          <h4>Training log</h4>
          <p className="cl-exec-sidebar-note">Live messages from the clustering job. When progress reaches 100%, load the result payload or open the results step.</p>
        </aside>
      </div>
    );
  }

  // ── EXECUTION: DONE (logs + open results) ──
  if (wizardView === 'execution' && executionPhase === 'done') {
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
              <span>Clustering finished successfully. {displayResult?.total_candidates_tested} model configurations tested.</span>
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
            onClick={onViewResults}
          >
            View Clustering Results →
          </button>
        </div>

        <aside className="aw-step-sidebar cl-exec-sidebar">
          <h4>Training log</h4>
          <p className="cl-exec-sidebar-note">Review the run output above, then open the results dashboard.</p>
        </aside>
      </div>
    );
  }

  // ── RESULTS DASHBOARD (wizard step: Clustering results) ──
  if (wizardView === 'execution') {
    return <div className="aw-loading">Starting…</div>;
  }
  if (!displayResult) {
    return <div className="aw-loading">Loading results…</div>;
  }

  const result = displayResult;
  const bestMetrics = result.best_metrics;
  const pieData = result.cluster_summaries.map((s) => ({
    name: `Cluster ${s.cluster_id}`,
    value: s.size,
    percentage: s.percentage,
  }));

  return (
    <div className="aw-step-content cl-results-page cl-results-page--grid">
      <div className="cl-results-top">
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

        <button className="aw-back-btn" style={{ marginBottom: 12 }} type="button" onClick={onBackToLogs}>
          ← Back to model search &amp; logs
        </button>

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
                <div className="cl-chart-zoom-header">
                  <h4>Cluster distribution</h4>
                  <div className="cl-chart-zoom-controls" aria-label="Chart zoom">
                    <button
                      type="button"
                      className="cl-chart-zoom-btn"
                      onClick={() => setPieVisualZoom((z) => Math.min(2.2, z + 0.2))}
                    >
                      +
                    </button>
                    <button
                      type="button"
                      className="cl-chart-zoom-btn"
                      onClick={() => setPieVisualZoom((z) => Math.max(0.65, z - 0.2))}
                    >
                      −
                    </button>
                    <span className="cl-chart-zoom-readout">{Math.round(pieVisualZoom * 100)}%</span>
                  </div>
                </div>
                <div className="cl-chart-zoom-viewport">
                  <ResponsiveContainer width="100%" height={Math.min(460, 210 * pieVisualZoom)}>
                    <PieChart margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
                      <Pie
                        data={pieData}
                        dataKey="value"
                        nameKey="name"
                        cx="36%"
                        cy="50%"
                        innerRadius={0}
                        outerRadius={78 * pieVisualZoom}
                        paddingAngle={1}
                        label={false}
                        labelLine={false}
                      >
                        {pieData.map((_entry, index) => (
                          <Cell key={index} fill={CLUSTER_COLORS[index % CLUSTER_COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip
                        formatter={(value: number, _name: string, item) => [
                          `${value} rows`,
                          (item?.payload as { name?: string })?.name ?? '',
                        ]}
                      />
                      <Legend
                        verticalAlign="middle"
                        align="right"
                        layout="vertical"
                        wrapperStyle={{ fontSize: 15, lineHeight: 1.6, paddingLeft: 8 }}
                        formatter={(value, entry) => {
                          const p = entry.payload as { percentage?: number; name?: string };
                          return (
                            <span className="cl-legend-cluster">
                              {p?.name ?? value} — {p?.percentage ?? '—'}%
                            </span>
                          );
                        }}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
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

          </div>
        )}

        {/* Scatter plot tab */}
        {activeTab === 'scatter' && result.pca_points && pcaScatterDomains && (
          <div className="cl-scatter-container">
            <div className="cl-chart-zoom-header">
              <h4>PCA 2D projection</h4>
              <div className="cl-chart-zoom-controls" aria-label="Plot zoom">
                <button
                  type="button"
                  className="cl-chart-zoom-btn"
                  onClick={() => setScatterDomainZoom((z) => Math.min(4, z * 1.2))}
                >
                  Zoom in
                </button>
                <button
                  type="button"
                  className="cl-chart-zoom-btn"
                  onClick={() => setScatterDomainZoom((z) => Math.max(0.35, z / 1.2))}
                >
                  Zoom out
                </button>
                <button type="button" className="cl-chart-zoom-btn" onClick={() => setScatterDomainZoom(1)}>
                  Reset
                </button>
                <span className="cl-chart-zoom-readout">{Math.round(scatterDomainZoom * 100)}%</span>
              </div>
            </div>
            <div className="cl-chart-zoom-viewport cl-chart-zoom-viewport--scatter">
              <ResponsiveContainer width="100%" height={440}>
                <ScatterChart margin={{ top: 16, right: 16, bottom: 16, left: 16 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="x"
                    name="PC1"
                    type="number"
                    domain={pcaScatterDomains.x}
                    allowDataOverflow
                    tick={{ fontSize: 13 }}
                    label={{ value: 'PC1', position: 'insideBottom', offset: -4, style: { fontSize: 13 } }}
                  />
                  <YAxis
                    dataKey="y"
                    name="PC2"
                    type="number"
                    domain={pcaScatterDomains.y}
                    allowDataOverflow
                    tick={{ fontSize: 13 }}
                    label={{ value: 'PC2', angle: -90, position: 'insideLeft', style: { fontSize: 13 } }}
                  />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                  <Legend wrapperStyle={{ fontSize: 14, paddingTop: 12 }} />
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
            <p className="cl-elbow-hint">
              Heuristic K = <strong>{elbow.recommended_k}</strong> (best silhouette among KMeans fits for K = 2…10 only).
            </p>
            <p className="cl-elbow-explainer">
              The <strong>leaderboard</strong> picks the best model by a combined score across <em>all</em> algorithms (KMeans, GMM, DBSCAN) and hyperparameters (e.g. K=6 can beat K=5).
              The elbow chart is only a <em>KMeans-only</em> guide. So silhouette can peak at K=5 here while the global best model uses K=6 — that is expected, not a bug.
            </p>
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

        {/* Feature importance tab — ANOVA F-ratio share per numeric feature (vs. cluster id as class) */}
        {activeTab === 'importance' && (
          <div className="cl-importance-container">
            <h4>Feature importance (ANOVA F-score share)</h4>
            <p className="cl-importance-hint">
              Higher bars mean the feature varies more across clusters (numeric columns only).
            </p>
            {importanceChartData.length > 0 ? (
              <>
                <ResponsiveContainer width="100%" height={Math.max(280, importanceChartData.length * 40)}>
                  <BarChart
                    data={importanceChartData}
                    layout="vertical"
                    margin={{ top: 8, right: 24, left: 8, bottom: 8 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                    <XAxis type="number" domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
                    <YAxis
                      dataKey="feature"
                      type="category"
                      width={Math.min(200, 28 + Math.max(...importanceChartData.map((d) => String(d.feature).length)) * 7)}
                      tick={{ fontSize: 12 }}
                    />
                    <Tooltip
                      formatter={(value: number | string) => [`${Number(value).toFixed(1)}% (relative)`, 'Importance']}
                    />
                    <Bar dataKey="importancePct" fill="#6366f1" radius={[0, 4, 4, 0]} maxBarSize={28} isAnimationActive={false} />
                  </BarChart>
                </ResponsiveContainer>
                <table className="cl-importance-table">
                  <thead>
                    <tr>
                      <th>Feature</th>
                      <th>Share</th>
                    </tr>
                  </thead>
                  <tbody>
                    {importanceChartData.map((row) => (
                      <tr key={row.feature}>
                        <td>{row.feature}</td>
                        <td>{(row.importance * 100).toFixed(2)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </>
            ) : (
              <p className="cl-no-data">No feature importance available — need at least two clusters and one numeric feature column.</p>
            )}
          </div>
        )}

        </div>

      <aside className="aw-step-sidebar cl-results-sidebar">
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
          <hr className="aw-divider-soft" />
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

        <p className="cl-results-sidebar-note">Models were scored with Silhouette, Calinski–Harabasz, and Davies–Bouldin.</p>
      </aside>
      </div>
    </div>
  );
}
