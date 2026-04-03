import { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import type { LeaderboardResponse, FeatureImportanceResponse } from '../types';
import * as api from '../api';

interface Props {
  runId: string;
}

interface AISummary {
  executive_summary: string;
  key_insights: string[];
  recommendations: string[];
  real_world_example: string;
}

interface GainsLiftRow {
  group: number;
  cumulative_data_pct: number;
  lift: number;
  gain_pct: number;
}

interface PredictionResult {
  model_id: string;
  prediction: string | null;
  class_probabilities?: Record<string, number>;
  error?: string;
}

export default function StepResults({ runId }: Props) {
  const [leaderboard, setLeaderboard] = useState<LeaderboardResponse | null>(null);
  const [featureImp, setFeatureImp] = useState<FeatureImportanceResponse | null>(null);
  const [bestModelData, setBestModelData] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'leaderboard' | 'performance' | 'predict'>('leaderboard');
  const [sortMetric, setSortMetric] = useState('');
  const [showMetricDropdown, setShowMetricDropdown] = useState(false);

  const [aiSummary, setAiSummary] = useState<AISummary | null>(null);
  const [aiLoading, setAiLoading] = useState(false);
  const [gainsLift, setGainsLift] = useState<GainsLiftRow[]>([]);

  const [featureValues, setFeatureValues] = useState<Record<string, string>>({});
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [predicting, setPredicting] = useState(false);
  const [showPasteCSV, setShowPasteCSV] = useState(false);
  const [csvText, setCsvText] = useState('');

  useEffect(() => {
    const load = async () => {
      try {
        const [lb, fi, bm] = await Promise.all([
          api.getLeaderboard(runId),
          api.getFeatureImportance(runId).catch(() => null),
          api.getBestModel(runId).catch(() => null),
        ]);
        setLeaderboard(lb);
        setFeatureImp(fi);
        setBestModelData(bm);

        if (lb && lb.models[0]) {
          const firstMetric = Object.keys(lb.models[0].metrics).find(k => lb.models[0].metrics[k] != null);
          if (firstMetric) setSortMetric(firstMetric);
        }

        const featureCols = (bm as Record<string, unknown>)?.feature_columns as string[] | undefined;
        if (featureCols) {
          const initial: Record<string, string> = {};
          featureCols.forEach(c => { initial[c] = ''; });
          setFeatureValues(initial);
        }

        api.getGainsLift(runId).then(gl => setGainsLift(gl.rows || [])).catch(() => {});
      } catch { /* ignore */ }
      finally { setLoading(false); }
    };
    load();
  }, [runId]);

  const handleGenerateSummary = async () => {
    setAiLoading(true);
    try {
      const summary = await api.generateAISummary(runId);
      setAiSummary(summary);
    } catch { /* ignore */ }
    finally { setAiLoading(false); }
  };

  const handleRandomRow = async () => {
    try {
      const data = await api.getRandomRow(runId);
      const fv = data.feature_values || {};
      const mapped: Record<string, string> = {};
      Object.keys(fv).forEach(k => { mapped[k] = String(fv[k]); });
      setFeatureValues(mapped);
    } catch { /* ignore */ }
  };

  const handlePasteCSV = () => {
    const cols = Object.keys(featureValues);
    const sep = csvText.includes('\t') ? '\t' : ',';
    const vals = csvText.trim().split(sep);
    const mapped: Record<string, string> = {};
    cols.forEach((c, i) => { mapped[c] = vals[i]?.trim() || ''; });
    setFeatureValues(mapped);
    setShowPasteCSV(false);
    setCsvText('');
  };

  const handlePredict = async () => {
    setPredicting(true);
    try {
      const numericValues: Record<string, unknown> = {};
      Object.entries(featureValues).forEach(([k, v]) => {
        const n = Number(v);
        numericValues[k] = isNaN(n) ? v : n;
      });
      const data = await api.predict(runId, numericValues);
      setPredictions(data.predictions || []);
    } catch { /* ignore */ }
    finally { setPredicting(false); }
  };

  const handleReset = () => {
    const reset: Record<string, string> = {};
    Object.keys(featureValues).forEach(k => { reset[k] = ''; });
    setFeatureValues(reset);
    setPredictions([]);
  };

  if (loading) return <div className="aw-loading">Loading results...</div>;
  if (!leaderboard) return <div className="aw-error">No results available yet.</div>;

  const allMetrics = (bestModelData as Record<string, unknown>)?.all_metrics as Record<string, number> || {};
  const bestModel = (bestModelData as Record<string, unknown>)?.best_model as Record<string, unknown> || {};
  const mlTask = (bestModelData as Record<string, unknown>)?.ml_task as string || leaderboard.ml_task;
  const targetCol = (bestModelData as Record<string, unknown>)?.target_column as string || '';

  const bestAlgo = (bestModel?.algorithm as string)
    || (leaderboard.models.find(m => m.is_best)?.algorithm)
    || (leaderboard.models[0]?.algorithm)
    || '';

  const metricKeys = leaderboard.models[0]
    ? Object.keys(leaderboard.models[0].metrics).filter(k => leaderboard.models[0].metrics[k] != null)
    : [];

  const sortedModels = [...leaderboard.models].sort((a, b) => {
    const av = a.metrics[sortMetric] ?? Infinity;
    const bv = b.metrics[sortMetric] ?? Infinity;
    return (Number(av) || 0) - (Number(bv) || 0);
  });

  const primaryMetric = leaderboard.primary_metric || metricKeys[0] || 'mean_per_class_error';
  const bestMetrics = bestModel?.metrics as Record<string, number> | undefined;
  const bestPrimaryValue = allMetrics[primaryMetric]
    ?? bestMetrics?.[primaryMetric]
    ?? leaderboard.models.find(m => m.is_best)?.metrics[primaryMetric]
    ?? leaderboard.models[0]?.metrics[primaryMetric]
    ?? null;

  const COLORS = ['#e67e22', '#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#1abc9c'];

  const comparisonData = leaderboard.models.slice(0, 10).map((m) => {
    const row: Record<string, unknown> = { name: `#${m.rank} ${m.algorithm}` };
    metricKeys.slice(0, 2).forEach(k => {
      const raw = m.metrics[k];
      if (raw == null) { row[k] = 0; return; }
      let val = Number(raw);
      if (isNaN(val)) { row[k] = 0; return; }
      row[k] = Number(val.toFixed(4));
    });
    return row;
  });

  const METRIC_LABELS: Record<string, string> = {
    auc: 'AUC', accuracy: 'ACCURACY', logloss: 'LOG LOSS', rmse: 'RMSE',
    mse: 'MSE', mae: 'MAE', r2: 'R²', f1: 'F1', precision: 'PRECISION',
    recall: 'RECALL', mean_per_class_error: 'MEAN_PER_CLASS_ERROR', rmsle: 'RMSLE',
  };

  return (
    <div className="aw-step-content">
      <div className="aw-step-main aw-step-main--wide">
        {/* Best Model Card */}
        <div className="aw-best-model-card">
          <div className="aw-best-model-info">
            <span className="aw-best-model-label">Best Model</span>
            <span className="aw-best-model-name">{bestAlgo}</span>
          </div>
          <div className="aw-best-model-metric">
            <span className="aw-best-model-metric-label">{(primaryMetric || '').toUpperCase()}</span>
            <span className="aw-best-model-metric-value">
              {bestPrimaryValue != null ? Number(bestPrimaryValue).toFixed(3) : '-'}
            </span>
          </div>
        </div>

        {/* Performance Metrics */}
        <div className="aw-perf-section">
          <h3 className="aw-perf-title">📈 Performance Metrics</h3>
          <div className="aw-perf-cards">
            {Object.entries(allMetrics).map(([k, v]) => (
              <div key={k} className="aw-perf-card">
                <span className="aw-perf-card-label">{METRIC_LABELS[k] || k.toUpperCase()}</span>
                <span className="aw-perf-card-value">
                  {v != null ? (k === 'accuracy' ? `${(Number(v) * 100).toFixed(1)}%` : Number(v).toFixed(4)) : '-'}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Model Leaderboard Header */}
        <div className="aw-lb-header">
          <div>
            <h3>Model Leaderboard</h3>
            <p className="aw-lb-subtitle">Top performing models ranked by {(sortMetric || '').toUpperCase()}</p>
          </div>
        </div>

        {/* Leaderboard Table */}
        <div className="aw-leaderboard">
          <table className="aw-lb-table">
            <thead>
              <tr>
                <th>Model ID</th>
                {metricKeys.map(k => (
                  <th key={k} className="aw-lb-th-sortable" onClick={() => setSortMetric(k)}>
                    {(METRIC_LABELS[k] || k).toUpperCase()}
                    {sortMetric === k && ' ▾'}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sortedModels.map((m, idx) => (
                <tr key={m.model_id} className={idx === 0 ? 'aw-lb-row--best' : ''}>
                  <td className="aw-lb-model-id">
                    {idx === 0 && <span className="aw-badge aw-badge--green">Best</span>}
                    {' '}{m.model_id}
                  </td>
                  {metricKeys.map(k => (
                    <td key={k}><strong>{m.metrics[k] != null ? Number(m.metrics[k]).toFixed(4) : '-'}</strong></td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Gains/Lift Analysis */}
        {gainsLift.length > 0 && (
          <div className="aw-gains-section">
            <h3>Gains/Lift Analysis</h3>
            <p className="aw-lb-subtitle">Model performance analysis by percentile groups</p>
            <table className="aw-lb-table">
              <thead>
                <tr><th>Group</th><th>Cumulative Data %</th><th>Lift</th><th>Gain %</th></tr>
              </thead>
              <tbody>
                {gainsLift.map(row => (
                  <tr key={row.group}>
                    <td>{row.group}</td>
                    <td>{row.cumulative_data_pct}%</td>
                    <td><strong>{row.lift}x</strong></td>
                    <td>{row.gain_pct}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Tabs: Leaderboard | Performance Chart | Predict */}
        <div className="aw-results-tabs">
          <button className={`aw-tab ${activeTab === 'leaderboard' ? 'aw-tab--active' : ''}`} onClick={() => setActiveTab('leaderboard')}>Leaderboard</button>
          <button className={`aw-tab ${activeTab === 'performance' ? 'aw-tab--active' : ''}`} onClick={() => setActiveTab('performance')}>Performance Chart</button>
          <button className={`aw-tab ${activeTab === 'predict' ? 'aw-tab--active' : ''}`} onClick={() => setActiveTab('predict')}>Predict</button>
        </div>

        {activeTab === 'leaderboard' && (
          <div className="aw-card-leaderboard">
            <div className="aw-card-lb-header">
              <span>All Models ({leaderboard.models.length})</span>
              <div className="aw-metric-dropdown-wrap">
                <button className="aw-metric-dropdown-btn" onClick={() => setShowMetricDropdown(!showMetricDropdown)}>
                  {(METRIC_LABELS[sortMetric] || sortMetric || '').toUpperCase()} ▾
                </button>
                {showMetricDropdown && (
                  <div className="aw-metric-dropdown">
                    {metricKeys.map(k => (
                      <button key={k} className={`aw-metric-option ${sortMetric === k ? 'aw-metric-option--active' : ''}`}
                        onClick={() => { setSortMetric(k); setShowMetricDropdown(false); }}>
                        {METRIC_LABELS[k] || k}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
            {sortedModels.map((m, i) => (
              <div key={m.model_id} className={`aw-model-card-lb ${i === 0 ? 'aw-model-card-lb--best' : ''}`}>
                <span className="aw-model-rank">#{i + 1}</span>
                <span className="aw-model-card-name">{m.model_id}</span>
                <span className="aw-model-card-metric">{m.metrics[sortMetric] != null ? Number(m.metrics[sortMetric]).toFixed(4) : '-'}</span>
                {i === 0 && <span className="aw-badge aw-badge--green">Best</span>}
              </div>
            ))}
            <a href={api.getExportUrl(runId, 'csv')} className="aw-btn aw-btn--primary aw-btn--full" download>
              📥 Download Leaderboard Report
            </a>
          </div>
        )}

        {activeTab === 'performance' && comparisonData.length > 0 && (
          <div className="aw-chart-container">
            <h3>Performance Comparison</h3>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={comparisonData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                {metricKeys.slice(0, 2).map((k, i) => (
                  <Bar key={k} dataKey={k} fill={COLORS[i]} name={METRIC_LABELS[k] || k} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {activeTab === 'predict' && (
          <div className="aw-predict-section">
            <h3>⊕ Model Prediction</h3>
            <p className="aw-lb-subtitle">Enter feature values to get predictions from all trained models</p>
            <div className="aw-predict-grid">
              {Object.keys(featureValues).map(col => (
                <div key={col} className="aw-predict-field">
                  <label><strong>{col}</strong></label>
                  <input
                    type="text"
                    placeholder={`Enter ${col}`}
                    value={featureValues[col]}
                    onChange={(e) => setFeatureValues(prev => ({ ...prev, [col]: e.target.value }))}
                  />
                </div>
              ))}
            </div>

            <div className="aw-predict-actions">
              <button className="aw-predict-helper-btn" onClick={handleRandomRow}>✦ Random Value from Dataset</button>
              <button className="aw-predict-helper-btn" onClick={() => setShowPasteCSV(!showPasteCSV)}>📋 Paste CSV Row</button>
            </div>

            {showPasteCSV && (
              <div className="aw-paste-csv">
                <label><strong>Paste CSV Row</strong> (comma or tab separated)</label>
                <textarea
                  value={csvText}
                  onChange={(e) => setCsvText(e.target.value)}
                  placeholder="e.g., value1, value2, value3 or value1\tvalue2\tvalue3"
                  rows={3}
                />
                <div className="aw-paste-csv-actions">
                  <button className="aw-btn aw-btn--primary" onClick={handlePasteCSV}>Apply</button>
                  <button className="aw-btn aw-btn--secondary" onClick={() => setShowPasteCSV(false)}>Cancel</button>
                </div>
              </div>
            )}

            <div className="aw-predict-submit">
              <button className="aw-btn aw-btn--primary aw-predict-btn" onClick={handlePredict} disabled={predicting}>
                {predicting ? 'Predicting...' : '✦ Predict with All Models'}
              </button>
              <button className="aw-btn aw-btn--secondary" onClick={handleReset}>Reset</button>
            </div>

            {predictions.length > 0 && (
              <div className="aw-prediction-results">
                <h3>📈 Prediction Results</h3>
                <p className="aw-lb-subtitle">Predictions from all {predictions.length} trained models</p>
                {predictions.map(p => (
                  <div key={p.model_id} className="aw-prediction-card">
                    <div className="aw-prediction-model">{p.model_id}</div>
                    <div className="aw-prediction-value">
                      Predicted {targetCol}: <span className="aw-prediction-number">{p.prediction}</span>
                    </div>
                    {p.class_probabilities && (
                      <div className="aw-prediction-probs">
                        <span className="aw-prediction-probs-label">Class Probabilities:</span>
                        {Object.entries(p.class_probabilities).map(([cls, prob]) => (
                          <span key={cls} className="aw-prob-badge">{cls}: {(Number(prob) * 100).toFixed(1)}%</span>
                        ))}
                      </div>
                    )}
                    {p.error && <span className="aw-prediction-error">{p.error}</span>}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Export */}
        <div className="aw-export-section">
          <h4>Export Results</h4>
          <div className="aw-export-buttons">
            <a href={api.getExportUrl(runId, 'csv')} className="aw-btn aw-btn--secondary" download>📥 Download CSV</a>
            <a href={api.getExportUrl(runId, 'json')} className="aw-btn aw-btn--secondary" download>📥 Download JSON</a>
          </div>
        </div>
      </div>

      {/* AI Results Summary Sidebar */}
      <div className="aw-step-sidebar">
        <div className="aw-ai-summary-panel">
          <div className="aw-ai-summary-header">
            <span className="aw-ai-summary-icon">🤖</span>
            <div>
              <h4>AI Results Summary</h4>
              <p className="aw-ai-desc">Quick insights and recommendations based on your pipeline results.</p>
            </div>
          </div>
          <div className="aw-ai-summary-tags">
            <span className="aw-badge aw-badge--orange">Target: {targetCol}</span>
            <span className="aw-badge aw-badge--green">Task: {mlTask?.toUpperCase()}</span>
          </div>
          <button className="aw-btn aw-btn--ai aw-btn--full" onClick={handleGenerateSummary} disabled={aiLoading}>
            {aiLoading ? '⏳ Generating...' : '✦ Generate Summary'}
          </button>

          {aiSummary && (
            <div className="aw-ai-summary-content">
              <div className="aw-ai-section">
                <h5>✨ Executive Summary</h5>
                <p>{aiSummary.executive_summary}</p>
              </div>
              <div className="aw-ai-section">
                <h5>🔑 Key Insights</h5>
                {aiSummary.key_insights.map((ins, i) => (
                  <div key={i} className="aw-ai-insight-card">{ins}</div>
                ))}
              </div>
              <div className="aw-ai-section">
                <h5>🎯 Recommendations</h5>
                {aiSummary.recommendations.map((rec, i) => (
                  <div key={i} className="aw-ai-rec-card">{rec}</div>
                ))}
              </div>
              <div className="aw-ai-section">
                <h5>🌍 Real-World Example</h5>
                <div className="aw-ai-example-card">{aiSummary.real_world_example}</div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
