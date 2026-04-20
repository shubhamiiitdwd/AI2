import { useState, useEffect, useCallback } from 'react';
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts';
import type {
  LeaderboardResponse,
  AISummaryResponse,
  ConfusionMatrixResponse,
  HoldoutEvaluationResponse,
  ResidualsResponse,
} from '../types';
import * as api from '../api';
import { aiSourceDisplay } from '../aiSource';

interface Props {
  runId: string;
  onBack?: () => void;
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

function formatMetricValue(k: string, v: number | null): string {
  if (v == null || Number.isNaN(Number(v))) return '—';
  if (k === 'accuracy') return `${(Number(v) * 100).toFixed(1)}%`;
  const x = Number(v);
  if (!Number.isFinite(x)) return '—';
  const ax = Math.abs(x);
  if (k === 'logloss' && ax > 0 && ax < 0.01) return x.toFixed(6);
  if (ax >= 1e7 || (ax > 0 && ax < 1e-4)) return x.toExponential(3);
  if (ax >= 1000) return x.toLocaleString(undefined, { maximumFractionDigits: 2 });
  return x.toFixed(4);
}

export default function StepResults({ runId, onBack }: Props) {
  const [leaderboard, setLeaderboard] = useState<LeaderboardResponse | null>(null);
  const [bestModelData, setBestModelData] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'leaderboard' | 'performance' | 'predict'>('leaderboard');
  const [sortMetric, setSortMetric] = useState('');
  const [showMetricDropdown, setShowMetricDropdown] = useState(false);

  const [aiSummary, setAiSummary] = useState<AISummaryResponse | null>(null);
  const [aiLoading, setAiLoading] = useState(false);
  const [gainsLift, setGainsLift] = useState<GainsLiftRow[]>([]);
  const [gainsLiftNote, setGainsLiftNote] = useState('');
  const [confusionMatrix, setConfusionMatrix] = useState<ConfusionMatrixResponse | null>(null);
  const [holdoutEval, setHoldoutEval] = useState<HoldoutEvaluationResponse | null>(null);
  const [residualsFallback, setResidualsFallback] = useState<ResidualsResponse | null>(null);

  const [featureValues, setFeatureValues] = useState<Record<string, string>>({});
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [predicting, setPredicting] = useState(false);
  const [showPasteCSV, setShowPasteCSV] = useState(false);
  const [csvText, setCsvText] = useState('');
  const [predictSampleError, setPredictSampleError] = useState('');

  const loadResults = useCallback(async (isInitial: boolean) => {
    try {
      const [lb, bm] = await Promise.all([
        api.getLeaderboard(runId),
        api.getBestModel(runId).catch(() => null),
      ]);
      setLeaderboard(lb);
      setBestModelData(bm);

      if (lb && isInitial) {
        if (lb.primary_metric) {
          setSortMetric(lb.primary_metric);
        } else if (lb.models[0]) {
          const firstMetric = Object.keys(lb.models[0].metrics).find(k => lb.models[0].metrics[k] != null);
          if (firstMetric) setSortMetric(firstMetric);
        }
      }

      const featureCols = (bm as Record<string, unknown>)?.feature_columns as string[] | undefined;
      if (featureCols && isInitial) {
        const initial: Record<string, string> = {};
        featureCols.forEach(c => { initial[c] = ''; });
        setFeatureValues(initial);
      }

      if ((lb?.ml_task || '').toLowerCase() === 'classification') {
        void api.getGainsLift(runId).then((gl: { rows?: GainsLiftRow[]; note?: string }) => {
          setGainsLift(gl.rows || []);
          setGainsLiftNote((gl.note || '').trim());
        }).catch(() => {
          setGainsLift([]);
          setGainsLiftNote('');
        });
      } else {
        setGainsLift([]);
        setGainsLiftNote('');
      }

      if (lb) {
        const task = (lb.ml_task || '').toLowerCase();
        if (task === 'classification') {
          void api.getConfusionMatrix(runId).then(setConfusionMatrix).catch(() => setConfusionMatrix(null));
          void api.getHoldoutEvaluation(runId).then(setHoldoutEval).catch(() => setHoldoutEval(null));
        } else if (task === 'regression') {
          void api.getHoldoutEvaluation(runId).then(setHoldoutEval).catch(() => setHoldoutEval(null));
          void api.getResiduals(runId).then(setResidualsFallback).catch(() => setResidualsFallback(null));
        }
      }
    } catch { /* ignore */ }
    finally {
      if (isInitial) setLoading(false);
    }
  }, [runId]);

  useEffect(() => {
    setLoading(true);
    void loadResults(true);
  }, [loadResults]);

  // Refresh leaderboard / best model when the backend updates (re-run, new metrics).
  useEffect(() => {
    const t = window.setInterval(() => {
      void loadResults(false);
    }, 20000);
    return () => window.clearInterval(t);
  }, [loadResults]);

  const handleGenerateSummary = async () => {
    setAiLoading(true);
    try {
      const summary = await api.generateAISummary(runId);
      setAiSummary(summary);
    } catch { /* ignore */ }
    finally { setAiLoading(false); }
  };

  const handleRandomRow = async () => {
    setPredictSampleError('');
    try {
      const data = await api.getRandomRow(runId);
      const fv = data.feature_values || {};
      const mapped: Record<string, string> = {};
      Object.keys(fv).forEach(k => {
        const v = fv[k];
        mapped[k] = v === null || v === undefined ? '' : String(v);
      });
      setFeatureValues(mapped);
    } catch {
      setPredictSampleError(
        'Could not load a random row from the training data. If the API restarted, reload this run from history so the dataset is back in memory.',
      );
    }
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

  const selectionMetricsRaw = (bestModelData as Record<string, unknown>)?.selection_metrics as
    Record<string, number> | undefined;
  const legacyAllMetrics = (bestModelData as Record<string, unknown>)?.all_metrics as Record<string, number> || {};
  const selectionMetrics = Object.keys(selectionMetricsRaw || {}).length > 0 ? selectionMetricsRaw! : legacyAllMetrics;
  const validationMetrics = ((bestModelData as Record<string, unknown>)?.validation_metrics as Record<string, number>) || {};
  const bestModel = (bestModelData as Record<string, unknown>)?.best_model as Record<string, unknown> || {};
  const mlTaskRaw = (bestModelData as Record<string, unknown>)?.ml_task as string || leaderboard.ml_task;
  const mlTask = (mlTaskRaw || '').toLowerCase();
  const targetCol = (bestModelData as Record<string, unknown>)?.target_column as string || '';
  const evaluationWarnings = ((bestModelData as Record<string, unknown>)?.evaluation_warnings as string[] | undefined)?.filter(Boolean) || [];
  const nTargetClasses = Number((bestModelData as Record<string, unknown>)?.n_target_classes ?? 2);

  const bestAlgo = (bestModel?.algorithm as string)
    || (leaderboard.models.find(m => m.is_best)?.algorithm)
    || (leaderboard.models[0]?.algorithm)
    || '';

  const metricKeys = leaderboard.models[0]
    ? Object.keys(leaderboard.models[0].metrics).filter(k => leaderboard.models[0].metrics[k] != null)
    : [];

  const HIGHER_IS_BETTER = new Set(['auc', 'accuracy', 'r2', 'f1', 'precision', 'recall']);

  const sortedModels = [...leaderboard.models].sort((a, b) => {
    const av = Number(a.metrics[sortMetric] ?? (HIGHER_IS_BETTER.has(sortMetric) ? -Infinity : Infinity));
    const bv = Number(b.metrics[sortMetric] ?? (HIGHER_IS_BETTER.has(sortMetric) ? -Infinity : Infinity));
    return HIGHER_IS_BETTER.has(sortMetric) ? bv - av : av - bv;
  });

  const primaryMetric = leaderboard.primary_metric || metricKeys[0] || 'mean_per_class_error';
  const bestMetrics = bestModel?.metrics as Record<string, number> | undefined;

  const selectionMetricOrder = mlTask === 'classification'
    ? (nTargetClasses <= 2 ? (['auc', 'logloss'] as const) : (['mean_per_class_error', 'logloss', 'auc'] as const))
    : (['mean_residual_deviance', 'rmse', 'mae'] as const);

  const modelSelectionEntries = selectionMetricOrder.map((k) => {
    const raw = selectionMetrics[k] ?? (k === 'mean_residual_deviance' ? selectionMetrics.deviance : undefined);
    return { key: k, value: raw != null && !Number.isNaN(Number(raw)) ? Number(raw) : null };
  });

  const bestPrimaryValue = (selectionMetrics[primaryMetric] ?? (primaryMetric === 'mean_residual_deviance' ? selectionMetrics.deviance : undefined))
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
    recall: 'RECALL', mean_per_class_error: 'MEAN PER CLASS ERROR', rmsle: 'RMSLE',
    mean_residual_deviance: 'DEVIANCE', deviance: 'DEVIANCE',
  };

  const metricsEvalSet = (bestModelData as Record<string, unknown>)?.metrics_eval_set as string | undefined;
  const metricsEvalRows = (bestModelData as Record<string, unknown>)?.metrics_eval_rows as number | undefined;
  const selectionMetricsNote =
    'These values come only from the H2O AutoML leaderboard (cross-validation) for the best-ranked model. They are not computed on the holdout test rows.';
  const validationMetricsNote =
    metricsEvalSet === 'holdout' && metricsEvalRows != null && metricsEvalRows > 0
      ? `Holdout split: ${metricsEvalRows.toLocaleString()} rows. Validation metrics use the same scored rows as the confusion matrix and dataset preview.`
      : metricsEvalSet === 'train' && metricsEvalRows != null && metricsEvalRows > 0
        ? `No holdout rows were available; validation used the training slice (${metricsEvalRows.toLocaleString()} rows).`
        : '';

  const regressionFeatureKeys = (() => {
    if (mlTask !== 'regression' || !holdoutEval?.regression_rows?.length) return [] as string[];
    const row = holdoutEval.regression_rows.find(r => r.features && Object.keys(r.features).length > 0);
    return row?.features ? Object.keys(row.features) : [];
  })();

  const classificationFeatureKeys = (() => {
    if (mlTask !== 'classification' || !holdoutEval?.classification_rows?.length) return [] as string[];
    const row = holdoutEval.classification_rows.find(r => r.features && Object.keys(r.features).length > 0);
    return row?.features ? Object.keys(row.features) : [];
  })();

  const classificationHoldoutPreview = (holdoutEval?.classification_rows || []).slice(0, 10);
  const classificationProbKeys = Array.from(
    new Set(classificationHoldoutPreview.flatMap((row) => Object.keys(row.probabilities || {}))),
  ).sort();

  const regressionHoldoutPreview = (holdoutEval?.regression_rows || []).slice(0, 10);

  return (
    <div className="aw-step-content">
      <div className="aw-step-main aw-step-main--wide">
        {onBack && (
          <button className="aw-back-btn" onClick={onBack}>← Back to Training</button>
        )}
        {/* Best Model Card */}
        <div className="aw-best-model-card">
          <div className="aw-best-model-info">
            <span className="aw-best-model-label">Best Model</span>
            <span className="aw-best-model-name">{bestAlgo}</span>
          </div>
          <div className="aw-best-model-metric">
            <span className="aw-best-model-metric-label">{(primaryMetric || '').toUpperCase()}</span>
            <span className="aw-best-model-metric-value aw-best-model-metric-value--wrap">
              {bestPrimaryValue != null ? formatMetricValue(primaryMetric, Number(bestPrimaryValue)) : '-'}
            </span>
          </div>
        </div>

        {evaluationWarnings.length > 0 && (
          <div className="aw-eval-warn" role="alert">
            <strong className="aw-eval-warn-title">Training / scoring notice</strong>
            {evaluationWarnings.map((w, i) => (
              <p key={i} className="aw-eval-warn-text">{w}</p>
            ))}
          </div>
        )}

        {/* 1 — Model selection (H2O leaderboard / CV only) */}
        <div className="aw-perf-section">
          <h3 className="aw-perf-title">Model Selection Metrics (H2O AutoML - Cross Validation)</h3>
          <p className="aw-lb-subtitle aw-metrics-scope">{selectionMetricsNote}</p>
          <div className="aw-perf-cards">
            {modelSelectionEntries.map(({ key: k, value: v }) => (
              <div key={k} className="aw-perf-card">
                <span className="aw-perf-card-label">{METRIC_LABELS[k] || k.toUpperCase()}</span>
                <span className="aw-perf-card-value">
                  {formatMetricValue(k, v)}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* 2a — Classification validation (sklearn holdout only) */}
        {mlTask === 'classification' && (
          <div className="aw-perf-section">
            <h3 className="aw-perf-title">Validation Metrics (sklearn - Holdout Data)</h3>
            {validationMetricsNote && (
              <p className="aw-lb-subtitle aw-metrics-scope">{validationMetricsNote}</p>
            )}
            <p className="aw-lb-subtitle">
              Macro precision, recall, and F1 from scikit-learn only. These are not leaderboard (CV) metrics and do not include AUC, log loss, or RMSE.
            </p>
            <div className="aw-perf-cards">
              {(['precision', 'recall', 'f1'] as const).map((k) => (
                <div key={k} className="aw-perf-card">
                  <span className="aw-perf-card-label">{METRIC_LABELS[k]}</span>
                  <span className="aw-perf-card-value">
                    {formatMetricValue(k, validationMetrics[k] ?? null)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* 2b — Regression validation (holdout actual / predicted / error) */}
        {mlTask === 'regression' && regressionHoldoutPreview.length > 0 && (
          <div className="aw-holdout-section">
            <h3 className="aw-holdout-title">Validation Results (Regression)</h3>
            {validationMetricsNote && (
              <p className="aw-lb-subtitle aw-metrics-scope">{validationMetricsNote}</p>
            )}
            <p className="aw-lb-subtitle">
              One scoring pass on the holdout rows. Error = actual − predicted (same as the dataset preview and CSV download).
            </p>
            <div className="aw-table-scroll">
              <table className="aw-holdout-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Actual</th>
                    <th>Predicted</th>
                    <th>Error</th>
                  </tr>
                </thead>
                <tbody>
                  {regressionHoldoutPreview.map((row, idx) => (
                    <tr key={`${idx}-${row.actual}`}>
                      <td>{idx + 1}</td>
                      <td>{formatMetricValue('raw', row.actual)}</td>
                      <td>{formatMetricValue('raw', row.predicted)}</td>
                      <td>{formatMetricValue('raw', row.error)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Train / holdout split + test-set preview */}
        {holdoutEval && (holdoutEval.train_rows > 0 || holdoutEval.test_rows > 0) && (
          <div className="aw-holdout-banner">
            <span className="aw-holdout-banner-title">Train / holdout split (actual row counts)</span>
            <span className="aw-holdout-pill aw-holdout-pill--train">
              Training {holdoutEval.train_rows.toLocaleString()} rows ({(holdoutEval.train_fraction_actual * 100).toFixed(1)}%)
            </span>
            <span className="aw-holdout-pill aw-holdout-pill--test">
              Holdout {holdoutEval.test_rows.toLocaleString()} rows ({(holdoutEval.test_fraction_actual * 100).toFixed(1)}%)
            </span>
            <span className="aw-holdout-note">
              Requested split: {(holdoutEval.train_ratio_config * 100).toFixed(0)}% train / {(holdoutEval.test_ratio_config * 100).toFixed(0)}% test.
              Actual split: {holdoutEval.train_rows.toLocaleString()} train ({(holdoutEval.train_fraction_actual * 100).toFixed(1)}%) and {holdoutEval.test_rows.toLocaleString()} holdout ({(holdoutEval.test_fraction_actual * 100).toFixed(1)}%) — whole rows only, so small datasets may differ slightly from the requested ratio.
            </span>
          </div>
        )}

        {mlTask === 'classification' && holdoutEval && holdoutEval.classification_rows.length > 0 && (
          <div className="aw-holdout-section">
            <h3 className="aw-holdout-title">Validation dataset with predictions (preview)</h3>
            <p className="aw-lb-subtitle">
              Top {classificationHoldoutPreview.length} holdout row(s) shown. The CSV download contains every holdout row with the same columns (features, actual, predicted, class probabilities).
              Holdout split: {holdoutEval.test_rows.toLocaleString()} rows. Confidence is the predicted class probability.
            </p>
            <div className="aw-holdout-actions">
              <a
                href={api.getHoldoutPredictionsCsvUrl(runId)}
                className="aw-btn aw-btn--secondary"
                download
              >
                Download full validation CSV
              </a>
            </div>
            <div className="aw-table-scroll">
              <table className="aw-holdout-table">
                <thead>
                  <tr>
                    <th>#</th>
                    {classificationFeatureKeys.map(k => (
                      <th key={k}>{k}</th>
                    ))}
                    <th>Actual</th>
                    <th>Predicted</th>
                    {classificationProbKeys.map((pk) => (
                      <th key={pk}>P({pk})</th>
                    ))}
                    <th>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {classificationHoldoutPreview.map((row, idx) => (
                    <tr key={`${row.actual}-${idx}`}>
                      <td>{idx + 1}</td>
                      {classificationFeatureKeys.map(k => (
                        <td key={k}>{row.features?.[k] != null ? String(row.features[k]) : '—'}</td>
                      ))}
                      <td>{row.actual}</td>
                      <td>{row.predicted}</td>
                      {classificationProbKeys.map((pk) => {
                        const pv = row.probabilities?.[pk];
                        return (
                          <td key={pk}>
                            {pv == null ? '—' : (pv <= 1 && pv >= 0 ? `${(pv * 100).toFixed(2)}%` : formatMetricValue('raw', pv))}
                          </td>
                        );
                      })}
                      <td>{(row.confidence * 100).toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {mlTask === 'classification' && holdoutEval && holdoutEval.test_rows > 0 && holdoutEval.classification_rows.length === 0 && (
          <div className="aw-holdout-section aw-holdout-section--warn" role="status">
            <p className="aw-lb-subtitle">
              Holdout has {holdoutEval.test_rows.toLocaleString()} rows, but per-row predictions could not be shown (scoring may have failed). Metrics and the confusion matrix still reflect the holdout set when available.
            </p>
          </div>
        )}

        {mlTask === 'regression' &&
          holdoutEval &&
          holdoutEval.regression_rows.length > 0 && (
            <div className="aw-holdout-section">
              <h3 className="aw-holdout-title">Validation dataset with predictions (preview)</h3>
              <p className="aw-lb-subtitle">
                Top {regressionHoldoutPreview.length} holdout row(s) with features, actual, predicted, and error (actual − predicted). The CSV download contains the full holdout ({holdoutEval.test_rows.toLocaleString()} rows).
              </p>
              <div className="aw-holdout-actions">
                <a
                  href={api.getHoldoutRegressionPredictionsCsvUrl(runId)}
                  className="aw-btn aw-btn--secondary"
                  download
                >
                  Download full validation CSV
                </a>
              </div>
              <div className="aw-table-scroll">
                <table className="aw-holdout-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      {regressionFeatureKeys.map(k => (
                        <th key={k}>{k}</th>
                      ))}
                      <th>Actual</th>
                      <th>Predicted</th>
                      <th>Error</th>
                    </tr>
                  </thead>
                  <tbody>
                    {regressionHoldoutPreview.map((row, idx) => (
                      <tr key={`${idx}-${row.actual}`}>
                        <td>{idx + 1}</td>
                        {regressionFeatureKeys.map(k => (
                          <td key={k}>{row.features?.[k] != null ? String(row.features[k]) : '—'}</td>
                        ))}
                        <td>{formatMetricValue('raw', row.actual)}</td>
                        <td>{formatMetricValue('raw', row.predicted)}</td>
                        <td>{formatMetricValue('raw', row.error)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

        {mlTask === 'regression' &&
          (!holdoutEval || holdoutEval.regression_rows.length === 0) &&
          residualsFallback &&
          residualsFallback.actual.length > 0 && (
            <div className="aw-holdout-section">
              <h3 className="aw-holdout-title">Prediction vs actual (sample)</h3>
              <p className="aw-lb-subtitle">From stored results — re-train to get the full holdout table with split info.</p>
              <div className="aw-table-scroll">
                <table className="aw-holdout-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Actual</th>
                      <th>Predicted</th>
                      <th>Error</th>
                    </tr>
                  </thead>
                  <tbody>
                    {residualsFallback.actual.slice(0, 50).map((a, idx) => {
                      const p = residualsFallback.predicted[idx];
                      const err =
                        residualsFallback.errors?.[idx] ?? (p !== undefined && a !== undefined ? Number(a) - Number(p) : 0);
                      return (
                        <tr key={idx}>
                          <td>{idx + 1}</td>
                          <td>{formatMetricValue('raw', a)}</td>
                          <td>{p !== undefined ? formatMetricValue('raw', p) : '—'}</td>
                          <td>{formatMetricValue('raw', err)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

        {mlTask === 'classification' && confusionMatrix && confusionMatrix.labels.length > 0 && (
          <div className="aw-cm-section">
            <h3 className="aw-holdout-title">Confusion matrix (sklearn - Holdout Data)</h3>
            <p className="aw-lb-subtitle">Rows: actual class · Columns: predicted class. Matches macro precision, recall, and F1 in the validation section (same holdout predictions).</p>
            <div className="aw-table-scroll">
              <table className="aw-cm-table">
                <thead>
                  <tr>
                    <th className="aw-cm-corner" />
                    {confusionMatrix.labels.map((lab) => (
                      <th key={lab}>{lab}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {confusionMatrix.matrix.slice(0, confusionMatrix.labels.length).map((row, i) => (
                    <tr key={i}>
                      <th>{confusionMatrix.labels[i] ?? i}</th>
                      {row.slice(0, confusionMatrix.labels.length).map((cell, j) => (
                        <td key={j} className="aw-cm-cell">
                          {typeof cell === 'number' ? cell.toLocaleString() : String(cell)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Gains/Lift — after confusion matrix; binary classification */}
        {mlTask === 'classification' && (gainsLift.length > 0 || Boolean(gainsLiftNote)) && (
          <div className="aw-gains-section">
            <h3 className="aw-perf-title">Gains / Lift</h3>
            {gainsLift.length > 0 && (
              <>
                <div className="aw-chart-container aw-gains-chart">
                  <ResponsiveContainer width="100%" height={280}>
                    <LineChart data={gainsLift}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="group" name="Group" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="lift" name="Lift" stroke="#e67e22" strokeWidth={2} dot />
                      <Line type="monotone" dataKey="cumulative_data_pct" name="Cumulative data %" stroke="#3498db" strokeWidth={2} dot />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <p className="aw-lb-subtitle">
                  {gainsLiftNote.includes('full imported')
                    ? 'Decile lift vs cumulative coverage (see note below for which frame was scored).'
                    : 'Decile lift vs cumulative data coverage (training frame after split).'}
                </p>
                <div className="aw-table-scroll">
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
              </>
            )}
            {gainsLiftNote && (
              <p className="aw-gains-note">{gainsLiftNote}</p>
            )}
          </div>
        )}

        {/* Model Leaderboard Header */}
        <div className="aw-lb-header">
          <div>
            <h3>Model Leaderboard</h3>
            <p className="aw-lb-subtitle">
              Top performing models ranked by {METRIC_LABELS[sortMetric] || (sortMetric || '').toUpperCase()}
              {' '}({HIGHER_IS_BETTER.has(sortMetric) ? 'higher is better' : 'lower is better'})
            </p>
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
              <button type="button" className="aw-predict-helper-btn" onClick={() => void handleRandomRow()}>
                ✦ Random Value from Dataset
              </button>
              <button type="button" className="aw-predict-helper-btn" onClick={() => setShowPasteCSV(!showPasteCSV)}>
                📋 Paste CSV Row
              </button>
            </div>

            {predictSampleError && (
              <div className="aw-eval-warn" role="alert" style={{ marginTop: 10 }}>
                <p className="aw-eval-warn-text">{predictSampleError}</p>
              </div>
            )}

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
            {aiSummary?.source ? (
              <span className={`aw-badge ${aiSourceDisplay(aiSummary.source).badgeClass}`}>
                AI Source: {aiSourceDisplay(aiSummary.source).label}
              </span>
            ) : null}
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
