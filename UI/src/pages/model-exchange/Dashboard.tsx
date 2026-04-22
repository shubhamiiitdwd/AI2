import { useState, useEffect, useCallback, useSyncExternalStore } from 'react';
import './Dashboard.css';
import { getTrainingHistory } from './tools/automl-wizard/api';
import type { TrainingRunSummary } from './tools/automl-wizard/types';
import {
  readModelExchangeSession,
  clearModelExchangeSession,
  getSessionActivityLabel,
  formatSessionTime,
  formatSessionRelative,
  type PersistedModelExchangeSession,
} from './modelExchangeSession';

/** Inclusive lower bound for dashboard stats & recent activity (YYYY-MM-DD). */
const STATS_FROM_YMD = import.meta.env.VITE_STATS_HISTORY_FROM_DATE || '2026-04-15';

function subscribeModelExchangeSession(onStoreChange: () => void) {
  const h = () => onStoreChange();
  window.addEventListener('aikosh-mx-session', h);
  const id = window.setInterval(h, 10000);
  return () => {
    window.removeEventListener('aikosh-mx-session', h);
    window.clearInterval(id);
  };
}

function getSessionSnapshot(): PersistedModelExchangeSession | null {
  return readModelExchangeSession();
}


interface Props {
  onStartProject: () => void;
  onOpenRunResults?: (run: TrainingRunSummary) => void;
  onResumeSession?: () => void;
}

function ymdFromCreatedAt(createdAt: string): string {
  return (createdAt || '').trim().slice(0, 10);
}

function formatYmdReadable(ymd: string): string {
  const [y, m, d] = ymd.split('-').map(Number);
  if (!y || !m || !d) return ymd;
  return new Intl.DateTimeFormat('en', { month: 'short', day: 'numeric', year: 'numeric' }).format(new Date(y, m - 1, d));
}

function isPlaceholderRun(r: TrainingRunSummary): boolean {
  return r.run_type === 'training' && !(r.dataset_name || '').trim() && (r.model_count || 0) === 0;
}

function runSucceeded(r: TrainingRunSummary): boolean {
  if (r.run_type === 'clustering' || r.ml_task === 'clustering') {
    return (r.model_count || 0) > 0 || (r.best_metric_value || 0) > 0;
  }
  return (r.model_count || 0) > 0;
}

function formatRelativeTime(createdAt: string): string {
  const t = Date.parse(createdAt.replace(' ', 'T'));
  if (Number.isNaN(t)) return '';
  const diffMs = Date.now() - t;
  const sec = Math.floor(diffMs / 1000);
  if (sec < 60) return 'Just now';
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min} min ago`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr} hr ago`;
  const day = Math.floor(hr / 24);
  if (day < 7) return `${day} day${day === 1 ? '' : 's'} ago`;
  return formatYmdReadable(ymdFromCreatedAt(createdAt));
}

function activityTags(run: TrainingRunSummary): string[] {
  const kind =
    run.run_type === 'clustering' || run.ml_task === 'clustering' ? 'clustering' : run.ml_task;
  const raw = [kind, run.run_type, run.best_algorithm || run.primary_metric].filter(
    (x): x is string => typeof x === 'string' && x.trim().length > 0,
  );
  return [...new Set(raw)].slice(0, 4);
}

function isClusteringRun(run: TrainingRunSummary): boolean {
  return run.run_type === 'clustering' || run.ml_task === 'clustering';
}

export default function Dashboard({ onStartProject, onOpenRunResults, onResumeSession }: Props) {
  const session = useSyncExternalStore(
    subscribeModelExchangeSession,
    getSessionSnapshot,
    getSessionSnapshot,
  );
  const [stats, setStats] = useState({
    models: 0,
    sessions: 0,
    datasets: 0,
    success: '0.0',
    taskKinds: 0,
  });
  const [statsPeriodLabel, setStatsPeriodLabel] = useState('');
  const [recentRuns, setRecentRuns] = useState<TrainingRunSummary[]>([]);

  const loadStats = useCallback(async () => {
    try {
      const history = await getTrainingHistory(400).catch(() => ({ runs: [] as TrainingRunSummary[] }));
      const fromYmd = STATS_FROM_YMD;
      setStatsPeriodLabel(formatYmdReadable(fromYmd));

      const windowRuns = history.runs.filter(
        (r) => r.created_at && ymdFromCreatedAt(r.created_at) >= fromYmd && !isPlaceholderRun(r),
      );

      // Models = AutoML leaderboard size + clustering candidates evaluated (both stored as model_count).
      const models = windowRuns.reduce((acc, r) => acc + (r.model_count || 0), 0);
      const sessions = windowRuns.length;
      const datasetIds = new Set(windowRuns.map((r) => r.dataset_id).filter(Boolean));
      const datasets = datasetIds.size;

      const ok = windowRuns.filter(runSucceeded).length;
      const successPct = sessions > 0 ? (100 * ok) / sessions : 0;

      const taskKindsSet = new Set<string>();
      windowRuns.forEach((r) => {
        const rt = (r.run_type || '').toLowerCase();
        const mt = (r.ml_task || '').toLowerCase();
        if (rt === 'clustering' || mt === 'clustering') taskKindsSet.add('clustering');
        else if (mt === 'classification' || mt === 'regression') taskKindsSet.add(mt);
        else if (rt === 'training' && mt) taskKindsSet.add(mt);
      });
      const taskKinds = taskKindsSet.size;

      setStats({
        models,
        sessions,
        datasets,
        success: successPct.toFixed(1),
        taskKinds,
      });

      const sorted = [...windowRuns].sort((a, b) => {
        const ta = Date.parse(a.created_at.replace(' ', 'T'));
        const tb = Date.parse(b.created_at.replace(' ', 'T'));
        return (Number.isNaN(tb) ? 0 : tb) - (Number.isNaN(ta) ? 0 : ta);
      });
      setRecentRuns(sorted.slice(0, 12));
    } catch {
      /* keep previous stats */
    }
  }, []);

  useEffect(() => {
    void loadStats();
    const t = window.setInterval(() => void loadStats(), 25000);
    return () => window.clearInterval(t);
  }, [loadStats]);

  const period = statsPeriodLabel || formatYmdReadable(STATS_FROM_YMD);

  return (
    <div className="mx-page">
      <section className="mx-hero">
        <div className="mx-hero-inner">
          <span className="mx-hero-badge">⚡ No-Code AI Platform</span>
          <h1 className="mx-hero-title">Model Exchange</h1>
          <p className="mx-hero-desc">
            Build, train, and deploy machine learning models with AutoML. Select your dataset,
            configure training parameters, and let our automated pipeline find the best model
            for your use case.
          </p>
          <button className="mx-hero-btn" onClick={onStartProject}>
            <span className="mx-hero-btn-icon">⚡</span>
            Start New Project
            <span className="mx-hero-btn-arrow">→</span>
          </button>
        </div>
        <div className="mx-hero-circle mx-hero-circle--1" />
        <div className="mx-hero-circle mx-hero-circle--2" />
        <div className="mx-hero-circle mx-hero-circle--3" />
      </section>

      <section className="mx-stats">
        <div className="mx-stat-card">
          <div className="mx-stat-label">Models evaluated</div>
          <div className="mx-stat-row">
            <span className="mx-stat-value">{stats.models}</span>
            <svg className="mx-stat-spark" viewBox="0 0 60 24" aria-hidden><polyline points="0,20 10,16 20,18 30,10 40,14 50,8 60,12" fill="none" strokeWidth="2"/></svg>
          </div>
          <div className="mx-stat-footnote">
            Since {period}
            {stats.taskKinds > 0
              ? ` · ${stats.taskKinds} task type${stats.taskKinds === 1 ? '' : 's'} (classification / regression / clustering)`
              : ''}
          </div>
        </div>
        <div className="mx-stat-card">
          <div className="mx-stat-label">Training Sessions</div>
          <div className="mx-stat-row">
            <span className="mx-stat-value">{stats.sessions}</span>
            <svg className="mx-stat-spark" viewBox="0 0 60 24" aria-hidden><polyline points="0,18 10,12 20,16 30,8 40,14 50,10 60,6" fill="none" strokeWidth="2"/></svg>
          </div>
          <div className="mx-stat-footnote">Since {period} · AutoML and clustering jobs</div>
        </div>
        <div className="mx-stat-card">
          <div className="mx-stat-label">Datasets Processed</div>
          <div className="mx-stat-row">
            <span className="mx-stat-value">{stats.datasets}</span>
            <svg className="mx-stat-spark" viewBox="0 0 60 24" aria-hidden><polyline points="0,20 10,14 20,18 30,6 40,16 50,10 60,8" fill="none" strokeWidth="2"/></svg>
          </div>
          <div className="mx-stat-footnote">Since {period} · uploads or runs in this window</div>
        </div>
        <div className="mx-stat-card">
          <div className="mx-stat-label">Success Rate</div>
          <div className="mx-stat-row">
            <span className="mx-stat-value">{stats.success}%</span>
            <svg className="mx-stat-spark" viewBox="0 0 60 24" aria-hidden><polyline points="0,16 10,18 20,14 30,12 40,18 50,8 60,16" fill="none" strokeWidth="2"/></svg>
          </div>
          <div className="mx-stat-footnote">
            100% × (successful jobs ÷ jobs in period). Training: ≥1 model. Clustering: composite score or candidates tested.
          </div>
        </div>
      </section>

      <section className="mx-panels mx-panels--triple" aria-label="Model Exchange overview">
        <div className="mx-panel mx-panel--activity">
          <div className="mx-panel-header">
            <h3><span className="mx-panel-icon">⚡</span> Recent Activity</h3>
            <p>Your latest AI projects and jobs</p>
          </div>
          <div className="mx-activity-list">
            {recentRuns.map((run) => {
              const runKey = (run.run_id && String(run.run_id)) || 'unknown';
              const title = run.dataset_name?.trim() || `Run ${runKey.slice(0, 8)}`;
              const tags = activityTags(run);
              const cl = isClusteringRun(run);
              return (
                <div key={runKey} className="mx-activity-item">
                  <div className="mx-activity-info">
                    <span className="mx-activity-title">{title}</span>
                    <span
                      className={
                        cl
                          ? 'mx-activity-badge mx-activity-badge--purple'
                          : 'mx-activity-badge mx-activity-badge--green'
                      }
                    >
                      {cl ? 'Clustering' : 'AutoML'}
                    </span>
                    <span className="mx-activity-check">✓</span>
                  </div>
                  <button
                    type="button"
                    className="mx-activity-view"
                    onClick={() => (onOpenRunResults ? onOpenRunResults(run) : onStartProject())}
                  >
                    View →
                  </button>
                  <div className="mx-activity-meta">
                    <span>⏱ {formatRelativeTime((run.created_at || '').replace(' ', 'T')) || '—'}</span>
                    <div className="mx-activity-tags">
                      {tags.map((t) => (
                        <span key={t} className="mx-activity-tag">{t}</span>
                      ))}
                    </div>
                  </div>
                </div>
              );
            })}
            {recentRuns.length === 0 && (
              <div className="mx-activity-empty">No recent activity in this period. Start a new project!</div>
            )}
          </div>
        </div>

        <div className="mx-panel mx-panel--session">
          <div className="mx-panel-header">
            <h3><span className="mx-panel-icon">◆</span> Session management</h3>
            <p>Resume the AutoML or clustering workflow where you left off</p>
          </div>
          {session ? (
            <div className="mx-session-card">
              <div className="mx-session-activity">
                {getSessionActivityLabel({
                  step: session.step,
                  mlTask: session.mlTask,
                  activity: session.activity,
                  dataset: session.dataset,
                })}
              </div>
              <div className="mx-session-time">
                <span className="mx-session-time-label">Last saved</span>
                <time dateTime={session.at} title={formatSessionTime(session.at)}>
                  {formatSessionRelative(session.at)} · {formatSessionTime(session.at)}
                </time>
              </div>
              <div className="mx-session-actions">
                <button
                  type="button"
                  className="mx-session-btn mx-session-btn--primary"
                  onClick={() => onResumeSession?.()}
                  disabled={!onResumeSession}
                >
                  Resume
                </button>
                <button
                  type="button"
                  className="mx-session-btn"
                  onClick={() => {
                    clearModelExchangeSession();
                    window.dispatchEvent(new Event('aikosh-mx-session'));
                  }}
                >
                  Clear
                </button>
              </div>
            </div>
          ) : (
            <div className="mx-session-empty">
              No saved in-progress session. Open <strong>Start New Project</strong> to work in the wizard; your
              place is saved automatically so you can resume here.
            </div>
          )}
        </div>

        <div className="mx-panel mx-panel--capabilities">
          <div className="mx-panel-header">
            <h3>AutoML Capabilities</h3>
            <p>Powerful machine learning at your fingertips</p>
          </div>
          <div className="mx-capabilities">
            <div className="mx-cap-card mx-cap-card--orange">
              <span className="mx-cap-icon">⚡</span>
              <div>
                <strong>Multiple Algorithms</strong>
                <p>Train with 10+ algorithms including Random Forest, XGBoost, Neural Networks, and more</p>
              </div>
            </div>
            <div className="mx-cap-card mx-cap-card--green">
              <span className="mx-cap-icon">📊</span>
              <div>
                <strong>Real-time Monitoring</strong>
                <p>Track training progress, metrics, and performance in real-time with live updates</p>
              </div>
            </div>
            <div className="mx-cap-card mx-cap-card--blue">
              <span className="mx-cap-icon">🔄</span>
              <div>
                <strong>Model Comparison</strong>
                <p>Compare metrics across models and select the best performer for deployment</p>
              </div>
            </div>
            <div className="mx-cap-card mx-cap-card--purple">
              <span className="mx-cap-icon">🎯</span>
              <div>
                <strong>Auto-Optimization</strong>
                <p>Automatic hyperparameter tuning for optimal model performance</p>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
