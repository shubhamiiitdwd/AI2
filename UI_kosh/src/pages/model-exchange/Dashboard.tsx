import { useState, useEffect, useCallback } from 'react';
import './Dashboard.css';
import { listDatasets, getTrainingHistory } from './tools/api';

interface Props {
  onStartProject: () => void;
}

interface RecentRun {
  id: string;
  title: string;
  tags: string[];
  time: string;
}

export default function Dashboard({ onStartProject }: Props) {
  const [stats, setStats] = useState({ models: 0, sessions: 0, datasets: 0, success: '0.0' });
  const [recentRuns] = useState<RecentRun[]>([
    { id: '1', title: 'Extraction', tags: ['ocr', 'automl'], time: 'Just now' },
  ]);

  const loadStats = useCallback(async () => {
    try {
      const [datasets, history] = await Promise.all([
        listDatasets(),
        getTrainingHistory().catch(() => ({ runs: [] })),
      ]);
      const trainingRuns = history.runs.filter(r => r.run_type === 'training');
      const models = trainingRuns.reduce((acc, r) => acc + (r.model_count || 0), 0);
      const sessions = trainingRuns.length;
      // Same rule as Results: 100% when the pipeline completed with models; 0% if nothing succeeded yet.
      const successPct = sessions > 0 && models > 0 ? 100 : 0;
      setStats({
        models,
        sessions,
        datasets: datasets.length,
        success: successPct.toFixed(1),
      });
    } catch {
      /* keep previous stats */
    }
  }, []);

  useEffect(() => {
    void loadStats();
    const t = window.setInterval(() => void loadStats(), 25000);
    return () => window.clearInterval(t);
  }, [loadStats]);

  return (
    <div className="mx-page">
      {/* ── Navbar ─────────────────────────────────────────────── */}
      <header className="mx-navbar">
        <div className="mx-navbar-left">
          <div className="mx-ntt-logo">
            <span className="mx-ntt-icon">◉</span>
            <span className="mx-ntt-text">NTT DATA</span>
          </div>
        </div>
        <nav className="mx-navbar-right">
          <button className="mx-nav-link" onClick={() => window.location.reload()}>↻ Reset</button>
          <button className="mx-nav-link">← Back to Home</button>
          <button className="mx-nav-link">↗ Logout</button>
        </nav>
      </header>

      {/* ── Hero Section ──────────────────────────────────────── */}
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
        {/* decorative circles */}
        <div className="mx-hero-circle mx-hero-circle--1" />
        <div className="mx-hero-circle mx-hero-circle--2" />
        <div className="mx-hero-circle mx-hero-circle--3" />
      </section>

      {/* ── Stats Cards ───────────────────────────────────────── */}
      <section className="mx-stats">
        <div className="mx-stat-card">
          <div className="mx-stat-label">Models Trained</div>
          <div className="mx-stat-row">
            <span className="mx-stat-value">{stats.models}</span>
            <svg className="mx-stat-spark" viewBox="0 0 60 24"><polyline points="0,20 10,16 20,18 30,10 40,14 50,8 60,12" fill="none" stroke="#e67e22" strokeWidth="2"/></svg>
          </div>
          <div className="mx-stat-footnote">All time · from AutoML runs</div>
        </div>
        <div className="mx-stat-card">
          <div className="mx-stat-label">Training Sessions</div>
          <div className="mx-stat-row">
            <span className="mx-stat-value">{stats.sessions}</span>
            <svg className="mx-stat-spark" viewBox="0 0 60 24"><polyline points="0,18 10,12 20,16 30,8 40,14 50,10 60,6" fill="none" stroke="#e67e22" strokeWidth="2"/></svg>
          </div>
          <div className="mx-stat-footnote">All time · completed sessions</div>
        </div>
        <div className="mx-stat-card">
          <div className="mx-stat-label">Datasets Processed</div>
          <div className="mx-stat-row">
            <span className="mx-stat-value">{stats.datasets}</span>
            <svg className="mx-stat-spark" viewBox="0 0 60 24"><polyline points="0,20 10,14 20,18 30,6 40,16 50,10 60,8" fill="none" stroke="#e67e22" strokeWidth="2"/></svg>
          </div>
          <div className="mx-stat-footnote">All time · uploaded datasets</div>
        </div>
        <div className="mx-stat-card">
          <div className="mx-stat-label">Success Rate</div>
          <div className="mx-stat-row">
            <span className="mx-stat-value">{stats.success}%</span>
            <svg className="mx-stat-spark" viewBox="0 0 60 24"><polyline points="0,16 10,18 20,14 30,12 40,18 50,8 60,16" fill="none" stroke="#e67e22" strokeWidth="2"/></svg>
          </div>
          <div className="mx-stat-footnote">100% when at least one run finished with models</div>
        </div>
      </section>

      {/* ── Bottom Panels ─────────────────────────────────────── */}
      <section className="mx-panels">
        {/* Recent Activity */}
        <div className="mx-panel mx-panel--activity">
          <div className="mx-panel-header">
            <h3><span className="mx-panel-icon">⚡</span> Recent Activity</h3>
            <p>Your latest AI projects and jobs</p>
          </div>
          <div className="mx-activity-list">
            {recentRuns.map(run => (
              <div key={run.id} className="mx-activity-item">
                <div className="mx-activity-info">
                  <span className="mx-activity-title">{run.title}</span>
                  <span className="mx-activity-badge mx-activity-badge--green">Data Exchange</span>
                  <span className="mx-activity-check">✓</span>
                </div>
                <button className="mx-activity-view" onClick={onStartProject}>View →</button>
                <div className="mx-activity-meta">
                  <span>⏱ {run.time}</span>
                  <div className="mx-activity-tags">
                    {run.tags.map(t => <span key={t} className="mx-activity-tag">{t}</span>)}
                  </div>
                </div>
              </div>
            ))}
            {recentRuns.length === 0 && (
              <div className="mx-activity-empty">No recent activity. Start a new project!</div>
            )}
          </div>
        </div>

        {/* AutoML Capabilities */}
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

      {/* ── Floating Chat Button ──────────────────────────────── */}
      <button className="mx-fab" title="Chat with AI Assistant">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M20 2H4C2.9 2 2 2.9 2 4V22L6 18H20C21.1 18 22 17.1 22 16V4C22 2.9 21.1 2 20 2Z" fill="white"/></svg>
      </button>
    </div>
  );
}
