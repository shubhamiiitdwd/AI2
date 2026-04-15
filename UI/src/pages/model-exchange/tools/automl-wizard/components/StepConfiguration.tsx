import { useState } from 'react';
import type { MLTask, ModelType } from '../types';
import { MODEL_INFO } from '../types';

interface Props {
  mlTask: MLTask;
  selectedModels: ModelType[];
  autoMode: boolean;
  trainTestSplit: number;
  nfolds: number;
  maxModels: number;
  maxRuntimeSecs: number;
  onTaskChange: (task: MLTask) => void;
  onModelsChange: (models: ModelType[]) => void;
  onAutoModeChange: (auto: boolean) => void;
  onSplitChange: (split: number) => void;
  onNfoldsChange: (n: number) => void;
  onMaxModelsChange: (n: number) => void;
  onMaxRuntimeChange: (n: number) => void;
  onStartTraining: () => void;
  onBack?: () => void;
}

const TASKS: { value: MLTask; label: string; desc: string; icon: string }[] = [
  { value: 'classification', label: 'Classification', desc: 'Predict categorical outcomes', icon: '📈' },
  { value: 'regression', label: 'Regression', desc: 'Predict continuous values', icon: '📉' },
];

const ALL_MODELS: ModelType[] = ['DRF', 'GLM', 'XGBoost', 'GBM', 'DeepLearning', 'StackedEnsemble'];

const SPEED_COLORS: Record<string, string> = {
  Fast: '#2ecc71',
  Medium: '#e67e22',
  Slow: '#e74c3c',
};

export default function StepConfiguration({
  mlTask, selectedModels, autoMode,
  trainTestSplit, nfolds, maxModels, maxRuntimeSecs,
  onTaskChange, onModelsChange, onAutoModeChange,
  onSplitChange, onNfoldsChange, onMaxModelsChange, onMaxRuntimeChange,
  onStartTraining, onBack,
}: Props) {
  const [hyperTab, setHyperTab] = useState<'basic' | 'advanced'>('basic');

  const toggleModel = (model: ModelType) => {
    onModelsChange(
      selectedModels.includes(model)
        ? selectedModels.filter((m) => m !== model)
        : [...selectedModels, model],
    );
  };

  return (
    <div className="aw-step-content">
      <div className="aw-step-main">
        {onBack && (
          <button className="aw-back-btn" onClick={onBack}>← Back to Configure Data</button>
        )}
        <div className="aw-auto-mode">
          <span>Auto Mode</span>
          <label className="aw-toggle">
            <input type="checkbox" checked={autoMode} onChange={(e) => onAutoModeChange(e.target.checked)} />
            <span className="aw-toggle-slider" />
          </label>
        </div>

        {autoMode ? (
          <div className="aw-auto-mode-info">
            <h3>Auto Mode Enabled</h3>
            <p>In Auto Mode, the system will automatically select the best ML task and models for your dataset.</p>
          </div>
        ) : (
          <>
            <h3 className="aw-section-title">Select ML Task</h3>
            <div className="aw-task-cards">
              {TASKS.map((t) => (
                <div
                  key={t.value}
                  className={`aw-task-card ${mlTask === t.value ? 'aw-task-card--active' : ''}`}
                  onClick={() => onTaskChange(t.value)}
                >
                  <span className="aw-task-icon">{t.icon}</span>
                  <span className="aw-task-label">{t.label}</span>
                  <span className="aw-task-desc">{t.desc}</span>
                </div>
              ))}
            </div>

            <h3 className="aw-section-title">Select Models to Train</h3>
            <div className="aw-model-cards">
              {ALL_MODELS.map((m) => {
                const info = MODEL_INFO[m];
                return (
                  <div
                    key={m}
                    className={`aw-model-card ${selectedModels.includes(m) ? 'aw-model-card--selected' : ''}`}
                    onClick={() => toggleModel(m)}
                  >
                    <div className="aw-model-card-header">
                      <input type="checkbox" checked={selectedModels.includes(m)} readOnly />
                      <strong>{info.name}</strong>
                      <span className="aw-speed-badge" style={{ backgroundColor: SPEED_COLORS[info.speed] }}>
                        {info.speed}
                      </span>
                    </div>
                    <p className="aw-model-desc">{info.description}</p>
                  </div>
                );
              })}
            </div>

            <h3 className="aw-section-title">Select Hyperparameters</h3>
            <div className="aw-hyper-tabs">
              <button className={`aw-tab ${hyperTab === 'basic' ? 'aw-tab--active' : ''}`} onClick={() => setHyperTab('basic')}>
                Basic Settings
              </button>
              <button className={`aw-tab ${hyperTab === 'advanced' ? 'aw-tab--active' : ''}`} onClick={() => setHyperTab('advanced')}>
                Advanced Settings
              </button>
            </div>

            {hyperTab === 'basic' ? (
              <div className="aw-hyper-grid">
                <div className="aw-hyper-field">
                  <label>Train/Test Split</label>
                  <select value={trainTestSplit} onChange={(e) => onSplitChange(Number(e.target.value))}>
                    <option value={0.8}>80/20</option>
                    <option value={0.7}>70/30</option>
                    <option value={0.9}>90/10</option>
                  </select>
                </div>
                <div className="aw-hyper-field">
                  <label>Cross-validation Folds</label>
                  <select value={nfolds} onChange={(e) => onNfoldsChange(Number(e.target.value))}>
                    <option value={3}>3</option>
                    <option value={5}>5</option>
                    <option value={10}>10</option>
                  </select>
                </div>
              </div>
            ) : (
              <div className="aw-hyper-grid">
                <div className="aw-hyper-field">
                  <label>Maximum Models</label>
                  <select value={maxModels} onChange={(e) => onMaxModelsChange(Number(e.target.value))}>
                    <option value={5}>5</option>
                    <option value={10}>10</option>
                    <option value={20}>20</option>
                    <option value={50}>50</option>
                  </select>
                </div>
                <div className="aw-hyper-field">
                  <label>Maximum Runtime(s)</label>
                  <select value={maxRuntimeSecs} onChange={(e) => onMaxRuntimeChange(Number(e.target.value))}>
                    <option value={60}>60</option>
                    <option value={120}>120</option>
                    <option value={300}>300</option>
                    <option value={600}>600</option>
                  </select>
                </div>
              </div>
            )}
          </>
        )}

        <button className="aw-btn aw-btn--primary aw-btn--full" onClick={onStartTraining}>
          Start Training →
        </button>
      </div>

      <div className="aw-step-sidebar">
        <h4>AutoML Workflow:</h4>
        <ul className="aw-workflow-list">
          <li>Search catalog for datasets</li>
          <li>Preview and configure data columns</li>
          <li className="aw-workflow-active">Select ML task and models</li>
          <li>Configure training settings</li>
          <li>Train and compare models</li>
          <li>View results and export</li>
        </ul>
      </div>
    </div>
  );
}
