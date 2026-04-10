import { useState, useEffect, useRef } from 'react';
import { useTrainingWebSocket } from '../hooks/useWebSocket';
import { getWsUrl } from '../api';
import type { TrainingStatusResponse } from '../types';
import * as api from '../api';

interface Props {
  runId: string;
  onComplete: () => void;
  reviewMode?: boolean;
  onBack?: () => void;
}

const STAGES = ['queued', 'data_check', 'features', 'training', 'evaluation', 'complete'];

function _countModelsFromLogs(messages: { message: string }[]): number {
  return messages.filter(m => m.message.includes('AutoML: starting ')).length;
}

function _extractActiveModel(messages: { message: string }[]): string {
  const ALGOS = ['StackedEnsemble', 'DeepLearning', 'XGBoost', 'GBM', 'GLM', 'DRF', 'XRT'];
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i].message;
    if (msg.includes('AutoML: starting') || msg.includes('New leader')) {
      for (const algo of ALGOS) {
        if (msg.includes(algo)) return algo;
      }
    }
  }
  return '-';
}
const STAGE_LABELS: Record<string, string> = {
  queued: 'Queued', data_check: 'Data Check', features: 'Features',
  training: 'Training', evaluation: 'Evaluation', complete: 'Complete',
};

export default function StepTraining({ runId, onComplete, reviewMode = false, onBack }: Props) {
  const wsUrl = getWsUrl(runId);
  const { messages, lastMessage, connected } = useTrainingWebSocket(wsUrl);
  const [status, setStatus] = useState<TrainingStatusResponse | null>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const [filter, setFilter] = useState('');
  const logsRef = useRef<HTMLDivElement>(null);
  const hasAutoCompleted = useRef(false);

  // HTTP polling as fallback — only fires when WebSocket is disconnected.
  useEffect(() => {
    const poll = setInterval(async () => {
      if (connected) return; // Skip when WS is live — avoids redundant calls
      try {
        const s = await api.getTrainingStatus(runId);
        setStatus(s);
        if (s.status === 'complete' || s.status === 'failed' || s.status === 'stopped') {
          clearInterval(poll);
          if (s.status === 'complete' && !reviewMode && !hasAutoCompleted.current) {
            hasAutoCompleted.current = true;
            setTimeout(onComplete, 1500);
          }
        }
      } catch { /* ignore */ }
    }, 2000);
    return () => clearInterval(poll);
  }, [runId, onComplete, reviewMode, connected]);

  // Sync status from every WebSocket message so progress bar updates in real-time
  useEffect(() => {
    if (!lastMessage) return;
    setStatus({
      run_id: runId,
      status: lastMessage.status as TrainingStatusResponse['status'],
      progress_percent: lastMessage.progress,
      current_stage: lastMessage.stage,
      message: lastMessage.message,
    });
    if (lastMessage.status === 'complete' || lastMessage.status === 'failed' || lastMessage.status === 'stopped') {
      if (lastMessage.status === 'complete' && !reviewMode && !hasAutoCompleted.current) {
        hasAutoCompleted.current = true;
        setTimeout(onComplete, 1500);
      }
    }
  }, [lastMessage, runId, onComplete, reviewMode]);

  useEffect(() => {
    if (autoScroll && logsRef.current) {
      logsRef.current.scrollTop = logsRef.current.scrollHeight;
    }
  }, [messages, autoScroll]);

  const progress = status?.progress_percent ?? lastMessage?.progress ?? 0;
  const currentStage = status?.current_stage ?? lastMessage?.stage ?? 'queued';
  const stageIndex = STAGES.indexOf(currentStage);

  const filteredMessages = filter
    ? messages.filter((m) => m.message.toLowerCase().includes(filter.toLowerCase()))
    : messages;

  const copyLogs = () => {
    const text = messages.map((m) => `[${m.timestamp}] ${m.message}`).join('\n');
    navigator.clipboard.writeText(text);
  };

  const isStarting = progress < 10 && currentStage === 'queued' && !reviewMode;
  const isComplete = status?.status === 'complete' || currentStage === 'complete';

  return (
    <div className="aw-step-content">
      <div className="aw-step-main">
        {onBack && (
          <button className="aw-back-btn" onClick={onBack}>← Back to Configuration</button>
        )}
        {isStarting && (
          <div className="aw-starting">
            <div className="aw-starting-icon">⏳</div>
            <p>Starting</p>
          </div>
        )}

        {reviewMode && isComplete && (
          <div className="aw-review-banner">
            <span>Training completed successfully.</span>
            <button className="aw-btn aw-btn--primary" onClick={onComplete}>View Results →</button>
          </div>
        )}

        <div className="aw-progress-section">
          <div className="aw-progress-label">
            {STAGE_LABELS[currentStage] || currentStage} · {progress}%
          </div>
          <div className="aw-progress-bar-bg">
            <div className="aw-progress-bar-fill" style={{ width: `${progress}%` }} />
          </div>
          <div className="aw-stage-pipeline">
            {STAGES.map((s, i) => (
              <div key={s} className="aw-stage-item-wrap">
                <div className={`aw-stage-dot ${i <= stageIndex ? 'aw-stage-dot--done' : ''} ${i === stageIndex ? 'aw-stage-dot--active' : ''}`} />
                <span className="aw-stage-label">{STAGE_LABELS[s]}</span>
                {i < STAGES.length - 1 && <div className={`aw-stage-line ${i < stageIndex ? 'aw-stage-line--done' : ''}`} />}
              </div>
            ))}
          </div>
          {stageIndex < STAGES.length - 1 && (
            <p className="aw-next-stage">Next: {STAGE_LABELS[STAGES[stageIndex + 1]]}</p>
          )}
        </div>

        <div className="aw-logs-section">
          <div className="aw-logs-header">
            <h4>⚡ Live Training Logs</h4>
            <span className={`aw-ws-status ${connected ? 'aw-ws-status--connected' : ''}`}>
              {connected ? '● Connected' : '○ Reconnecting'}
            </span>
            <input
              className="aw-logs-filter"
              placeholder="Filter logs..."
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
            />
            <label className="aw-logs-autoscroll">
              <input type="checkbox" checked={autoScroll} onChange={(e) => setAutoScroll(e.target.checked)} />
              Auto-scroll
            </label>
            <button className="aw-logs-copy" onClick={copyLogs}>Copy</button>
          </div>
          <div className="aw-logs-terminal" ref={logsRef}>
            {filteredMessages.length === 0 && (
              <div className="aw-logs-empty">
                <span className="aw-logs-cursor">⚡</span>
                <span>Waiting for training to start...</span>
              </div>
            )}
            {filteredMessages.map((m, i) => (
              <div key={i} className="aw-log-line">
                <span className="aw-log-time">[{new Date(m.timestamp).toLocaleTimeString()}]</span>
                <span className="aw-log-msg">{m.message}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="aw-step-sidebar">
        <div className="aw-training-status-panel">
          <h4>Training Status</h4>
          <div className="aw-status-row">
            <span>Status</span>
            <span className={`aw-status-badge ${status?.status === 'complete' ? 'aw-status-badge--complete' : status?.status === 'failed' ? 'aw-status-badge--failed' : 'aw-status-badge--running'}`}>
              {status?.status === 'complete' ? 'Complete' : status?.status === 'failed' ? 'Failed' : 'Running'}
            </span>
          </div>
          {!isComplete ? (
            <div className="aw-status-row">
              <span>Training</span>
              <span className="aw-status-badge aw-status-badge--running">{_extractActiveModel(messages)}</span>
            </div>
          ) : (
            <div className="aw-status-row">
              <span>Models Trained</span>
              <span className="aw-status-value">{_countModelsFromLogs(messages)}</span>
            </div>
          )}
          <div className="aw-status-row">
            <span>Total Logs</span>
            <span className="aw-status-value">{messages.length}</span>
          </div>
        </div>
        <div className="aw-tips-panel">
          <h4>💡 Tips</h4>
          <ul>
            <li>Use the filter to search for specific events</li>
            <li>Copy logs for debugging or sharing</li>
            <li>Training time varies based on data size</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
