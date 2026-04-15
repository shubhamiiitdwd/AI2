import { useEffect, useMemo, useState } from 'react';
import {
  CheckCircle,
  CheckSquare,
  ChevronLeft,
  Code,
  Cpu,
  Edit2,
  Layout,
  Minimize2,
  MoveHorizontal,
  Play,
  Square,
} from 'lucide-react';
import { API_BASE_URL } from '../utils/api';

type DatasetInfo = {
  name: string;
  rows: number;
  columns: number;
};

type StageId = 'creation' | 'transformation' | 'extraction' | 'scaling' | 'selection';
type StageAction = 'suggestions' | 'regenerate' | 'apply';

type FeatureSuggestion = {
  id: string;
  name: string;
  desc: string;
  reasoning: string;
  python_code: string;
  selected: boolean;
};

type StageData = {
  title: string;
  desc: string;
  items: FeatureSuggestion[];
};

type StageResult = {
  preview?: Record<string, unknown>;
};

type FeatureGenerationStepProps = {
  onNext: (result?: StageResult) => void;
  onPrev: () => void;
  sessionId?: string;
  datasetInfo?: DatasetInfo;
};

const STAGES: { id: StageId; title: string; icon: React.ElementType }[] = [
  { id: 'creation', title: 'Feature Creation', icon: Layout },
  { id: 'transformation', title: 'Transformation', icon: MoveHorizontal },
  { id: 'extraction', title: 'Feature Extraction', icon: Minimize2 },
  { id: 'scaling', title: 'Scaling', icon: Cpu },
  { id: 'selection', title: 'Feature Selection', icon: CheckCircle },
];

const INITIAL_STAGE_DATA: Record<StageId, StageData> = {
  creation: { title: 'Feature Creation', desc: 'Generate new features from existing columns', items: [] },
  transformation: { title: 'Transformation', desc: 'Apply mathematical transformations', items: [] },
  extraction: { title: 'Feature Extraction', desc: 'Extract latent features via dimensionality reduction', items: [] },
  scaling: { title: 'Scaling', desc: 'Normalize features to similar ranges', items: [] },
  selection: { title: 'Feature Selection', desc: 'Remove redundant or low-value features', items: [] },
};

const getStagePath = (stageId: StageId, action: StageAction): string => {
  switch (stageId) {
    case 'creation':
      return `/team2_feature_eng/features/${action}`;
    case 'transformation':
      return `/team2_feature_eng/transformations/${action}`;
    case 'extraction':
      return `/team2_feature_eng/extractions/${action}`;
    case 'scaling':
      return `/team2_feature_eng/scaling/${action}`;
    case 'selection':
      return `/team2_feature_eng/selection/${action}`;
    default:
      return `/team2_feature_eng/features/${action}`;
  }
};

const toSuggestion = (raw: Record<string, unknown>, index: number): FeatureSuggestion => ({
  id: typeof raw.feature_code === 'string' ? raw.feature_code : `feature-${index}`,
  name: typeof raw.feature_name === 'string' ? raw.feature_name : `feature_${index + 1}`,
  desc: typeof raw.description === 'string' ? raw.description : 'No description provided',
  reasoning: typeof raw.reasoning === 'string' ? raw.reasoning : 'No reasoning provided',
  python_code: typeof raw.python_code === 'string' ? raw.python_code : '',
  selected: true,
});

const FeatureGenerationStep = ({ onNext, onPrev, sessionId, datasetInfo }: FeatureGenerationStepProps) => {
  const info = datasetInfo ?? { name: 'Dataset', rows: 0, columns: 0 };

  const [stageIndex, setStageIndex] = useState(0);
  const [stageData, setStageData] = useState<Record<StageId, StageData>>(INITIAL_STAGE_DATA);
  const [isLoading, setIsLoading] = useState(false);
  const [isApplying, setIsApplying] = useState(false);
  const [viewCodeRow, setViewCodeRow] = useState<FeatureSuggestion | null>(null);
  const [editRow, setEditRow] = useState<FeatureSuggestion | null>(null);
  const [modifyPrompt, setModifyPrompt] = useState('');
  const [isRegenerating, setIsRegenerating] = useState(false);

  const currentStage = STAGES[stageIndex];
  const currentData = stageData[currentStage.id];
  const items = currentData.items;

  useEffect(() => {
    const fetchSuggestions = async () => {
      if (!sessionId || items.length > 0) return;

      setIsLoading(true);
      try {
        const url = `${API_BASE_URL}${getStagePath(currentStage.id, 'suggestions')}`;
        const res = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: sessionId,
            num_suggestions: currentStage.id === 'selection' ? 10 : 5,
          }),
        });

        if (!res.ok) throw new Error('Failed to fetch feature suggestions');

        const payload = (await res.json()) as { suggestions?: Array<Record<string, unknown>> };
        const mapped = (payload.suggestions ?? []).map((item, index) => toSuggestion(item, index));

        setStageData((prev) => ({
          ...prev,
          [currentStage.id]: { ...prev[currentStage.id], items: mapped },
        }));
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : 'Failed to fetch suggestions';
        alert(message);
      } finally {
        setIsLoading(false);
      }
    };

    void fetchSuggestions();
  }, [currentStage.id, items.length, sessionId]);

  const toggleSelect = (id: string) => {
    setStageData((prev) => {
      const updated = prev[currentStage.id].items.map((item) =>
        item.id === id ? { ...item, selected: !item.selected } : item
      );
      return { ...prev, [currentStage.id]: { ...prev[currentStage.id], items: updated } };
    });
  };

  const toggleAll = () => {
    const allSelected = items.every((item) => item.selected);
    setStageData((prev) => {
      const updated = prev[currentStage.id].items.map((item) => ({ ...item, selected: !allSelected }));
      return { ...prev, [currentStage.id]: { ...prev[currentStage.id], items: updated } };
    });
  };

  const handleRegenerate = async () => {
    if (!modifyPrompt.trim() || !editRow || !sessionId) return;

    setIsRegenerating(true);
    try {
      const url = `${API_BASE_URL}${getStagePath(currentStage.id, 'regenerate')}`;
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          feature_code: editRow.id,
          modification_request: modifyPrompt,
          current_reasoning: editRow.reasoning,
          current_code: editRow.python_code,
        }),
      });
      if (!res.ok) throw new Error('Failed to regenerate suggestion');

      const regenerated = toSuggestion((await res.json()) as Record<string, unknown>, 0);
      regenerated.selected = true;

      setStageData((prev) => {
        const updated = prev[currentStage.id].items.map((item) => (item.id === editRow.id ? regenerated : item));
        return { ...prev, [currentStage.id]: { ...prev[currentStage.id], items: updated } };
      });

      setEditRow(null);
      setModifyPrompt('');
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to regenerate suggestion';
      alert(message);
    } finally {
      setIsRegenerating(false);
    }
  };

  const nextStage = async () => {
    if (!sessionId) {
      if (stageIndex < STAGES.length - 1) setStageIndex((prev) => prev + 1);
      else onNext({});
      return;
    }

    const selectedFeatures = items
      .filter((item) => item.selected)
      .map((item) => ({ feature_code: item.id, python_code: item.python_code }));

    if (selectedFeatures.length > 0) {
      setIsApplying(true);
      try {
        const url = `${API_BASE_URL}${getStagePath(currentStage.id, 'apply')}`;
        const res = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: sessionId, selected_features: selectedFeatures }),
        });

        if (!res.ok) {
          const errorData = (await res.json()) as { detail?: string };
          throw new Error(errorData.detail ?? 'Failed to apply features for this stage');
        }

        const result = (await res.json()) as StageResult;
        if (stageIndex < STAGES.length - 1) {
          setStageIndex((prev) => prev + 1);
        } else {
          onNext(result);
        }
        return;
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : 'Failed to apply stage';
        alert(message);
        return;
      } finally {
        setIsApplying(false);
      }
    }

    if (stageIndex < STAGES.length - 1) setStageIndex((prev) => prev + 1);
    else onNext({ preview: {} });
  };

  const selectedCount = useMemo(() => items.filter((item) => item.selected).length, [items]);

  return (
    <div>
      <div className="card text-center mb-8">
        <h2 className="mb-4" style={{ fontSize: '24px', fontWeight: 'bold' }}>
          AI-Powered Feature Engineering
        </h2>
        <p className="text-muted mb-8 text-sm">Systematically generate, transform, and select features</p>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
          <div>
            <h4 style={{ fontWeight: 600, fontSize: '14px', margin: 0 }}>Progress</h4>
            <p className="text-sm text-muted">Stage {stageIndex + 1} of 5</p>
          </div>
          <button className="btn btn-text text-sm" onClick={() => onNext({ preview: {} })}>
            Skip to Preview
          </button>
        </div>

        <div style={{ display: 'flex', borderBottom: '1px solid var(--border-color)', marginBottom: '32px' }}>
          {STAGES.map((stage, index) => {
            const Icon = stage.icon;
            const isActive = stageIndex === index;
            const isCompleted = index < stageIndex;
            return (
              <div
                key={stage.id}
                style={{
                  flex: 1,
                  padding: '12px 0',
                  fontSize: '13px',
                  fontWeight: 500,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  color: isActive ? 'var(--primary)' : isCompleted ? 'var(--text-main)' : 'var(--text-muted)',
                  borderBottom: isActive ? '2px solid var(--primary)' : '2px solid transparent',
                  opacity: !isActive && !isCompleted ? 0.5 : 1,
                }}
              >
                <Icon size={18} style={{ marginBottom: '4px' }} />
                {stage.title}
              </div>
            );
          })}
        </div>

        <div style={{ textAlign: 'left', marginBottom: '32px' }}>
          <h3 style={{ fontSize: '20px', fontWeight: 600, marginBottom: '4px' }}>{currentData.title}</h3>
          <p className="text-muted text-sm mb-6">{currentData.desc}</p>

          {isLoading ? (
            <div style={{ padding: '64px', textAlign: 'center', backgroundColor: 'var(--surface-soft)', borderRadius: '12px', border: '1px dashed var(--border-color)' }}>
              <div
                style={{
                  display: 'inline-block',
                  border: '3px solid var(--surface-muted)',
                  borderRadius: '50%',
                  borderTopColor: 'var(--primary)',
                  width: '32px',
                  height: '32px',
                  animation: 'spin 1s linear infinite',
                  marginBottom: '16px',
                }}
              />
              <h4 style={{ fontWeight: 600 }}>AI is generating {currentStage.title.toLowerCase()} suggestions...</h4>
            </div>
          ) : items.length === 0 ? (
            <div style={{ padding: '64px', textAlign: 'center', backgroundColor: 'var(--surface-soft)', borderRadius: '12px', border: '1px dashed var(--border-color)' }}>
              <h4 style={{ fontWeight: 600 }}>No suggestions generated.</h4>
            </div>
          ) : (
            <>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <h4 style={{ fontWeight: 600, fontSize: '14px' }}>AI Suggestions ({selectedCount}/{items.length} selected)</h4>
                <button className="btn btn-outline btn-sm" onClick={toggleAll}>
                  {items.every((item) => item.selected) ? <Square size={16} /> : <CheckSquare size={16} />}
                  {items.every((item) => item.selected) ? 'Deselect All' : 'Select All'}
                </button>
              </div>

              <div style={{ border: '1px solid var(--border-color)', borderRadius: '12px', overflow: 'hidden' }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th style={{ width: '60px', textAlign: 'center' }}>Select</th>
                      <th>Feature Name</th>
                      <th>Description</th>
                      <th>Reasoning</th>
                      <th style={{ textAlign: 'center', width: '80px' }}>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {items.map((row) => (
                      <tr key={row.id}>
                        <td style={{ textAlign: 'center' }} onClick={() => toggleSelect(row.id)}>
                          <div style={{ cursor: 'pointer', display: 'flex', justifyContent: 'center', color: row.selected ? 'var(--primary)' : 'var(--border-color)' }}>
                            {row.selected ? <CheckSquare size={20} /> : <Square size={20} />}
                          </div>
                        </td>
                        <td>
                          <span style={{ fontWeight: 600 }}>{row.name}</span>
                        </td>
                        <td>
                          <p className="text-sm text-muted">{row.desc}</p>
                        </td>
                        <td>
                          <p className="text-sm text-muted" style={{ fontStyle: 'italic', backgroundColor: 'var(--surface-muted)', padding: '8px', borderRadius: '4px' }}>
                            {row.reasoning}
                          </p>
                        </td>
                        <td>
                          <div style={{ display: 'flex', gap: '8px', justifyContent: 'center' }}>
                            <button
                              onClick={() => setViewCodeRow(row)}
                              className="btn btn-icon btn-sm"
                              title="View Code"
                              style={{ backgroundColor: 'transparent', border: 'none', cursor: 'pointer', padding: '4px', color: 'var(--text-color)' }}
                            >
                              <Code size={18} />
                            </button>
                            <button
                              onClick={() => setEditRow(row)}
                              className="btn btn-icon btn-sm"
                              title="Modify Suggestion"
                              style={{ backgroundColor: 'transparent', border: 'none', cursor: 'pointer', padding: '4px', color: 'var(--text-color)' }}
                            >
                              <Edit2 size={18} />
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <button className="btn-outline btn text-sm" onClick={onPrev} style={{ borderRadius: '8px' }}>
          <ChevronLeft size={16} /> Back to {stageIndex === 0 ? 'Data Quality' : STAGES[stageIndex - 1].title}
        </button>
        <div style={{ display: 'flex', gap: '16px', fontSize: '13px', color: 'var(--text-muted)' }}>
          <div style={{ border: '1px solid var(--border-color)', borderRadius: '999px', padding: '4px 12px' }}>Dataset: {info.name}</div>
          <div style={{ border: '1px solid var(--border-color)', borderRadius: '999px', padding: '4px 12px' }}>{info.rows} rows</div>
          <div style={{ border: '1px solid var(--border-color)', borderRadius: '999px', padding: '4px 12px' }}>{info.columns} columns</div>
        </div>
        <div style={{ display: 'flex', gap: '16px' }}>
          <button className="btn btn-text text-sm" onClick={() => (stageIndex < STAGES.length - 1 ? setStageIndex((prev) => prev + 1) : onNext({ preview: {} }))}>
            Skip Stage
          </button>
          <button className="btn btn-primary" onClick={nextStage} disabled={isLoading || isApplying}>
            {isApplying ? 'Applying...' : (<><Play size={16} /> {stageIndex === STAGES.length - 1 ? 'Apply & Continue' : 'Apply & Next Stage'}</>)}
          </button>
        </div>
      </div>

      {viewCodeRow && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0,0,0,0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
          }}
        >
          <div style={{ backgroundColor: 'var(--bg-panel)', padding: '24px', borderRadius: '12px', width: '500px', maxWidth: '90%' }}>
            <h3 style={{ fontSize: '18px', fontWeight: 600, marginBottom: '16px' }}>Implementation Code</h3>
            <p className="text-sm text-muted mb-2">
              Code for feature: <span style={{ fontWeight: 600 }}>{viewCodeRow.name}</span>
            </p>
            <div
              style={{
                backgroundColor: 'var(--code-bg)',
                color: 'var(--code-text)',
                padding: '16px',
                borderRadius: '8px',
                fontFamily: 'monospace',
                fontSize: '14px',
                marginBottom: '24px',
                overflowX: 'auto',
                whiteSpace: 'pre-wrap',
              }}
            >
              {viewCodeRow.python_code || 'No code available.'}
            </div>
            <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
              <button className="btn btn-outline" onClick={() => setViewCodeRow(null)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {editRow && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0,0,0,0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
          }}
        >
          <div style={{ backgroundColor: 'var(--bg-panel)', padding: '24px', borderRadius: '12px', width: '500px', maxWidth: '90%' }}>
            <h3 style={{ fontSize: '18px', fontWeight: 600, marginBottom: '16px' }}>Modify Suggestion</h3>
            <p className="text-sm text-muted mb-4">
              Provide instructions to modify the AI suggestion for: <span style={{ fontWeight: 600 }}>{editRow.name}</span>
            </p>

            <div style={{ marginBottom: '16px' }}>
              <label style={{ display: 'block', fontSize: '13px', fontWeight: 600, marginBottom: '8px', color: 'var(--text-muted)' }}>
                Current Reasoning:
              </label>
              <div style={{ backgroundColor: 'var(--surface-soft)', padding: '12px', borderRadius: '8px', fontSize: '14px' }}>{editRow.reasoning}</div>
            </div>

            <div style={{ marginBottom: '24px' }}>
              <label style={{ display: 'block', fontSize: '13px', fontWeight: 600, marginBottom: '8px' }}>Your Modification Request:</label>
              <textarea
                className="form-control"
                rows={3}
                placeholder="E.g., 'Use a ratio instead' or 'Scale it logarithmically'"
                value={modifyPrompt}
                onChange={(e) => setModifyPrompt(e.target.value)}
                style={{ width: '100%', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-color)', resize: 'vertical' }}
              />
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px' }}>
              <button
                className="btn btn-text"
                onClick={() => {
                  setEditRow(null);
                  setModifyPrompt('');
                }}
              >
                Cancel
              </button>
              <button className="btn btn-primary" onClick={handleRegenerate} disabled={isRegenerating}>
                {isRegenerating ? 'Regenerating...' : 'Regenerate'}
              </button>
            </div>
          </div>
        </div>
      )}

      <style dangerouslySetInnerHTML={{ __html: '@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }' }} />
    </div>
  );
};

export default FeatureGenerationStep;
