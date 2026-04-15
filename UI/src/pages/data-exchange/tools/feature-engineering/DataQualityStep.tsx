import { useEffect, useState } from 'react';
import { Check, CheckSquare, ChevronLeft, Code, Edit2, Info, Square } from 'lucide-react';
import { API_BASE_URL } from '../utils/api';

type DatasetInfo = {
  name: string;
  rows: number;
  columns: number;
};

type QualitySuggestion = {
  id: string;
  column: string;
  issue: string;
  display_name: string;
  operation: string;
  reason: string;
  code?: string;
  selected: boolean;
};

type ApplyFixResult = {
  preview?: Record<string, unknown>;
};

type DataQualityStepProps = {
  onNext: (result?: ApplyFixResult) => void;
  onPrev: () => void;
  sessionId?: string;
  datasetInfo?: DatasetInfo;
};

const toSuggestion = (raw: Record<string, unknown>, index: number): QualitySuggestion => {
  const id = typeof raw.id === 'string' ? raw.id : `fix-${index}`;
  const column = typeof raw.column === 'string' ? raw.column : 'unknown_column';
  const issue = typeof raw.issue === 'string' ? raw.issue : 'quality_issue';
  const displayName = typeof raw.display_name === 'string' ? raw.display_name : '';
  const operation = typeof raw.operation === 'string' ? raw.operation : '';
  const reason = typeof raw.reason === 'string' ? raw.reason : 'No reason provided';
  const code = typeof raw.code === 'string' ? raw.code : undefined;

  return {
    id,
    column,
    issue,
    display_name: displayName,
    operation,
    reason,
    code,
    selected: true,
  };
};

const DataQualityStep = ({ onNext, onPrev, sessionId, datasetInfo }: DataQualityStepProps) => {
  const info = datasetInfo ?? { name: 'Dataset', rows: 0, columns: 0 };

  const [selections, setSelections] = useState<QualitySuggestion[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isApplying, setIsApplying] = useState(false);
  const [outliers, setOutliers] = useState(0);
  const [viewCodeRow, setViewCodeRow] = useState<QualitySuggestion | null>(null);
  const [editRow, setEditRow] = useState<QualitySuggestion | null>(null);
  const [modifyPrompt, setModifyPrompt] = useState('');
  const [isRegenerating, setIsRegenerating] = useState(false);

  useEffect(() => {
    const fetchQuality = async () => {
      if (!sessionId) {
        setIsLoading(false);
        return;
      }

      setIsLoading(true);
      try {
        const res = await fetch(`${API_BASE_URL}/team2_feature_eng/data-quality/${sessionId}`);
        if (!res.ok) throw new Error('Failed to fetch data quality suggestions');

        const payload = (await res.json()) as {
          suggestions?: Array<Record<string, unknown>>;
          total_suggestions?: number;
        };

        const mapped = (payload.suggestions ?? []).map((item, index) => toSuggestion(item, index));
        setSelections(mapped);
        setOutliers(typeof payload.total_suggestions === 'number' ? payload.total_suggestions : mapped.length);
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : 'Failed to fetch suggestions';
        alert(message);
      } finally {
        setIsLoading(false);
      }
    };

    void fetchQuality();
  }, [sessionId]);

  const handleApplyFixes = async () => {
    const selectedIds = selections.filter((item) => item.selected).map((item) => item.id);
    if (selectedIds.length === 0) {
      onNext();
      return;
    }

    if (!sessionId) {
      onNext();
      return;
    }

    setIsApplying(true);
    try {
      const res = await fetch(`${API_BASE_URL}/team2_feature_eng/apply-fixes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          fix_ids: selectedIds,
        }),
      });

      if (!res.ok) throw new Error('Failed to apply quality fixes');
      const result = (await res.json()) as ApplyFixResult;
      onNext(result);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to apply fixes';
      alert(message);
    } finally {
      setIsApplying(false);
    }
  };

  const handleRegenerate = async () => {
    if (!modifyPrompt.trim() || !editRow || !sessionId) return;

    setIsRegenerating(true);
    try {
      const res = await fetch(`${API_BASE_URL}/team2_feature_eng/data-quality/regenerate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          fix_id: editRow.id,
          prompt: modifyPrompt,
        }),
      });
      if (!res.ok) throw new Error('Regeneration failed');

      const regenerated = toSuggestion((await res.json()) as Record<string, unknown>, 0);
      regenerated.selected = true;

      setSelections((prev) => prev.map((fix) => (fix.id === editRow.id ? regenerated : fix)));
      setEditRow(null);
      setModifyPrompt('');
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to regenerate';
      alert(message);
    } finally {
      setIsRegenerating(false);
    }
  };

  const toggleSelect = (id: string) => {
    setSelections((prev) => prev.map((item) => (item.id === id ? { ...item, selected: !item.selected } : item)));
  };

  const toggleAll = () => {
    const allSelected = selections.every((item) => item.selected);
    setSelections((prev) => prev.map((item) => ({ ...item, selected: !allSelected })));
  };

  const selectedCount = selections.filter((item) => item.selected).length;

  return (
    <div>
      <div className="card text-center mb-8">
        <h2 className="mb-4" style={{ fontSize: '24px', fontWeight: 'bold' }}>
          Data Quality Improvement
        </h2>
        <p className="text-muted mb-8 text-sm">
          Review and approve AI-generated solutions for missing values, outliers, and duplicates
        </p>

        {isLoading ? (
          <div style={{ padding: '64px', textAlign: 'center' }}>
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
            <p>AI is analyzing data quality...</p>
          </div>
        ) : (
          <>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '24px', marginBottom: '32px' }}>
              <div style={{ padding: '24px', border: '1px solid var(--border-color)', borderRadius: '12px', textAlign: 'center' }}>
                <h4 className="text-muted text-sm mb-2">Missing Values</h4>
                <div style={{ fontSize: '32px', fontWeight: 'bold', color: 'var(--success)' }}>0</div>
              </div>
              <div style={{ padding: '24px', border: '1px solid var(--primary)', backgroundColor: 'var(--surface-warn-soft)', borderRadius: '12px', textAlign: 'center' }}>
                <h4 className="text-primary text-sm mb-2" style={{ color: 'var(--primary)', fontWeight: 600 }}>
                  Suggestions
                </h4>
                <div style={{ fontSize: '32px', fontWeight: 'bold', color: 'var(--primary)' }}>{outliers}</div>
              </div>
              <div style={{ padding: '24px', border: '1px solid var(--border-color)', borderRadius: '12px', textAlign: 'center' }}>
                <h4 className="text-muted text-sm mb-2">Duplicates</h4>
                <div style={{ fontSize: '24px', marginTop: '4px', fontWeight: 'bold', color: 'var(--text-muted)' }}>Coming Soon</div>
              </div>
            </div>

            <div style={{ textAlign: 'left', marginBottom: '32px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <div>
                  <h3 style={{ fontSize: '18px', fontWeight: 600, marginBottom: '4px' }}>
                    Quality Improvement Plan ({selectedCount} selected)
                  </h3>
                  <p className="text-muted text-sm">Review each suggestion and select which ones to apply.</p>
                </div>
                <button className="btn btn-outline btn-sm" onClick={toggleAll}>
                  {selections.length > 0 && selections.every((item) => item.selected) ? <Square size={16} /> : <CheckSquare size={16} />}
                  Toggle All
                </button>
              </div>

              <div style={{ border: '1px solid var(--border-color)', borderRadius: '12px', overflow: 'hidden' }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th style={{ width: '60px', textAlign: 'center' }}>Select</th>
                      <th>Column</th>
                      <th>Issue Type</th>
                      <th>Fix</th>
                      <th>Reasoning</th>
                      <th style={{ textAlign: 'center', width: '80px' }}>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {selections.map((row) => (
                      <tr key={row.id}>
                        <td style={{ textAlign: 'center' }} onClick={() => toggleSelect(row.id)}>
                          <div style={{ cursor: 'pointer', display: 'flex', justifyContent: 'center', color: row.selected ? 'var(--primary)' : 'var(--border-color)' }}>
                            {row.selected ? <CheckSquare size={20} /> : <Square size={20} />}
                          </div>
                        </td>
                        <td>
                          <span style={{ fontWeight: 600 }}>{row.column}</span>
                        </td>
                        <td>
                          <span className="badge badge-orange">{row.issue}</span>
                        </td>
                        <td>
                          <code style={{ backgroundColor: 'var(--surface-muted)', padding: '4px 8px', borderRadius: '4px', fontSize: '13px' }}>
                            {row.display_name || row.operation || '-'}
                          </code>
                        </td>
                        <td>
                          <p className="text-sm text-muted">{row.reason}</p>
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
                    {selections.length === 0 && (
                      <tr>
                        <td colSpan={6} style={{ padding: '32px', textAlign: 'center', color: 'var(--text-muted)' }}>
                          No quality issues detected.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}

        <div style={{ textAlign: 'left', border: '1px dashed var(--border-color)', padding: '24px', borderRadius: '12px', backgroundColor: 'var(--surface-soft)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
            <Info size={18} color="var(--text-muted)" />
            <h4 style={{ fontWeight: 600 }}>Duplicate Handling</h4>
          </div>
          <p className="text-sm text-muted">Duplicate detection and removal will be implemented in the next update.</p>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <button className="btn-outline btn text-sm" onClick={onPrev} style={{ borderRadius: '8px' }}>
          <ChevronLeft size={16} /> Back to Data Profile
        </button>
        <div style={{ display: 'flex', gap: '16px', fontSize: '13px', color: 'var(--text-muted)' }}>
          <div style={{ border: '1px solid var(--border-color)', borderRadius: '999px', padding: '4px 12px' }}>Dataset: {info.name}</div>
          <div style={{ border: '1px solid var(--border-color)', borderRadius: '999px', padding: '4px 12px' }}>{info.rows} rows</div>
          <div style={{ border: '1px solid var(--border-color)', borderRadius: '999px', padding: '4px 12px' }}>{info.columns} columns</div>
        </div>
        <div style={{ display: 'flex', gap: '16px' }}>
          <button className="btn btn-text text-sm" onClick={() => onNext()}>
            Skip Quality Fixes
          </button>
          <button className="btn btn-primary" onClick={handleApplyFixes} disabled={isLoading || isApplying}>
            {isApplying ? 'Applying...' : (<><Check size={16} /> Apply Selected Fixes ({selectedCount})</>)}
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
              Code for column: <span style={{ fontWeight: 600 }}>{viewCodeRow.column}</span>
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
              {viewCodeRow.code ?? 'No code available.'}
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
              Provide instructions to modify the AI suggestion for: <span style={{ fontWeight: 600 }}>{editRow.column}</span>
            </p>

            <div style={{ marginBottom: '16px' }}>
              <label style={{ display: 'block', fontSize: '13px', fontWeight: 600, marginBottom: '8px', color: 'var(--text-muted)' }}>
                Current Reasoning:
              </label>
              <div style={{ backgroundColor: 'var(--surface-soft)', padding: '12px', borderRadius: '8px', fontSize: '14px' }}>
                {editRow.reason}
              </div>
            </div>

            <div style={{ marginBottom: '24px' }}>
              <label style={{ display: 'block', fontSize: '13px', fontWeight: 600, marginBottom: '8px' }}>Your Modification Request:</label>
              <textarea
                className="form-control"
                rows={3}
                placeholder="E.g., 'Use forward fill instead of mean imputation'"
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

export default DataQualityStep;
