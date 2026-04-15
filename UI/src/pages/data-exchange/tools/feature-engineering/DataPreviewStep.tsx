import { useEffect, useMemo, useState } from 'react';
import { ChevronLeft, ChevronRight, Send, Terminal } from 'lucide-react';
import { API_BASE_URL } from '../utils/api';

type DatasetInfo = {
  name: string;
  rows: number;
  columns: number;
};

type PreviewValue = string | number | boolean | null;
type PreviewData = Record<string, PreviewValue[]>;

type CustomTransformResponse = {
  preview?: Record<string, unknown>;
};

type DataPreviewStepProps = {
  onNext: () => void;
  onPrev: () => void;
  data?: Record<string, unknown>;
  datasetInfo?: DatasetInfo;
  sessionId?: string;
};

const normalizePreviewData = (raw: unknown): PreviewData => {
  if (!raw || typeof raw !== 'object') return {};

  const output: PreviewData = {};
  Object.entries(raw as Record<string, unknown>).forEach(([key, value]) => {
    if (!Array.isArray(value)) return;
    output[key] = value.map((item) => {
      if (item === null || typeof item === 'string' || typeof item === 'number' || typeof item === 'boolean') {
        return item;
      }
      return String(item);
    });
  });

  return output;
};

const DataPreviewStep = ({ onNext, onPrev, data, datasetInfo, sessionId }: DataPreviewStepProps) => {
  const info = datasetInfo ?? { name: 'Dataset', rows: 0, columns: 0 };

  const [localData, setLocalData] = useState<PreviewData>(normalizePreviewData(data ?? {}));
  const [prompt, setPrompt] = useState('');
  const [isApplying, setIsApplying] = useState(false);
  const [customCounter, setCustomCounter] = useState(0);

  useEffect(() => {
    if (!sessionId) return;

    const fetchPreview = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/team2_feature_eng/preview/${sessionId}`);
        if (!res.ok) throw new Error('Failed to load dataset preview.');

        const result = (await res.json()) as { preview?: Record<string, unknown> };
        setLocalData(normalizePreviewData(result.preview ?? {}));
      } catch {
        // Keep last available preview in UI when request fails.
      }
    };

    void fetchPreview();
  }, [sessionId]);

  const handleCustomTransform = async () => {
    if (!prompt.trim() || !sessionId) return;

    setIsApplying(true);
    try {
      const res = await fetch(`${API_BASE_URL}/team2_feature_eng/custom-transform/apply`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, prompt }),
      });

      const result = (await res.json()) as CustomTransformResponse & { detail?: string };
      if (!res.ok) throw new Error(result.detail ?? 'Transformation failed');

      const updated = normalizePreviewData(result.preview ?? {});
      setLocalData((prev) => ({ ...prev, ...updated }));
      setCustomCounter((prev) => prev + 1);
      setPrompt('');
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Transformation failed';
      alert(message);
    } finally {
      setIsApplying(false);
    }
  };

  const columns = useMemo(() => Object.keys(localData), [localData]);
  const rowCount = columns.length > 0 ? localData[columns[0]].length : 0;

  const rows = useMemo(() => {
    const output: Record<string, PreviewValue>[] = [];
    for (let rowIndex = 0; rowIndex < rowCount; rowIndex += 1) {
      const row: Record<string, PreviewValue> = {};
      columns.forEach((col) => {
        row[col] = localData[col][rowIndex] ?? null;
      });
      output.push(row);
    }
    return output;
  }, [columns, localData, rowCount]);

  return (
    <div>
      <div className="card text-center mb-8">
        <h2 className="mb-4" style={{ fontSize: '24px', fontWeight: 'bold' }}>
          Data Preview & Custom Transformations
        </h2>
        <p className="text-muted mb-8 text-sm">
          Review your transformed data and apply additional custom changes
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', marginBottom: '32px' }}>
          <div style={{ padding: '16px', border: '1px solid var(--border-color)', borderRadius: '12px', textAlign: 'left' }}>
            <p className="text-sm text-muted mb-1">Transformations Applied</p>
            <h3 style={{ fontSize: '20px', fontWeight: 'bold' }}>{columns.length}</h3>
            <p className="text-sm" style={{ color: 'var(--success)' }}>
              New Features Added
            </p>
          </div>
          <div style={{ padding: '16px', border: '1px solid var(--border-color)', borderRadius: '12px', textAlign: 'left' }}>
            <p className="text-sm text-muted mb-1">Total Rows</p>
            <h3 style={{ fontSize: '24px', fontWeight: 'bold' }}>{info.rows}</h3>
          </div>
          <div style={{ padding: '16px', border: '1px solid var(--border-color)', borderRadius: '12px', textAlign: 'left' }}>
            <p className="text-sm text-muted mb-1">Total Columns</p>
            <h3 style={{ fontSize: '24px', fontWeight: 'bold' }}>{info.columns + columns.length}</h3>
          </div>
          <div style={{ padding: '16px', border: '1px solid var(--border-color)', borderRadius: '12px', textAlign: 'left' }}>
            <p className="text-sm text-muted mb-1">Custom Transformations</p>
            <h3 style={{ fontSize: '24px', fontWeight: 'bold' }}>{customCounter}</h3>
          </div>
        </div>

        <div style={{ textAlign: 'left', marginBottom: '32px' }}>
          <h3 style={{ fontSize: '18px', fontWeight: 600, marginBottom: '4px' }}>Latest Dataset Mapping</h3>
          <p className="text-muted text-sm mb-4">
            Showing first {rows.length} rows of the dataset state spanning {columns.length} features.
          </p>

          <div
            style={{
              border: '1px solid var(--border-color)',
              borderRadius: '12px',
              overflow: 'hidden',
              overflowX: 'auto',
              maxWidth: '100vw',
            }}
          >
            <table className="data-table" style={{ fontSize: '13px' }}>
              <thead>
                <tr>
                  {columns.map((key) => (
                    <th key={key}>{key}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row, rowIndex) => (
                  <tr key={rowIndex}>
                    {columns.map((col) => (
                      <td key={`${rowIndex}-${col}`}>{String(row[col] ?? '')}</td>
                    ))}
                  </tr>
                ))}
                {rows.length === 0 && (
                  <tr>
                    <td colSpan={Math.max(columns.length, 1)} style={{ padding: '32px', textAlign: 'center', color: 'var(--text-muted)' }}>
                      Apply features to see a preview here.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        <div
          style={{
            textAlign: 'left',
            border: '1px solid var(--border-color)',
            padding: '24px',
            borderRadius: '12px',
            backgroundColor: 'var(--surface-soft)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
            <Terminal size={18} color="var(--primary)" />
            <h4 style={{ fontWeight: 600, fontSize: '16px' }}>Custom Transformation</h4>
          </div>
          <p className="text-sm text-muted mb-4">Describe any additional changes you'd like to make to your data</p>

          <div style={{ position: 'relative' }}>
            <textarea
              className="text-sm w-full"
              style={{
                width: '100%',
                padding: '16px',
                borderRadius: '8px',
                border: '1px solid var(--border-color)',
                minHeight: '100px',
                resize: 'vertical',
              }}
              placeholder="E.g., 'Create a new column that is the ratio of column A to column B'"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
            />
            <button
              className="btn btn-primary"
              style={{ position: 'absolute', bottom: '16px', right: '16px', padding: '8px 16px' }}
              onClick={handleCustomTransform}
              disabled={isApplying}
            >
              {isApplying ? 'Applying...' : (<><Send size={16} /> Generate & Apply</>)}
            </button>
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <button className="btn-outline btn text-sm" onClick={onPrev} style={{ borderRadius: '8px' }}>
          <ChevronLeft size={16} /> Back to Feature Generation
        </button>
        <div style={{ display: 'flex', gap: '16px', fontSize: '13px', color: 'var(--text-muted)' }}>
          <div style={{ border: '1px solid var(--border-color)', borderRadius: '999px', padding: '4px 12px' }}>
            Dataset: {info.name}
          </div>
          <div style={{ border: '1px solid var(--border-color)', borderRadius: '999px', padding: '4px 12px' }}>
            {info.rows} rows
          </div>
          <div style={{ border: '1px solid var(--border-color)', borderRadius: '999px', padding: '4px 12px' }}>
            {info.columns + columns.length} columns
          </div>
        </div>
        <button className="btn btn-primary" onClick={onNext}>
          Continue to Export <ChevronRight size={16} />
        </button>
      </div>
    </div>
  );
};

export default DataPreviewStep;
