import { useState, useEffect, useRef } from 'react';
import { X, Database, Calendar, Layers, Table as TableIcon, Loader2, Download } from 'lucide-react';
import type { DatasetEntry } from './services';
import type { DatasetMetadata } from '../types';
import { fetchDatasetSample, downloadDatasetCSV, importDataGovForAutoml } from './services';

interface Props {
  dataset: DatasetEntry;
  onClose: () => void;
  onImportComplete?: (meta: DatasetMetadata) => void;
}

type Tab = 'overview' | 'preview';

const DatasetDetailsHub = ({ dataset, onClose, onImportComplete }: Props) => {
  const [tab, setTab] = useState<Tab>('overview');
  const [sample, setSample] = useState<{
    records: Record<string, unknown>[];
    field: { id?: string; type?: string; name?: string }[];
    total?: number;
    error?: string;
  } | null>(null);
  const [sampleLoading, setSampleLoading] = useState(false);
  const [importPct, setImportPct] = useState<number | null>(null);
  const [importBusy, setImportBusy] = useState(false);
  const [importErr, setImportErr] = useState<string | null>(null);
  const panelRef = useRef<HTMLDivElement>(null);

  const indexId = dataset.id?.trim() || '';

  useEffect(() => {
    if (tab !== 'preview' || !indexId) return;
    let cancelled = false;
    setSampleLoading(true);
    setSample(null);
    fetchDatasetSample(indexId)
      .then((res) => {
        if (!cancelled)
          setSample({
            records: (res.records as Record<string, unknown>[]) || [],
            field: (res.field as { id?: string; type?: string; name?: string }[]) || [],
            total: typeof res.total === 'number' ? res.total : undefined,
            error: res.error,
          });
      })
      .catch(() => {
        if (!cancelled) setSample({ records: [], field: [], error: 'Failed to load preview' });
      })
      .finally(() => {
        if (!cancelled) setSampleLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [tab, indexId, dataset.name]);

  const runImport = async () => {
    if (!indexId || !onImportComplete || importBusy) return;
    setImportErr(null);
    setImportBusy(true);
    setImportPct(0);
    try {
      const meta = await importDataGovForAutoml(
        indexId,
        dataset.name,
        dataset.desc,
        (e) => setImportPct(e.pct),
        100_000,
      );
      onImportComplete(meta);
      onClose();
    } catch (e) {
      setImportErr(e instanceof Error ? e.message : 'Import failed');
    } finally {
      setImportBusy(false);
      setImportPct(null);
    }
  };

  return (
    <>
      <style>{`
        .dg-hub-backdrop { position: fixed; inset: 0; background: rgba(15, 23, 42, 0.55); z-index: 1200; }
        .dg-hub-panel {
          position: fixed; top: 0; right: 0; width: min(440px, 100vw); height: 100vh; z-index: 1201;
          background: var(--bg-elevated, #1e293b); border-left: 1px solid var(--border-mid, rgba(255,255,255,0.1));
          display: flex; flex-direction: column; box-shadow: -8px 0 32px rgba(0,0,0,0.35);
        }
        .dg-hub-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; padding: 16px 18px; border-bottom: 1px solid var(--border-dim, rgba(255,255,255,0.08)); }
        .dg-hub-title { font-size: 15px; font-weight: 700; color: var(--t-primary, #f1f5f9); margin: 0 0 4px; line-height: 1.3; }
        .dg-hub-org { font-size: 12px; color: var(--t-muted, #94a3b8); margin: 0; }
        .dg-hub-close { flex-shrink: 0; background: transparent; border: none; color: var(--t-secondary, #64748b); cursor: pointer; padding: 4px; border-radius: 6px; }
        .dg-hub-close:hover { color: var(--t-primary, #f1f5f9); background: rgba(255,255,255,0.06); }
        .dg-hub-tabs { display: flex; border-bottom: 1px solid var(--border-dim, rgba(255,255,255,0.08)); }
        .dg-hub-tab {
          flex: 1; display: flex; align-items: center; justify-content: center; gap: 6px; padding: 10px 8px;
          font-size: 12px; font-weight: 600; color: var(--t-muted, #94a3b8); background: none; border: none; cursor: pointer;
          border-bottom: 2px solid transparent; margin-bottom: -1px;
        }
        .dg-hub-tab--on { color: var(--brand, #fb923c); border-bottom-color: var(--brand, #fb923c); }
        .dg-hub-body { flex: 1; overflow: auto; padding: 14px 18px 100px; font-size: 13px; color: var(--t-secondary, #cbd5e1); }
        .dg-hub-foot {
          position: absolute; bottom: 0; left: 0; right: 0; padding: 12px 18px 18px;
          border-top: 1px solid var(--border-dim, rgba(255,255,255,0.08)); background: var(--bg-elevated, #1e293b);
        }
        .dg-hub-actions { display: flex; flex-direction: column; gap: 8px; }
        .dg-hub-row { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--t-muted, #94a3b8); margin-bottom: 8px; }
        .dg-hub-schema { font-family: ui-monospace, monospace; font-size: 11px; background: var(--aw-inset, rgba(0,0,0,0.2)); border-radius: 8px; padding: 10px; max-height: 200px; overflow: auto; white-space: pre-wrap; }
        .dg-hub-table-wrap { overflow: auto; max-height: 320px; border-radius: 8px; border: 1px solid var(--border-dim, rgba(255,255,255,0.08)); }
        .dg-hub-table { width: 100%; border-collapse: collapse; font-size: 11px; }
        .dg-hub-table th, .dg-hub-table td { padding: 6px 8px; text-align: left; border-bottom: 1px solid var(--border-dim, rgba(255,255,255,0.06)); color: var(--t-primary, #e2e8f0); }
        .dg-hub-table th { position: sticky; top: 0; background: var(--aw-inset, #0f172a); z-index: 1; }
        .dg-hub-progress { height: 4px; border-radius: 2px; background: rgba(255,255,255,0.1); overflow: hidden; margin-top: 4px; }
        .dg-hub-progress > i { display: block; height: 100%; background: var(--brand, #fb923c); transition: width 0.2s; }
        @keyframes dg-spin { to { transform: rotate(360deg); } }
        .dg-spin { animation: dg-spin 0.9s linear infinite; display: inline-block; vertical-align: middle; }
      `}</style>
      <div className="dg-hub-backdrop" onClick={onClose} aria-hidden />
      <aside className="dg-hub-panel" ref={panelRef} onClick={(e) => e.stopPropagation()}>
        <header className="dg-hub-head">
          <div>
            <h3 className="dg-hub-title">{dataset.name}</h3>
            <p className="dg-hub-org">{dataset.org}</p>
          </div>
          <button type="button" className="dg-hub-close" onClick={onClose} aria-label="Close">
            <X size={22} />
          </button>
        </header>

        <nav className="dg-hub-tabs">
          <button
            type="button"
            className={`dg-hub-tab ${tab === 'overview' ? 'dg-hub-tab--on' : ''}`}
            onClick={() => setTab('overview')}
          >
            <Layers size={14} /> Overview
          </button>
          <button
            type="button"
            className={`dg-hub-tab ${tab === 'preview' ? 'dg-hub-tab--on' : ''}`}
            onClick={() => setTab('preview')}
          >
            <TableIcon size={14} /> Preview
          </button>
        </nav>

        <div className="dg-hub-body">
          {tab === 'overview' && (
            <>
              {!indexId && (
                <p style={{ color: '#f87171' }}>This listing has no resource id — pick another dataset.</p>
              )}
              {dataset.updated && (
                <div className="dg-hub-row">
                  <Calendar size={14} /> Updated: {dataset.updated}
                </div>
              )}
              <div className="dg-hub-row">
                <Database size={14} /> {dataset.category}
              </div>
              <p style={{ marginTop: 12, lineHeight: 1.5 }}>{dataset.desc || 'No description.'}</p>
              <h4 style={{ margin: '16px 0 8px', fontSize: 13, color: 'var(--t-primary, #f1f5f9)' }}>Schema</h4>
              <div className="dg-hub-schema">
                {(dataset.fields || []).length
                  ? (dataset.fields || [])
                      .map((f) => `${f.name} (${f.type || 'unknown'})`)
                      .join('\n')
                  : 'No field metadata from the catalog API.'}
              </div>
            </>
          )}

          {tab === 'preview' && (
            <>
              {!indexId ? (
                <p>Cannot preview without a resource id.</p>
              ) : sampleLoading ? (
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Loader2 className="dg-spin" size={18} /> Loading sample…
                </div>
              ) : sample?.error ? (
                <p style={{ color: '#f87171' }}>{sample.error}</p>
              ) : sample && sample.records.length > 0 ? (
                <div className="dg-hub-table-wrap">
                  <table className="dg-hub-table">
                    <thead>
                      <tr>
                        {Object.keys(sample.records[0]).map((k) => (
                          <th key={k}>{k}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {sample.records.slice(0, 100).map((row, ri) => (
                        <tr key={ri}>
                          {Object.keys(sample.records[0]).map((k) => (
                            <td key={k}>{String((row as Record<string, unknown>)[k] ?? '')}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p>No rows returned.</p>
              )}
            </>
          )}
        </div>

        <footer className="dg-hub-foot">
          <div className="dg-hub-actions">
            {importErr && <p style={{ color: '#f87171', fontSize: 12, margin: 0 }}>{importErr}</p>}
            {importBusy && importPct !== null && (
              <div>
                <span style={{ fontSize: 11, color: 'var(--t-muted)' }}>Importing… {importPct}%</span>
                <div className="dg-hub-progress">
                  <i style={{ width: `${importPct}%` }} />
                </div>
              </div>
            )}
            {indexId && (
              <>
                {onImportComplete && (
                  <button
                    type="button"
                    className="aw-btn aw-btn--primary aw-btn--full"
                    disabled={importBusy}
                    onClick={() => void runImport()}
                  >
                    {importBusy ? 'Importing…' : 'Import for AutoML training'}
                  </button>
                )}
                <button
                  type="button"
                  className="aw-btn aw-btn--secondary aw-btn--full"
                  disabled={importBusy}
                  onClick={() => indexId && downloadDatasetCSV(indexId, dataset.name)}
                >
                  <Download size={16} style={{ marginRight: 6, verticalAlign: 'middle' }} />
                  Download CSV (browser)
                </button>
              </>
            )}
          </div>
        </footer>
      </aside>
    </>
  );
};

export default DatasetDetailsHub;
