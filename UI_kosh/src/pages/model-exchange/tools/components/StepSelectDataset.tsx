import { useState, useRef, useEffect } from 'react';
import type { DatasetMetadata, HFDatasetInfo } from '../types';
import * as api from '../api';

interface Props {
  dataset: DatasetMetadata | null;
  onSelect: (ds: DatasetMetadata) => void;
}

const TASK_COLORS: Record<string, string> = {
  classification: '#27ae60',
  regression: '#2563eb',
  clustering: '#9333ea',
};

export default function StepSelectDataset({ dataset, onSelect }: Props) {
  const [datasets, setDatasets] = useState<DatasetMetadata[]>([]);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const [activeSource, setActiveSource] = useState<'local' | 'huggingface'>('local');
  const [hfDatasets, setHfDatasets] = useState<HFDatasetInfo[]>([]);
  const [hfFilter, setHfFilter] = useState<string>('all');
  const [hfLoading, setHfLoading] = useState(false);
  const [importing, setImporting] = useState<string | null>(null);

  useEffect(() => {
    api.listDatasets().then(setDatasets).catch(() => {});
  }, []);

  useEffect(() => {
    if (activeSource === 'huggingface' && hfDatasets.length === 0) {
      setHfLoading(true);
      api.browseHFDatasets()
        .then(setHfDatasets)
        .catch(() => {})
        .finally(() => setHfLoading(false));
    }
  }, [activeSource]);

  const handleUpload = async (file: File) => {
    setUploading(true);
    try {
      const ds = await api.uploadDataset(file);
      setDatasets((prev) => {
        const filtered = prev.filter((d) => d.filename !== ds.filename);
        return [...filtered, ds];
      });
      onSelect(ds);
    } catch {
      alert('Upload failed. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (e: React.MouseEvent, ds: DatasetMetadata) => {
    e.stopPropagation();
    try {
      await api.deleteDataset(ds.id);
      setDatasets((prev) => prev.filter((d) => d.id !== ds.id));
    } catch { /* ignore */ }
  };

  const handleImportHF = async (hfId: string) => {
    setImporting(hfId);
    try {
      const ds = await api.importHFDataset(hfId);
      setDatasets((prev) => {
        const filtered = prev.filter((d) => d.filename !== ds.filename);
        return [...filtered, ds];
      });
      onSelect(ds);
    } catch {
      alert('Import failed. Please try again.');
    } finally {
      setImporting(null);
    }
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith('.csv')) handleUpload(file);
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const filteredHF = hfFilter === 'all'
    ? hfDatasets
    : hfDatasets.filter((d) => d.task === hfFilter);

  return (
    <div className="aw-step-content">
      <div className="aw-step-main">
        <button className="aw-back-btn" disabled>← Back</button>

        {dataset ? (
          <div className="aw-dataset-info">
            <h3 className="aw-dataset-name">{dataset.filename}</h3>
            <p className="aw-dataset-desc">{dataset.description}</p>
            <div className="aw-meta-cards">
              <div className="aw-meta-card">
                <span className="aw-meta-label">Total Rows</span>
                <span className="aw-meta-value">{dataset.total_rows}</span>
              </div>
              <div className="aw-meta-card">
                <span className="aw-meta-label">Total Columns</span>
                <span className="aw-meta-value">{dataset.total_columns}</span>
              </div>
              <div className="aw-meta-card">
                <span className="aw-meta-label">Size</span>
                <span className="aw-meta-value">{formatSize(dataset.size_bytes)}</span>
              </div>
              <div className="aw-meta-card">
                <span className="aw-meta-label">Category</span>
                <span className="aw-badge">{dataset.category}</span>
              </div>
            </div>
          </div>
        ) : (
          <>
            {/* Source Tabs */}
            <div className="aw-results-tabs" style={{ marginBottom: 20 }}>
              <button
                className={`aw-tab ${activeSource === 'local' ? 'aw-tab--active' : ''}`}
                onClick={() => setActiveSource('local')}
              >
                Upload / My Datasets
              </button>
              <button
                className={`aw-tab ${activeSource === 'huggingface' ? 'aw-tab--active' : ''}`}
                onClick={() => setActiveSource('huggingface')}
              >
                🤗 HuggingFace Datasets
              </button>
            </div>

            {activeSource === 'local' && (
              <>
                <div
                  className={`aw-upload-zone ${dragOver ? 'aw-upload-zone--active' : ''}`}
                  onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={onDrop}
                  onClick={() => fileRef.current?.click()}
                >
                  <div className="aw-upload-icon">📁</div>
                  <p>{uploading ? 'Uploading...' : 'Drag & drop a CSV file here, or click to browse'}</p>
                  <input
                    ref={fileRef}
                    type="file"
                    accept=".csv"
                    hidden
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) handleUpload(file);
                    }}
                  />
                </div>

                {datasets.length > 0 && (
                  <div className="aw-dataset-catalog">
                    <h4>Or select from existing datasets:</h4>
                    <div className="aw-dataset-list">
                      {datasets.map((ds) => (
                        <div key={ds.id} className="aw-dataset-item" onClick={() => onSelect(ds)}>
                          <span className="aw-dataset-item-name">{ds.filename}</span>
                          <div className="aw-dataset-item-right">
                            <span className="aw-dataset-item-meta">{ds.total_rows} rows · {ds.total_columns} cols</span>
                            <button className="aw-dataset-delete-btn" onClick={(e) => handleDelete(e, ds)} title="Delete dataset">✕</button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}

            {activeSource === 'huggingface' && (
              <div className="aw-hf-browser">
                <p className="aw-hf-desc">
                  Browse curated small datasets from HuggingFace. Select one to import it instantly for training.
                </p>

                {/* Task filter pills */}
                <div className="aw-hf-filters">
                  {['all', 'classification', 'regression', 'clustering'].map((t) => (
                    <button
                      key={t}
                      className={`aw-hf-filter-pill ${hfFilter === t ? 'aw-hf-filter-pill--active' : ''}`}
                      onClick={() => setHfFilter(t)}
                    >
                      {t === 'all' ? 'All Tasks' : t.charAt(0).toUpperCase() + t.slice(1)}
                    </button>
                  ))}
                </div>

                {hfLoading ? (
                  <div className="aw-loading">Loading datasets from HuggingFace...</div>
                ) : (
                  <div className="aw-hf-grid">
                    {filteredHF.map((ds) => (
                      <div key={ds.hf_id} className="aw-hf-card">
                        <div className="aw-hf-card-header">
                          <span className="aw-hf-card-name">{ds.name}</span>
                          <span
                            className="aw-hf-task-badge"
                            style={{ background: `${TASK_COLORS[ds.task] || '#888'}20`, color: TASK_COLORS[ds.task] || '#888' }}
                          >
                            {ds.task}
                          </span>
                        </div>
                        <p className="aw-hf-card-desc">{ds.description}</p>
                        <div className="aw-hf-card-meta">
                          <span>{ds.rows.toLocaleString()} rows</span>
                          <span>{ds.cols} cols</span>
                          <span>{ds.size_kb < 1024 ? `${ds.size_kb} KB` : `${(ds.size_kb / 1024).toFixed(1)} MB`}</span>
                          <span className="aw-hf-target-hint">Target: {ds.target_hint}</span>
                        </div>
                        <div className="aw-hf-card-footer">
                          <a
                            className="aw-hf-card-source"
                            href={ds.hf_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            onClick={(e) => e.stopPropagation()}
                            title="View on HuggingFace"
                          >
                            🤗 {ds.hf_id}
                          </a>
                          <button
                            className="aw-btn aw-btn--primary aw-hf-import-btn"
                            onClick={() => handleImportHF(ds.hf_id)}
                            disabled={importing !== null}
                          >
                            {importing === ds.hf_id ? 'Importing...' : 'Import & Use'}
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {!hfLoading && filteredHF.length === 0 && (
                  <div className="aw-loading">No datasets found for this task filter.</div>
                )}
              </div>
            )}
          </>
        )}
      </div>

      <div className="aw-step-sidebar">
        <h4>AutoML Workflow:</h4>
        <ul className="aw-workflow-list">
          <li className="aw-workflow-active">Search catalog for datasets</li>
          <li>Preview and configure data columns</li>
          <li>Select ML task and models</li>
          <li>Configure training settings</li>
          <li>Train and compare models</li>
          <li>View results and export</li>
        </ul>
      </div>
    </div>
  );
}
