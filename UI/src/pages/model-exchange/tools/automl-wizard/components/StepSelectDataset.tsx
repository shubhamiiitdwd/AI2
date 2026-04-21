import { useState, useRef, useEffect } from 'react';
import type { DatasetMetadata, DatasetWorkflowInsightResponse } from '../types';
import * as api from '../api';
import { aiSourceDisplay } from '../aiSource';
import CatalogView from '../catalog/CatalogView';

type TaskChoice = 'auto' | 'classification' | 'regression' | 'clustering';

interface Props {
  dataset: DatasetMetadata | null;
  onSelect: (ds: DatasetMetadata) => void;
  /** When set, catalog import runs this (e.g. advance wizard) instead of only `onSelect`. */
  onCatalogImportComplete?: (ds: DatasetMetadata) => void | Promise<void>;
  onClusteringSelect?: (ds: DatasetMetadata) => void;
  onContinue?: (taskChoice?: TaskChoice) => void | Promise<void>;
  onClearDataset?: () => void;
}

export default function StepSelectDataset({
  dataset,
  onSelect,
  onCatalogImportComplete,
  onClusteringSelect,
  onContinue,
  onClearDataset,
}: Props) {
  const [datasets, setDatasets] = useState<DatasetMetadata[]>([]);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const [taskChoice, setTaskChoice] = useState<TaskChoice>('auto');
  const [workflowInsight, setWorkflowInsight] = useState<DatasetWorkflowInsightResponse | null>(null);
  const [insightLoading, setInsightLoading] = useState(false);
  const [catalogOpen, setCatalogOpen] = useState(false);

  useEffect(() => {
    api.listDatasets().then(setDatasets).catch(() => {});
  }, []);

  useEffect(() => {
    if (!dataset?.id) {
      setWorkflowInsight(null);
      return;
    }
    let cancelled = false;
    setInsightLoading(true);
    api.getDatasetWorkflowInsight(dataset.id)
      .then((ins) => {
        if (!cancelled) setWorkflowInsight(ins);
      })
      .catch(() => {
        if (!cancelled) setWorkflowInsight(null);
      })
      .finally(() => {
        if (!cancelled) setInsightLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [dataset?.id]);

  const handleUpload = async (file: File) => {
    setUploading(true);
    try {
      const ds = await api.uploadDataset(file);
      setDatasets((prev) => {
        const filtered = prev.filter((d) => d.filename !== ds.filename);
        return [...filtered, ds];
      });
      onSelect(ds); // sets dataset state, stays on step 0 to show info + task picker
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

  return (
    <div className="aw-step-content">
      <div className="aw-step-main">
        {dataset && onClearDataset && (
          <button className="aw-back-btn" onClick={onClearDataset}>← Choose Different Dataset</button>
        )}

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

            <div className="aw-dataset-nav-actions">
              <button
                type="button"
                className="aw-btn aw-btn--secondary aw-dataset-nav-actions__btn"
                title="Navigation to Data Exchange will be wired later"
                onClick={() => {}}
              >
                Data Exchange
              </button>
            </div>

            {(insightLoading || workflowInsight) && (
              <div className="aw-dataset-insight">
                {insightLoading && (
                  <p className="aw-dataset-insight-loading">Analyzing dataset with AI (Azure when configured)…</p>
                )}
                {!insightLoading && workflowInsight && (
                  <>
                    <div className="aw-dataset-insight-header">
                      <span className="aw-dataset-insight-title">{workflowInsight.headline}</span>
                      {workflowInsight.source && (
                        <span className={`aw-badge ${aiSourceDisplay(workflowInsight.source).badgeClass}`}>
                          {aiSourceDisplay(workflowInsight.source).label}
                        </span>
                      )}
                    </div>
                    <p className="aw-dataset-insight-detail">{workflowInsight.detail}</p>
                    {(workflowInsight.data_characteristics || '').trim() !== '' && (
                      <div className="aw-insight-block">
                        <h5 className="aw-insight-block-title">What kind of data is this?</h5>
                        <p className="aw-insight-block-body">{workflowInsight.data_characteristics}</p>
                      </div>
                    )}
                    {(workflowInsight.preprocessing_guidance || '').trim() !== '' && (
                      <div className="aw-insight-block">
                        <h5 className="aw-insight-block-title">Preprocessing and data quality</h5>
                        <p className="aw-insight-block-body">{workflowInsight.preprocessing_guidance}</p>
                      </div>
                    )}
                    {(workflowInsight.feature_engineering_guidance || '').trim() !== '' && (
                      <div className="aw-insight-block">
                        <h5 className="aw-insight-block-title">Feature engineering</h5>
                        <p className="aw-insight-block-body">{workflowInsight.feature_engineering_guidance}</p>
                      </div>
                    )}
                    {workflowInsight.needs_data_exchange && (
                      <p className="aw-insight-exchange-hint">
                        This dataset likely needs work in <strong>Data Exchange</strong> before reliable AutoML.
                        Use the Data Exchange button above when navigation is available.
                      </p>
                    )}
                  </>
                )}
              </div>
            )}

            <div className="aw-task-picker">
              <label className="aw-task-picker-label">I know the ML task for this dataset:</label>
              <div className="aw-task-picker-pills">
                {(['auto', 'classification', 'regression', 'clustering'] as TaskChoice[]).map((t) => (
                  <button
                    key={t}
                    className={`aw-task-pill ${taskChoice === t ? 'aw-task-pill--active' : ''} ${t !== 'auto' ? `aw-task-pill--${t}` : ''}`}
                    onClick={() => setTaskChoice(t)}
                  >
                    {t === 'auto' ? 'Let AI Decide' : t.charAt(0).toUpperCase() + t.slice(1)}
                  </button>
                ))}
              </div>
              {taskChoice === 'auto' && (
                <p className="aw-task-picker-hint">AI will analyze the dataset and recommend the best task in the next step.</p>
              )}
              {taskChoice === 'clustering' && (
                <p className="aw-task-picker-hint">Dataset will be sent directly to the clustering pipeline (no target column needed).</p>
              )}
            </div>

            <button
              className="aw-btn aw-btn--primary aw-btn--full"
              onClick={async () => {
                if (!dataset) return;
                if (taskChoice === 'clustering' && onClusteringSelect) {
                  onClusteringSelect(dataset);
                  return;
                }
                await onContinue?.(taskChoice);
              }}
            >
              {taskChoice === 'clustering' ? 'Start Clustering Pipeline →' : 'Continue to Configure Data →'}
            </button>
          </div>
        ) : (
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

            <div className="aw-upload-alt">
              <span className="aw-upload-alt-label">or</span>
              <button
                type="button"
                className="aw-btn aw-btn--secondary aw-btn--full"
                onClick={() => setCatalogOpen(true)}
              >
                Browse data.gov.in catalog (India)
              </button>
              <p className="aw-upload-alt-hint">
                Search government datasets, preview rows, then import for training. Requires{' '}
                <code>DATA_GOV_API_KEY</code> in the API <code>.env</code>.
              </p>
            </div>

            {catalogOpen && (
              <div
                className="aw-catalog-modal-backdrop"
                role="presentation"
                onClick={() => setCatalogOpen(false)}
              >
                <div
                  className="aw-catalog-modal"
                  role="dialog"
                  aria-modal="true"
                  aria-label="Government dataset catalog"
                  onClick={(e) => e.stopPropagation()}
                >
                  <button
                    type="button"
                    className="aw-catalog-modal-close"
                    aria-label="Close catalog"
                    onClick={() => setCatalogOpen(false)}
                  >
                    ✕
                  </button>
                  <CatalogView
                    onDatasetImported={async (meta) => {
                      setDatasets((prev) => {
                        const filtered = prev.filter((d) => d.filename !== meta.filename);
                        return [...filtered, meta];
                      });
                      setCatalogOpen(false);
                      if (onCatalogImportComplete) {
                        await onCatalogImportComplete(meta);
                      } else {
                        await onSelect(meta);
                      }
                    }}
                  />
                </div>
              </div>
            )}

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
      </div>

      <div className="aw-step-sidebar">
        <h4>AutoML Workflow:</h4>
        <ul className="aw-workflow-list">
          <li className="aw-workflow-active">Upload or select dataset</li>
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
