import { useState, useRef, useEffect } from 'react';
import type { DatasetMetadata } from '../types';
import * as api from '../api';

type TaskChoice = 'auto' | 'classification' | 'regression' | 'clustering';

interface Props {
  dataset: DatasetMetadata | null;
  onSelect: (ds: DatasetMetadata) => void;
  onClusteringSelect?: (ds: DatasetMetadata) => void;
  onContinue?: (taskChoice?: TaskChoice) => void;
  onClearDataset?: () => void;
}

export default function StepSelectDataset({ dataset, onSelect, onClusteringSelect, onContinue, onClearDataset }: Props) {
  const [datasets, setDatasets] = useState<DatasetMetadata[]>([]);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const [taskChoice, setTaskChoice] = useState<TaskChoice>('auto');

  useEffect(() => {
    api.listDatasets().then(setDatasets).catch(() => {});
  }, []);

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
              onClick={() => {
                if (taskChoice === 'clustering' && onClusteringSelect) {
                  onClusteringSelect(dataset);
                } else {
                  onContinue?.(taskChoice);
                }
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
