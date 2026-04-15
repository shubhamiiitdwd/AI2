import { useRef, useState } from 'react';
import type { ChangeEvent, DragEvent } from 'react';
import { Database, Target, UploadCloud, X } from 'lucide-react';
import { API_BASE_URL } from '../utils/api';

type SelectDatasetResult = {
  sessionId: string;
  info?: Record<string, unknown>;
  profile?: Record<string, unknown>;
};

type SelectDatasetStepProps = {
  onNext: (result: SelectDatasetResult) => void;
};

const SelectDatasetStep = ({ onNext }: SelectDatasetStepProps) => {
  const [activeTab, setActiveTab] = useState<'catalog' | 'upload'>('upload');
  const [isHovering, setIsHovering] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [targetCol, setTargetCol] = useState('');
  const [availableColumns, setAvailableColumns] = useState<string[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const extractHeaders = (selectedFile: File) => {
    if (!selectedFile.name.toLowerCase().endsWith('.csv')) {
      setAvailableColumns([]);
      return;
    }

    const reader = new FileReader();
    reader.onload = (event: ProgressEvent<FileReader>) => {
      const text = event.target?.result;
      if (typeof text !== 'string') {
        setAvailableColumns([]);
        return;
      }

      const firstLine = text.split('\n')[0];
      if (!firstLine) {
        setAvailableColumns([]);
        return;
      }

      const cols = firstLine.split(',').map((col: string) => col.trim().replace(/^"|"$/g, ''));
      setAvailableColumns(cols.filter((col: string) => col.length > 0));
    };
    reader.readAsText(selectedFile.slice(0, 1024 * 10));
  };

  const handleFileSelect = (selectedFile: File) => {
    setFile(selectedFile);
    setError(null);
    setTargetCol('');
    extractHeaders(selectedFile);
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    handleFileSelect(e.target.files[0]);
  };

  const uploadToBackend = async (selectedFile?: File) => {
    const fileToUpload = selectedFile ?? file;

    if (!fileToUpload) {
      fileInputRef.current?.click();
      return;
    }

    setIsUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', fileToUpload);
    if (targetCol.trim()) {
      formData.append('target_col', targetCol.trim());
    }

    try {
      const uploadRes = await fetch(`${API_BASE_URL}/team2_feature_eng/upload`, {
        method: 'POST',
        body: formData,
      });

      const uploadData = (await uploadRes.json()) as Record<string, unknown>;
      if (!uploadRes.ok) {
        const detail = typeof uploadData.detail === 'string' ? uploadData.detail : 'Upload failed';
        throw new Error(detail);
      }

      const sessionId = typeof uploadData.session_id === 'string' ? uploadData.session_id : '';
      if (!sessionId) throw new Error('Missing session id from upload response');

      const profileRes = await fetch(`${API_BASE_URL}/team2_feature_eng/profile/${sessionId}`);
      const profileData = (await profileRes.json()) as Record<string, unknown>;
      if (!profileRes.ok) throw new Error('Failed to profile dataset');

      onNext({
        sessionId,
        info: (uploadData.info as Record<string, unknown>) ?? {},
        profile: profileData,
      });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Upload failed';
      setError(message);
    } finally {
      setIsUploading(false);
    }
  };

  const onDropFile = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsHovering(false);

    if (!e.dataTransfer.files || e.dataTransfer.files.length === 0) return;
    handleFileSelect(e.dataTransfer.files[0]);
  };

  return (
    <div className="card main-content text-center">
      <h2 className="mb-4" style={{ fontSize: '24px', fontWeight: 'bold' }}>
        Select Your Dataset
      </h2>
      <p className="text-muted mb-8 text-sm">
        Upload a CSV or Excel file to begin AI-powered feature engineering
      </p>

      <div className="flex" style={{ borderBottom: '1px solid var(--border-color)', marginBottom: '32px' }}>
        <div
          onClick={() => setActiveTab('catalog')}
          style={{
            flex: 1,
            padding: '16px',
            cursor: 'pointer',
            borderBottom: activeTab === 'catalog' ? '2px solid var(--primary)' : '2px solid transparent',
            color: activeTab === 'catalog' ? 'var(--primary)' : 'var(--text-muted)',
          }}
        >
          <Database size={18} style={{ display: 'inline', marginRight: '8px' }} />
          <span style={{ fontWeight: 500 }}>Catalog</span>
        </div>
        <div
          onClick={() => setActiveTab('upload')}
          style={{
            flex: 1,
            padding: '16px',
            cursor: 'pointer',
            borderBottom: activeTab === 'upload' ? '2px solid var(--primary)' : '2px solid transparent',
            color: activeTab === 'upload' ? 'var(--primary)' : 'var(--text-muted)',
          }}
        >
          <UploadCloud size={18} style={{ display: 'inline', marginRight: '8px' }} />
          <span style={{ fontWeight: 500 }}>Upload Dataset</span>
        </div>
      </div>

      {activeTab === 'upload' && (
        <>
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setIsHovering(true);
            }}
            onDragLeave={() => setIsHovering(false)}
            onDrop={onDropFile}
            style={{
              border: `2px dashed ${isHovering ? 'var(--primary)' : file ? '#10b981' : 'var(--border-color)'}`,
              borderRadius: '12px',
              padding: '48px',
              backgroundColor: isHovering ? 'var(--surface-warn-soft)' : file ? 'var(--surface-success-soft)' : 'var(--surface-soft)',
              transition: 'all 0.2s',
              marginBottom: '24px',
            }}
          >
            <div
              style={{
                display: 'inline-flex',
                justifyContent: 'center',
                alignItems: 'center',
                width: '64px',
                height: '64px',
                backgroundColor: file ? 'var(--surface-success-strong)' : 'var(--surface-muted)',
                borderRadius: '12px',
                marginBottom: '16px',
              }}
            >
              <UploadCloud size={32} color={file ? '#16a34a' : 'var(--text-muted)'} />
            </div>

            <h3 className="mb-4" style={{ fontWeight: 600, fontSize: '18px' }}>
              {file ? `✓ ${file.name}` : 'Upload Your Dataset'}
            </h3>
            <p className="text-muted mb-8 text-sm">
              {file
                ? 'File selected. Click "Upload & Process" below to continue.'
                : 'Drag & drop a CSV or Excel file here, or click the button below'}
            </p>

            <input
              type="file"
              accept=".csv,.xlsx,.xls"
              ref={fileInputRef}
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />

            <button
              className="btn btn-primary"
              onClick={() => fileInputRef.current?.click()}
              style={{ marginBottom: '8px' }}
            >
              <UploadCloud size={16} /> Choose File
            </button>

            {file && (
              <button
                onClick={() => {
                  setFile(null);
                  setError(null);
                  setAvailableColumns([]);
                }}
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'var(--text-muted)',
                  cursor: 'pointer',
                  marginLeft: '12px',
                  fontSize: '13px',
                }}
              >
                <X size={14} style={{ display: 'inline', marginRight: '4px' }} /> Clear
              </button>
            )}
          </div>

          <div
            style={{
              marginBottom: '24px',
              textAlign: 'left',
              background: 'var(--surface-soft)',
              padding: '16px',
              borderRadius: '10px',
              border: '1px solid var(--border-color)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
              <Target size={16} color="var(--primary)" />
              <label style={{ fontWeight: 600, fontSize: '14px' }}>Target Column (Optional)</label>
            </div>
            <p style={{ fontSize: '13px', color: 'var(--text-muted)', marginBottom: '10px' }}>
              Which column should the AI try to predict? Leave blank and the AI will auto-detect it.
            </p>

            {availableColumns.length > 0 ? (
              <select
                value={targetCol}
                onChange={(e) => setTargetCol(e.target.value)}
                style={{
                  width: '100%',
                  padding: '10px 12px',
                  borderRadius: '8px',
                  border: '1px solid var(--border-color)',
                  fontSize: '14px',
                  backgroundColor: 'var(--bg-panel)',
                }}
              >
                <option value="">-- Let AI Auto-Detect --</option>
                {availableColumns.map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))}
              </select>
            ) : (
              <input
                type="text"
                value={targetCol}
                onChange={(e) => setTargetCol(e.target.value)}
                placeholder="e.g., price, salary, churn, survived"
                style={{
                  width: '100%',
                  padding: '10px 12px',
                  borderRadius: '8px',
                  border: '1px solid var(--border-color)',
                  fontSize: '14px',
                }}
              />
            )}
          </div>

          {error && (
            <div
              style={{
                backgroundColor: 'var(--surface-danger-soft)',
                border: '1px solid var(--border-danger)',
                color: 'var(--text-danger)',
                padding: '12px 16px',
                borderRadius: '8px',
                marginBottom: '16px',
                textAlign: 'left',
                fontSize: '14px',
              }}
            >
              ⚠️ {error}
            </div>
          )}

          <button
            className="btn btn-primary"
            style={{ width: '100%', padding: '14px' }}
            onClick={() => uploadToBackend()}
            disabled={isUploading || !file}
          >
            <UploadCloud size={16} />
            {isUploading ? 'Uploading & Profiling...' : 'Upload & Process Dataset'}
          </button>
        </>
      )}

      {activeTab === 'catalog' && (
        <div className="text-left" style={{ padding: '32px' }}>
          <div
            style={{
              border: '1px solid var(--border-color)',
              borderRadius: '8px',
              padding: '16px',
              display: 'flex',
              gap: '8px',
              alignItems: 'center',
              color: 'var(--text-muted)',
            }}
          >
            Search datasets by name, description, or tags...
          </div>
          <p className="mt-4 text-sm text-muted">Dataset catalog integration coming soon.</p>
        </div>
      )}

      <style
        dangerouslySetInnerHTML={{
          __html: '@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }',
        }}
      />
    </div>
  );
};

export default SelectDatasetStep;
