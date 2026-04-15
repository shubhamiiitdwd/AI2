import { useState } from 'react';
import {
  BarChart2,
  CheckCircle,
  ChevronLeft,
  Cpu,
  Database,
  Download,
  FileJson,
  FileSpreadsheet,
  Shield,
} from 'lucide-react';
import { API_BASE_URL } from '../utils/api';

type DatasetInfo = {
  name: string;
  rows: number;
  columns: number;
};

type DownloadFormat = 'csv' | 'json' | 'parquet';

type ExportStepProps = {
  onPrev: () => void;
  datasetInfo?: DatasetInfo;
  sessionId?: string;
};

const ExportStep = ({ onPrev, datasetInfo, sessionId }: ExportStepProps) => {
  const info = datasetInfo ?? { name: 'Dataset', rows: 0, columns: 0 };
  const [downloading, setDownloading] = useState<DownloadFormat | null>(null);

  const handleDownload = async (format: DownloadFormat) => {
    if (!sessionId) {
      alert('No active session. Please upload a dataset first.');
      return;
    }

    setDownloading(format);
    try {
      const url = `${API_BASE_URL}/team2_feature_eng/download/${sessionId}?format=${format}`;
      const response = await fetch(url);

      if (!response.ok) {
        const errorData = (await response.json()) as { detail?: string };
        throw new Error(errorData.detail ?? `Failed to download ${format.toUpperCase()} file`);
      }

      const blob = await response.blob();
      const downloadUrl = window.URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = downloadUrl;
      anchor.download = `${info.name.replace(/\.[^.]+$/, '') || 'engineered_dataset'}_engineered.${format}`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      window.URL.revokeObjectURL(downloadUrl);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Download failed';
      alert(`Download failed: ${message}`);
    } finally {
      setDownloading(null);
    }
  };

  return (
    <div>
      <div className="card text-center mb-8">
        <div
          style={{
            display: 'inline-flex',
            justifyContent: 'center',
            alignItems: 'center',
            width: '72px',
            height: '72px',
            backgroundColor: 'var(--surface-success-strong)',
            borderRadius: '50%',
            marginBottom: '20px',
          }}
        >
          <CheckCircle size={36} color="#16a34a" />
        </div>

        <h2 className="mb-4" style={{ fontSize: '28px', fontWeight: 'bold' }}>
          Feature Engineering Complete!
        </h2>
        <p className="text-muted mb-8 text-sm" style={{ maxWidth: '600px', margin: '0 auto 32px' }}>
          Your dataset has been successfully transformed with AI-powered feature engineering.
          Download your ML-ready dataset in your preferred format.
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px', marginBottom: '40px' }}>
          <div style={{ padding: '20px', border: '1px solid var(--border-color)', borderRadius: '12px', textAlign: 'center' }}>
            <p className="text-sm text-muted mb-1">Dataset</p>
            <h3 style={{ fontSize: '16px', fontWeight: 'bold', color: 'var(--text-main)' }}>{info.name}</h3>
          </div>
          <div style={{ padding: '20px', border: '1px solid var(--surface-success-strong)', backgroundColor: 'var(--surface-success-soft)', borderRadius: '12px', textAlign: 'center' }}>
            <p className="text-sm text-muted mb-1">Total Rows</p>
            <h3 style={{ fontSize: '24px', fontWeight: 'bold', color: '#16a34a' }}>{info.rows.toLocaleString()}</h3>
          </div>
          <div style={{ padding: '20px', border: '1px solid var(--surface-success-strong)', backgroundColor: 'var(--surface-success-soft)', borderRadius: '12px', textAlign: 'center' }}>
            <p className="text-sm text-muted mb-1">Total Columns</p>
            <h3 style={{ fontSize: '24px', fontWeight: 'bold', color: '#16a34a' }}>{info.columns}</h3>
          </div>
        </div>

        <div style={{ textAlign: 'left', marginBottom: '48px' }}>
          <h3 style={{ fontSize: '18px', fontWeight: 600, marginBottom: '8px' }}>Download Transformed Dataset</h3>
          <p className="text-muted text-sm mb-6">Choose your preferred format to export the ML-ready dataset</p>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '24px' }}>
            <div style={{ padding: '24px', border: '1px solid var(--border-color)', borderRadius: '12px', textAlign: 'center', display: 'flex', flexDirection: 'column' }}>
              <FileSpreadsheet size={40} color="#10b981" style={{ margin: '0 auto 16px' }} />
              <h4 style={{ fontWeight: 'bold', marginBottom: '8px' }}>CSV Format</h4>
              <p className="text-sm text-muted mb-6 flex-1">Compatible with Excel, Pandas, and most ML tools</p>
              <button className="btn btn-primary" style={{ width: '100%' }} onClick={() => handleDownload('csv')} disabled={downloading !== null}>
                <Download size={16} />
                {downloading === 'csv' ? 'Downloading...' : 'Download CSV'}
              </button>
            </div>

            <div style={{ padding: '24px', border: '1px solid var(--border-color)', borderRadius: '12px', textAlign: 'center', display: 'flex', flexDirection: 'column' }}>
              <FileJson size={40} color="#3b82f6" style={{ margin: '0 auto 16px' }} />
              <h4 style={{ fontWeight: 'bold', marginBottom: '8px' }}>JSON Format</h4>
              <p className="text-sm text-muted mb-6 flex-1">Perfect for APIs and web applications</p>
              <button className="btn btn-outline" style={{ width: '100%' }} onClick={() => handleDownload('json')} disabled={downloading !== null}>
                <Download size={16} />
                {downloading === 'json' ? 'Downloading...' : 'Download JSON'}
              </button>
            </div>

            <div style={{ padding: '24px', border: '1px solid var(--border-color)', borderRadius: '12px', textAlign: 'center', display: 'flex', flexDirection: 'column' }}>
              <Database size={40} color="#8b5cf6" style={{ margin: '0 auto 16px' }} />
              <h4 style={{ fontWeight: 'bold', marginBottom: '8px' }}>Parquet Format</h4>
              <p className="text-sm text-muted mb-6 flex-1">Optimized for big data and Apache Spark</p>
              <button className="btn btn-outline" style={{ width: '100%' }} onClick={() => handleDownload('parquet')} disabled={downloading !== null}>
                <Download size={16} />
                {downloading === 'parquet' ? 'Downloading...' : 'Download Parquet'}
              </button>
            </div>
          </div>

          <p className="text-sm text-muted mt-6 text-center" style={{ fontStyle: 'italic' }}>
            All transformations, new features, scaling, and data quality improvements are preserved in the exported file.
          </p>
        </div>

        <div style={{ textAlign: 'left', borderTop: '1px solid var(--border-color)', paddingTop: '32px' }}>
          <h3 style={{ fontSize: '18px', fontWeight: 600, marginBottom: '8px' }}>Continue with Other Features</h3>
          <p className="text-muted text-sm mb-6">Use your transformed dataset with other AI Sphere features</p>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '24px' }}>
            <div style={{ padding: '20px', backgroundColor: 'var(--surface-soft)', borderRadius: '12px', border: '1px solid var(--border-color)', display: 'flex', gap: '16px' }}>
              <div style={{ backgroundColor: 'var(--surface-muted)', padding: '12px', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                <Cpu color="#4f46e5" size={24} />
              </div>
              <div>
                <h4 style={{ fontWeight: 'bold', marginBottom: '4px' }}>AutoML Training</h4>
                <p className="text-sm text-muted mb-2">Build ML models</p>
              </div>
            </div>

            <div style={{ padding: '20px', backgroundColor: 'var(--surface-soft)', borderRadius: '12px', border: '1px solid var(--border-color)', display: 'flex', gap: '16px' }}>
              <div style={{ backgroundColor: 'var(--surface-danger-soft)', padding: '12px', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                <Shield color="#ef4444" size={24} />
              </div>
              <div>
                <h4 style={{ fontWeight: 'bold', marginBottom: '4px' }}>Data Anonymization</h4>
                <p className="text-sm text-muted mb-2">Privacy protection</p>
              </div>
            </div>

            <div style={{ padding: '20px', backgroundColor: 'var(--surface-soft)', borderRadius: '12px', border: '1px solid var(--border-color)', display: 'flex', gap: '16px' }}>
              <div style={{ backgroundColor: 'var(--surface-warn-soft)', padding: '12px', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                <BarChart2 color="#d97706" size={24} />
              </div>
              <div>
                <h4 style={{ fontWeight: 'bold', marginBottom: '4px' }}>Data Visualization</h4>
                <p className="text-sm text-muted mb-2">Explore with charts</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <button className="btn-outline btn text-sm" onClick={onPrev} style={{ borderRadius: '8px' }}>
          <ChevronLeft size={16} /> Back to Data Preview
        </button>
        <div style={{ display: 'flex', gap: '16px', fontSize: '13px', color: 'var(--text-muted)' }}>
          <div style={{ border: '1px solid var(--border-color)', borderRadius: '999px', padding: '4px 12px' }}>Dataset: {info.name}</div>
          <div style={{ border: '1px solid var(--border-color)', borderRadius: '999px', padding: '4px 12px' }}>{info.rows} rows</div>
          <div style={{ border: '1px solid var(--border-color)', borderRadius: '999px', padding: '4px 12px' }}>{info.columns} columns</div>
        </div>
        <div />
      </div>
    </div>
  );
};

export default ExportStep;
