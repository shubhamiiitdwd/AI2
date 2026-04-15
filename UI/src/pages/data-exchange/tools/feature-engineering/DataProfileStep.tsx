import { useMemo, useState } from 'react';
import { AlertCircle, ChevronLeft, ChevronRight } from 'lucide-react';

type DatasetInfo = {
  name: string;
  rows: number;
  columns: number;
};

type ProfileColumnDetail = {
  column: string;
  dtype: string;
  unique: number;
  missing: number;
  mean?: number;
  median?: number;
  std?: number;
  min?: number;
  max?: number;
  skew?: number;
  outlier_percent?: number;
};

type ProfilePayload = {
  overview?: {
    rows: number;
    columns: number;
    missing_percent: number;
    memory_mb: number;
  };
  types?: {
    numeric: number;
    categorical: number;
    datetime: number;
  };
  quality?: {
    completeness: number;
    feature_richness: number;
  };
  column_details?: ProfileColumnDetail[];
};

type DataProfileStepProps = {
  onNext: () => void;
  onPrev: () => void;
  data?: Record<string, unknown>;
  datasetInfo?: DatasetInfo;
};

type ProfileTab = 'Overview' | 'Data Quality' | 'Distributions' | 'Missing Values' | 'Column Details';

const DataProfileStep = ({ onNext, onPrev, data, datasetInfo }: DataProfileStepProps) => {
  const [activeTab, setActiveTab] = useState<ProfileTab>('Overview');

  const payload = (data ?? {}) as ProfilePayload;
  const info = datasetInfo ?? { name: 'Dataset', rows: 0, columns: 0 };

  const overview = payload.overview ?? {
    rows: info.rows,
    columns: info.columns,
    missing_percent: 0,
    memory_mb: 0,
  };

  const types = payload.types ?? { numeric: 0, categorical: 0, datetime: 0 };
  const quality = payload.quality ?? { completeness: 0, feature_richness: 0 };
  const columns = payload.column_details ?? [];

  const numericCols = useMemo(
    () => columns.filter((col) => col.dtype.includes('int') || col.dtype.includes('float')),
    [columns]
  );
  const categoricalCols = useMemo(
    () => columns.filter((col) => !col.dtype.includes('int') && !col.dtype.includes('float')),
    [columns]
  );

  return (
    <div>
      <div className="card text-center mb-8">
        <h2 className="mb-4" style={{ fontSize: '24px', fontWeight: 'bold' }}>
          Dataset Profile
        </h2>
        <p className="text-muted mb-8 text-sm">Comprehensive analysis of your dataset structure and quality</p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '24px', marginBottom: '32px' }}>
          <div style={{ padding: '24px', border: '1px solid var(--border-color)', borderRadius: '12px', textAlign: 'left' }}>
            <p className="text-sm text-muted mb-2">Total Rows</p>
            <h3 style={{ fontSize: '28px', fontWeight: 'bold' }}>{overview.rows}</h3>
          </div>
          <div style={{ padding: '24px', border: '1px solid var(--border-color)', borderRadius: '12px', textAlign: 'left' }}>
            <p className="text-sm text-muted mb-2">Total Columns</p>
            <h3 style={{ fontSize: '28px', fontWeight: 'bold' }}>{overview.columns}</h3>
          </div>
          <div style={{ padding: '24px', border: '1px solid var(--border-color)', borderRadius: '12px', textAlign: 'left' }}>
            <p className="text-sm text-muted mb-2">Missing Data</p>
            <h3 style={{ fontSize: '28px', fontWeight: 'bold' }}>{overview.missing_percent.toFixed(2)}%</h3>
          </div>
        </div>

        <div style={{ display: 'flex', borderBottom: '1px solid var(--border-color)', marginBottom: '32px' }}>
          {(['Overview', 'Data Quality', 'Distributions', 'Missing Values', 'Column Details'] as ProfileTab[]).map((tab) => (
            <div
              key={tab}
              onClick={() => setActiveTab(tab)}
              style={{
                flex: 1,
                padding: '12px 16px',
                fontSize: '14px',
                fontWeight: 500,
                cursor: 'pointer',
                color: activeTab === tab ? 'var(--text-main)' : 'var(--text-muted)',
                borderBottom: activeTab === tab ? '2px solid var(--primary)' : '2px solid transparent',
              }}
            >
              {tab}
            </div>
          ))}
        </div>

        {activeTab === 'Overview' && (
          <div style={{ textAlign: 'left', padding: '24px', border: '1px solid var(--border-color)', borderRadius: '12px' }}>
            <h4 style={{ fontWeight: 600, fontSize: '18px', marginBottom: '8px' }}>Column Type Distribution</h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px', marginBottom: '24px' }}>
              <div style={{ padding: '12px', border: '1px solid var(--border-color)', borderRadius: '8px' }}>Numeric: {types.numeric}</div>
              <div style={{ padding: '12px', border: '1px solid var(--border-color)', borderRadius: '8px' }}>
                Categorical: {types.categorical}
              </div>
              <div style={{ padding: '12px', border: '1px solid var(--border-color)', borderRadius: '8px' }}>Datetime: {types.datetime}</div>
            </div>

            <h4 style={{ fontWeight: 600, fontSize: '18px', marginBottom: '8px' }}>Data Health</h4>
            <div style={{ marginBottom: '12px' }}>Completeness: {quality.completeness.toFixed(1)}%</div>
            <div style={{ marginBottom: '12px' }}>Feature Richness: {quality.feature_richness.toFixed(1)}%</div>
            <div>Estimated Memory: {overview.memory_mb.toFixed(2)} MB</div>
          </div>
        )}

        {activeTab === 'Data Quality' && (
          <div style={{ textAlign: 'left', padding: '24px', border: '1px solid var(--border-color)', borderRadius: '12px' }}>
            <h4 style={{ fontWeight: 600, fontSize: '18px', marginBottom: '12px' }}>Quality Score Breakdown</h4>
            <div style={{ marginBottom: '8px' }}>Completeness Score: {quality.completeness.toFixed(1)}%</div>
            <div style={{ marginBottom: '8px' }}>Feature Richness: {quality.feature_richness.toFixed(1)}%</div>
            <div>Columns with missing values: {columns.filter((col) => col.missing > 0).length}</div>
          </div>
        )}

        {activeTab === 'Distributions' && (
          <div style={{ textAlign: 'left', padding: '24px', border: '1px solid var(--border-color)', borderRadius: '12px' }}>
            <h4 style={{ fontWeight: 600, fontSize: '18px', marginBottom: '12px' }}>Numeric Summary</h4>
            <div style={{ border: '1px solid var(--border-color)', borderRadius: '8px', overflow: 'hidden' }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>Mean</th>
                    <th>Median</th>
                    <th>Skew</th>
                    <th>Outliers</th>
                  </tr>
                </thead>
                <tbody>
                  {numericCols.map((col) => (
                    <tr key={col.column}>
                      <td>{col.column}</td>
                      <td>{(col.mean ?? 0).toFixed(2)}</td>
                      <td>{(col.median ?? 0).toFixed(2)}</td>
                      <td>{(col.skew ?? 0).toFixed(2)}</td>
                      <td>{(col.outlier_percent ?? 0).toFixed(2)}%</td>
                    </tr>
                  ))}
                  {numericCols.length === 0 && (
                    <tr>
                      <td colSpan={5} style={{ padding: '16px', textAlign: 'center' }}>
                        No numeric columns found.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {activeTab === 'Missing Values' && (
          <div
            style={{
              textAlign: 'center',
              padding: '64px',
              border: '1px solid var(--border-color)',
              borderRadius: '12px',
              backgroundColor: overview.missing_percent > 0 ? 'var(--surface-danger-soft)' : 'var(--surface-success-soft)',
            }}
          >
            <AlertCircle
              size={48}
              color={overview.missing_percent > 0 ? 'var(--error)' : 'var(--success)'}
              style={{ margin: '0 auto 16px' }}
            />
            <h4
              style={{
                fontWeight: 600,
                fontSize: '20px',
                color: overview.missing_percent > 0 ? 'var(--error)' : 'var(--success)',
                marginBottom: '8px',
              }}
            >
              {overview.missing_percent > 0 ? `${overview.missing_percent.toFixed(2)}% Missing Data` : 'No Missing Values'}
            </h4>
            <p className="text-muted text-sm">
              {overview.missing_percent > 0
                ? 'Your dataset contains missing values that may need treatment.'
                : 'Your dataset is complete with no missing data.'}
            </p>
          </div>
        )}

        {activeTab === 'Column Details' && (
          <div style={{ textAlign: 'left', padding: '24px', border: '1px solid var(--border-color)', borderRadius: '12px' }}>
            <h4 style={{ fontWeight: 600, fontSize: '18px', marginBottom: '16px' }}>Numeric Columns ({types.numeric})</h4>
            {numericCols.map((col) => (
              <div key={col.column} style={{ padding: '10px 0', borderBottom: '1px solid var(--border-color)' }}>
                <div style={{ fontWeight: 600 }}>{col.column}</div>
                <div className="text-sm text-muted">
                  Range: [{(col.min ?? 0).toFixed(2)}, {(col.max ?? 0).toFixed(2)}] | Mean: {(col.mean ?? 0).toFixed(2)}
                </div>
              </div>
            ))}

            <h4 style={{ fontWeight: 600, fontSize: '18px', marginTop: '20px', marginBottom: '16px' }}>
              Categorical Columns ({types.categorical})
            </h4>
            {categoricalCols.map((col) => (
              <div key={col.column} style={{ padding: '10px 0', borderBottom: '1px solid var(--border-color)' }}>
                <div style={{ fontWeight: 600 }}>{col.column}</div>
                <div className="text-sm text-muted">{col.unique} unique values</div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div style={{ display: 'flex', alignItems: 'center' }}>
        <button className="btn-outline btn text-sm" onClick={onPrev} style={{ borderRadius: '8px' }}>
          <ChevronLeft size={16} /> Back to Select Dataset
        </button>
        <div style={{ display: 'flex', gap: '16px', marginLeft: 'auto', fontSize: '13px', color: 'var(--text-muted)' }}>
          <div style={{ border: '1px solid var(--border-color)', borderRadius: '999px', padding: '4px 12px' }}>
            Dataset: {info.name}
          </div>
          <div style={{ border: '1px solid var(--border-color)', borderRadius: '999px', padding: '4px 12px' }}>
            {info.rows} rows
          </div>
          <div style={{ border: '1px solid var(--border-color)', borderRadius: '999px', padding: '4px 12px' }}>
            {info.columns} columns
          </div>
        </div>
        <button className="btn btn-primary" onClick={onNext} style={{ marginLeft: '16px' }}>
          Continue to Data Quality <ChevronRight size={16} />
        </button>
      </div>
    </div>
  );
};

export default DataProfileStep;
