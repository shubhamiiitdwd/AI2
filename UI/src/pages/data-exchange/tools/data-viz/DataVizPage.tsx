import { useState } from 'react';
import { ArrowLeft, BarChart2, Database, Upload, Search } from 'lucide-react';
import '../ToolPage.css';

type DatasetTab = 'catalog' | 'upload';

const DATAVIZ_API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://localhost:8000';
const DATAVIZ_BACKEND_ENABLED = false;

const DATASETS = [
  {
    name: 'Details of Beneficiaries under Saakshar Bharat Programme (karkkum Bharatham Thittam) by Districts in Tamil Nadu from 2010-11 to 2017-18',
    desc: 'Details of Beneficiaries under Saakshar Bharat Programme (karkkum Bharatham Thittam) by Districts in Tamil Nadu from 2010-11 to 2017-18',
    category: 'Adult Education',
  },
  {
    name: 'Enrolment in school education from 1950-51 to 2009-10',
    desc: 'No description available',
    category: 'Adult Education',
  },
  {
    name: 'District-wise Number of Recognised Schools in India',
    desc: 'Statistical data on school infrastructure across all districts in India',
    category: 'Education',
  },
  {
    name: 'Customer Purchase Behavior Analytics',
    desc: 'Shopping patterns and purchase history across categories',
    category: 'Commerce',
  },
  {
    name: 'Sales Performance Data Q4 2025',
    desc: 'Regional sales data with trend indicators and forecasts',
    category: 'Business',
  },
  {
    name: 'Product Inventory Records',
    desc: 'Real-time inventory tracking data with stock levels',
    category: 'Operations',
  },
];

const DataVizPage = ({ onBack }: { onBack: () => void }) => {
  const [activeTab, setActiveTab] = useState<DatasetTab>('catalog');
  const [search,    setSearch]    = useState('');
  const [selected,  setSelected]  = useState<string | null>(null);

  const postDataVizPlaceholder = async (path: string, payload: Record<string, unknown>) => {
    if (!DATAVIZ_BACKEND_ENABLED) return;
    try {
      await fetch(`${DATAVIZ_API_BASE}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
    } catch {
      // Placeholder intentionally ignores network failures until backend is connected.
    }
  };

  const handleTabChange = (tab: DatasetTab) => {
    setActiveTab(tab);
    void postDataVizPlaceholder('/dataviz/tab', { tab });
  };

  const handleSearchChange = (value: string) => {
    setSearch(value);
    void postDataVizPlaceholder('/dataviz/search', { query: value });
  };

  const handleDatasetSelect = (datasetName: string) => {
    setSelected(datasetName);
    void postDataVizPlaceholder('/dataviz/dataset/select', { datasetName });
  };

  const filtered = DATASETS.filter(d =>
    d.name.toLowerCase().includes(search.toLowerCase()) ||
    d.desc.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="xt-page">
      <button className="xt-back-btn" onClick={onBack}>
        <ArrowLeft size={14} /> Back to Data Tools
      </button>

      {/* Header */}
      <div className="xt-header">
        <div className="xt-header-icon">
          <BarChart2 size={26} />
        </div>
        <div>
          <h1 className="xt-header-title">Data Visualization</h1>
          <p className="xt-header-subtitle">Select Dataset</p>
        </div>
      </div>

      {/* Dataset selection panel */}
      <div className="xt-panel">
        <h2 className="xt-panel-title">Select Your Dataset</h2>
        <p className="xt-panel-subtitle">
          Choose from catalog or{' '}
          <span className="xt-link">upload your own file</span>
        </p>

        {/* Tabs */}
        <div className="xt-tabs">
          <button
            className={`xt-tab${activeTab === 'catalog' ? ' active' : ''}`}
            onClick={() => handleTabChange('catalog')}
          >
            <Database size={15} /> Catalog
          </button>
          <button
            className={`xt-tab${activeTab === 'upload' ? ' active' : ''}`}
            onClick={() => handleTabChange('upload')}
          >
            <Upload size={15} /> Upload Dataset
          </button>
        </div>

        {activeTab === 'catalog' && (
          <>
            {/* Search */}
            <div className="xt-search-wrap">
              <Search size={15} className="xt-search-icon" />
              <input
                className="xt-search-input"
                placeholder="Search datasets by name, description, or tags..."
                value={search}
                onChange={e => handleSearchChange(e.target.value)}
              />
            </div>

            <p className="xt-count">Found {filtered.length > 3 ? '1913' : filtered.length} datasets</p>

            <div className="xt-dataset-list">
              {filtered.map(d => (
                <div
                  key={d.name}
                  className={`xt-dataset-row${selected === d.name ? ' selected' : ''}`}
                  onClick={() => handleDatasetSelect(d.name)}
                >
                  <div className="xt-dataset-info">
                    <span className="xt-dataset-name">{d.name}</span>
                    <span className="xt-dataset-desc">{d.desc}</span>
                  </div>
                  <span className="xt-dataset-badge">{d.category}</span>
                </div>
              ))}
            </div>
          </>
        )}

        {activeTab === 'upload' && (
          <div className="xt-upload-zone">
            <Upload size={36} color="#fb8c36" />
            <p>Drag &amp; drop a file here, or click to browse</p>
            <p style={{ fontSize: 12, color: '#cbd5e1' }}>Supports CSV, Excel, JSON — up to 100 MB</p>
            <button className="xt-upload-btn">
              <Upload size={14} /> Browse File
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default DataVizPage;
