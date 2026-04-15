import { useState } from 'react';
import {
  ArrowLeft, Shield, Upload, Lock, Download, Database, Search,
} from 'lucide-react';
import '../ToolPage.css';

type AnonStep = 'input' | 'classification' | 'techniques' | 'export';
type DatasetTab = 'catalog' | 'upload';

const ANON_API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://localhost:8000';
const ANON_BACKEND_ENABLED = false;

const ANON_STEPS: { id: AnonStep; label: string; Icon: React.ElementType }[] = [
  { id: 'input',          label: 'Data Input',       Icon: Upload   },
  { id: 'classification', label: 'Classification',   Icon: Shield   },
  { id: 'techniques',     label: 'Techniques',       Icon: Lock     },
  { id: 'export',         label: 'Results & Export', Icon: Download },
];

const DATASETS = [
  {
    name: 'Details of Beneficiaries under Saakshar Bharat Programme (karkkum Bharatham Thittam) by Districts in Tamil Nadu from 2010-11 to 2017-18',
    desc: 'No description available',
    category: 'Adult Education',
  },
  {
    name: 'Enrolment in school education from 1950-51 to 2009-10',
    desc: 'No description available',
    category: 'Adult Education',
  },
  {
    name: 'Customer PII Dataset',
    desc: 'Customer names, emails, addresses and phone numbers',
    category: 'Commerce',
  },
  {
    name: 'Healthcare Patient Records',
    desc: 'Patient demographics and medical history data',
    category: 'Health',
  },
  {
    name: 'Financial Transaction Logs',
    desc: 'Bank account numbers and transaction details',
    category: 'Finance',
  },
];

const WORKFLOW_STEPS = [
  'Upload or select dataset for PII detection',
  'Classify sensitivity and assess risk',
  'Apply anonymization techniques',
  'Validate privacy-utility trade-off',
  'Generate compliance reports',
  'Export secure data',
];

const DataAnonymizationPage = ({ onBack }: { onBack: () => void }) => {
  const [activeStep, setActiveStep] = useState<AnonStep>('input');
  const [activeTab,  setActiveTab]  = useState<DatasetTab>('catalog');
  const [search,     setSearch]     = useState('');
  const [selected,   setSelected]   = useState<string | null>(null);

  const postAnonPlaceholder = async (path: string, payload: Record<string, unknown>) => {
    if (!ANON_BACKEND_ENABLED) return;
    try {
      await fetch(`${ANON_API_BASE}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
    } catch {
      // Placeholder intentionally ignores network failures until backend is connected.
    }
  };

  const handleStepChange = (step: AnonStep) => {
    setActiveStep(step);
    void postAnonPlaceholder('/anonymization/step', { step });
  };

  const handleTabChange = (tab: DatasetTab) => {
    setActiveTab(tab);
    void postAnonPlaceholder('/anonymization/tab', { tab });
  };

  const handleSearchChange = (value: string) => {
    setSearch(value);
    void postAnonPlaceholder('/anonymization/search', { query: value });
  };

  const handleDatasetSelect = (datasetName: string) => {
    setSelected(datasetName);
    void postAnonPlaceholder('/anonymization/dataset/select', { datasetName });
  };

  const activeIdx = ANON_STEPS.findIndex(s => s.id === activeStep);
  const filtered  = DATASETS.filter(d =>
    d.name.toLowerCase().includes(search.toLowerCase()) ||
    d.desc.toLowerCase().includes(search.toLowerCase())
  );

  // Build stepper as flat array – avoids Fragment-without-key issue
  const stepperNodes: React.ReactNode[] = [];
  ANON_STEPS.forEach((step, idx) => {
    const isActive = step.id === activeStep;
    const isDone   = activeIdx > idx;
    if (idx > 0) {
      stepperNodes.push(
        <div key={`conn-${idx}`} className={`xt-connector${isDone ? ' done' : ''}`} />
      );
    }
    stepperNodes.push(
      <div key={step.id} className="xt-step" onClick={() => handleStepChange(step.id)}>
        <div className={`xt-step-circle${isActive ? ' active' : isDone ? ' done' : ''}`}>
          <step.Icon size={20} />
        </div>
        <span className={`xt-step-label${isActive ? ' active' : ''}`}>{step.label}</span>
      </div>
    );
  });

  // Placeholder icon for non-input steps
  const PlaceholderIcon  = ANON_STEPS.find(s => s.id === activeStep)!.Icon;
  const placeholderLabel = ANON_STEPS.find(s => s.id === activeStep)!.label;

  return (
    <div className="xt-page">
      <button className="xt-back-btn" onClick={onBack}>
        <ArrowLeft size={14} /> Back to Data Tools
      </button>

      {/* Header */}
      <div className="xt-header">
        <div className="xt-header-icon">
          <Shield size={26} />
        </div>
        <div>
          <h1 className="xt-header-title">Data Anonymization</h1>
          <p className="xt-header-subtitle">
            Secure your data with advanced anonymization techniques
          </p>
        </div>
      </div>

      {/* Stepper */}
      <div className="xt-stepper-card">
        <div className="xt-stepper">{stepperNodes}</div>
      </div>

      {/* Two-column layout */}
      {activeStep === 'input' ? (
        <div className="xt-anon-layout">

          {/* Left — dataset selection */}
          <div className="xt-panel">
            <h2 className="xt-panel-title">Select Your Dataset</h2>
            <p className="xt-panel-subtitle">
              Choose from catalog or{' '}
              <span className="xt-link">upload your own file</span>{' '}
              for PII detection &amp; anonymization
            </p>

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
                <h3 className="xt-browse-title">Browse Catalog</h3>
                <p className="xt-browse-subtitle">
                  Select from government datasets for privacy analysis
                </p>
                <div className="xt-search-wrap">
                  <Search size={15} className="xt-search-icon" />
                  <input
                    className="xt-search-input"
                    placeholder="Search datasets by keyword..."
                    value={search}
                    onChange={e => handleSearchChange(e.target.value)}
                  />
                </div>
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
                <p style={{ fontSize: 12, color: '#cbd5e1' }}>
                  Supports CSV, Excel, JSON — up to 100 MB
                </p>
                <button className="xt-upload-btn">
                  <Upload size={14} /> Browse File
                </button>
              </div>
            )}
          </div>

          {/* Right — workflow sidebar */}
          <div className="xt-workflow-card">
            <p className="xt-workflow-title">Anonymization Workflow:</p>
            <ul className="xt-workflow-list">
              {WORKFLOW_STEPS.map(step => (
                <li key={step}>{step}</li>
              ))}
            </ul>
          </div>

        </div>
      ) : (
        <div className="xt-panel" style={{ textAlign: 'center', padding: '60px 40px' }}>
          <PlaceholderIcon size={48} color="#94a3b8" />
          <p style={{ marginTop: 16, color: '#64748b', fontSize: 15 }}>
            <strong>{placeholderLabel}</strong> — select a dataset first to continue.
          </p>
        </div>
      )}
    </div>
  );
};

export default DataAnonymizationPage;
