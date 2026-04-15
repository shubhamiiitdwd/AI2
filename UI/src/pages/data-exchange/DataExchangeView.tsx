import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  FileText, Sparkles, ScanLine, BarChart2, Lock, ArrowRight,
  TrendingUp, Activity, Star, Users, ChevronRight, Download, ChevronDown, Trash2
} from 'lucide-react';
import { useRecentProjects, getFeatureLabel, RecentProject } from '../useRecentProjects';
import './DataExchangeView.css';
import watermarkLogo from '../../assets/looplogo.png';
import OcrPage               from './tools/ocr/OcrPage';
import DataVizPage           from './tools/data-viz/DataVizPage';
import FeatureEngineeringPage from './tools/feature-engineering/FeatureEngineeringPage';
import DataAnonymizationPage  from './tools/data-anonymization/DataAnonymizationPage';


// ─── Types ───────────────────────────────────────────────────────────────────
type ToolId = 'ocr' | 'viz' | 'feat' | 'anon' | null;

interface GoldenDataset {
  id: number;
  title: string;
  description: string;
  tags: string[];
  filename: string;
  size: string;
}

// ─── Dashboard Static Data ──────────────────────────────────────────────────
const DUMMY_GOLDEN_DATASETS: GoldenDataset[] = [
  { id: 1, title: 'Crop Recommendation Dataset', description: 'Comprehensive soil and weather metrics for advanced agricultural analytics.', tags: ['Tabular', 'Agriculture'], filename: 'crop_recommendation.csv', size: '2 MB' },
  { id: 2, title: 'Stroke Prediction Dataset', description: 'Patient data containing health parameters for stroke probability prediction.', tags: ['Healthcare'], filename: 'stroke_prediction.csv', size: '500 KB' },
  { id: 3, title: 'Supply Chain Dataset', description: 'Historical operations data for product flow and delivery trend analysis.', tags: ['Business', 'Tabular'], filename: 'supply_chain.json', size: '15 MB' },
  { id: 4, title: 'Financial Sentiment Analysis', description: 'Text dataset of financial news headlines labeled with market sentiment.', tags: ['NLP', 'Finance'], filename: 'financial_sentiment.csv', size: '8 MB' },
  { id: 5, title: 'Retail Core Sales', description: 'Weekly sales data across various retail store departments.', tags: ['Business', 'Popular'], filename: 'retail_sales.csv', size: '12 MB' }
];

const ACTIVITY_DATA =[
  { id: 1, type: 'Data Visualization',  desc: 'Generated sales dashboard with 5 charts',           user: 'John Doe',     time: '2025-12-17 10:30 AM', status: 'completed' },
  { id: 2, type: 'OCR Conversion',      desc: 'Processed 15 PDF documents',                        user: 'Sarah Smith',  time: '2025-12-17 09:45 AM', status: 'completed' },
  { id: 3, type: 'Data Anonymization',  desc: 'Anonymized customer dataset (1,500 records)',        user: 'Mike Johnson', time: '2025-12-17 09:15 AM', status: 'completed' },
  { id: 4, type: 'Feature Engineering', desc: 'Applied 8 transformations on product data',          user: 'Emily Chen',   time: '2025-12-17 08:30 AM', status: 'completed' },
  { id: 5, type: 'OCR Conversion',      desc: 'Extracted text from 23 images',                      user: 'David Lee',    time: '2025-12-16 05:20 PM', status: 'completed' },
];

const STATS_DATA =[
  { value: '1,247', label: 'Total Operations',     change: '+12.5%', positive: true  },
  { value: '8,432', label: 'Documents Processed',  change: '+8.2%',  positive: true  },
  { value: '156',   label: 'Active Users',          change: '+5.1%',  positive: true  },
  { value: '2.4s',  label: 'Avg. Processing Time', change: '-13.3%', positive: false },
];

// ─── Tool card data ──────────────────────────────────────────────────────────
const dataTools =[
  { id: 'ocr',  icon: <ScanLine size={32} />,  title: 'OCR Conversion',      iconBoxType: 'orange', description: 'Transform images and scanned documents into editable text. Extract data from PDFs, images, and handwritten documents with advanced optical character recognition.' },
  { id: 'viz',  icon: <BarChart2 size={32} />, title: 'Data Visualization',  iconBoxType: 'dark',   description: 'Create insightful charts, graphs, and interactive dashboards. Explore your data visually with powerful visualization tools and statistical analysis.' },
  { id: 'feat', icon: <Sparkles size={32} />,  title: 'Feature Engineering', iconBoxType: 'dark',   description: 'Transform and optimize your dataset features. Apply advanced techniques to create meaningful features that improve model performance and accuracy.' },
  { id: 'anon', icon: <Lock size={32} />,      title: 'Data Anonymization',  iconBoxType: 'dark',   description: 'Protect sensitive information with robust anonymization techniques. Ensure data privacy and compliance while maintaining data utility for analysis.' },
];

interface DataToolCardProps {
  tool: (typeof dataTools)[number];
  onOpen: (id: ToolId) => void;
}

const DataToolCard = ({ tool, onOpen }: DataToolCardProps) => {
  const [hovered, setHovered] = useState(false);

  return (
    <article
      className={`tool-card tool-card--expand${hovered ? ' is-open' : ''}`}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <div className="tool-card-content">
        <div className={`tool-icon-box ${tool.iconBoxType}`}>{tool.icon}</div>
        <h3 className="tool-title">{tool.title}</h3>
        <p className="tool-description">{tool.description}</p>
      </div>
      <button
        className={`tool-open-btn${hovered ? ' tool-open-btn--open' : ''}`}
        onClick={() => onOpen(tool.id as ToolId)}
        aria-label={`Open ${tool.title}`}
      >
        <span className="tool-open-btn-icon"><ArrowRight size={16} /></span>
        <span className="tool-open-btn-text">Open Tool</span>
      </button>
    </article>
  );
};

// ─── DataExchangeView ────────────────────────────────────────────────────────
const DataExchangeView = () => {
  const navigate = useNavigate();
  const { projects, addProject, deleteProject } = useRecentProjects();
  const [activeTool, setActiveTool] = useState<ToolId>(null);
  const[showStartModal, setShowStartModal] = useState(false);
  const[projectName, setProjectName] = useState('');
  const [startFeature, setStartFeature] = useState<ToolId>(null);
  const[isFeatureOpen, setIsFeatureOpen] = useState(false);

  // Static Golden Datasets State
  const[goldenDatasets, setGoldenDatasets] = useState<GoldenDataset[]>(DUMMY_GOLDEN_DATASETS);

  useEffect(() => {
    window.scrollTo(0, 0);
  },[activeTool]);

  useEffect(() => {
    const pendingTool = sessionStorage.getItem('home-start-tool');
    if (pendingTool === 'ocr' || pendingTool === 'viz' || pendingTool === 'feat' || pendingTool === 'anon') {
      setActiveTool(pendingTool);
    }
    if (pendingTool) sessionStorage.removeItem('home-start-tool');
  },[]);

  const startOptions: { value: Exclude<ToolId, null>; label: string }[] =[
    { value: 'ocr', label: 'OCR Conversion' },
    { value: 'viz', label: 'Data Visualization' },
    { value: 'feat', label: 'Feature Engineering' },
    { value: 'anon', label: 'Data Anonymization' },
  ];

  const handleStartProject = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!projectName.trim() || !startFeature) return;
    addProject(projectName, startFeature as string);
    setShowStartModal(false);
    setIsFeatureOpen(false);
    setActiveTool(startFeature);
  };

  const handleDownloadDataset = (filename: string) => {
    // Do nothing for dummy datasets
  };

  const getTagStyleClass = (tag: string) => {
    const lower = tag.toLowerCase();
    if (lower.includes('diabetes') || lower.includes('healthcare')) return 'trending';
    if (lower.includes('tabular') || lower.includes('business')) return 'popular';
    return 'downloaded'; 
  };

  if (activeTool === 'ocr')  return <OcrPage               onBack={() => setActiveTool(null)} />;
  if (activeTool === 'viz')  return <DataVizPage            onBack={() => setActiveTool(null)} />;
  if (activeTool === 'feat') return <FeatureEngineeringPage onBack={() => setActiveTool(null)} />;
  if (activeTool === 'anon') return <DataAnonymizationPage  onBack={() => setActiveTool(null)} />;

  return (
    <div className="view-animate-in data-exchange-container">
      <div className="data-hero">
        <div className="data-watermark" aria-hidden="true">
          <div className="data-watermark-halo" />
          <img src={watermarkLogo} alt="" className="data-watermark-img" />
        </div>

        <h1 className="hero-title">Data Exchange</h1>
        <div className="hero-badge">
          <FileText size={14} className="hero-badge-icon" />
          <span>Data Processing Tools</span>
        </div>
        <p className="hero-subtitle">
          Powerful tools for data processing, transformation, and analysis. Transform your raw<br />
          data into actionable insights.
        </p>
        <button
          className="data-start-btn"
          onClick={() => {
            setShowStartModal(true);
            setIsFeatureOpen(false);
          }}
        >
          <Sparkles size={16} />
          <span>Start New Project</span>
          <ArrowRight size={16} />
        </button>
      </div>

      {showStartModal && (
        <div
          className="data-modal-backdrop"
          onClick={() => {
            setShowStartModal(false);
            setIsFeatureOpen(false);
          }}
        >
          <div className="data-modal" onClick={(e) => e.stopPropagation()}>
            <button
              className="data-modal-close"
              type="button"
              onClick={() => {
                setShowStartModal(false);
                setIsFeatureOpen(false);
              }}
              aria-label="Close"
            >
              x
            </button>

            <h3 className="data-modal-title">Start New Project</h3>
            <p className="data-modal-sub">Create a new data processing project and select features to include.</p>

            <form className="data-modal-form" onSubmit={handleStartProject}>
              <label className="data-modal-label" htmlFor="data-project-name">Project Name</label>
              <input
                id="data-project-name"
                className="data-modal-input"
                placeholder="Enter project name..."
                value={projectName}
                onChange={(e) => setProjectName(e.target.value)}
              />

              <label className="data-modal-label" htmlFor="data-start-feature">Where to Start</label>
              <div className="data-select" id="data-start-feature">
                <button
                  type="button"
                  className={`data-select-trigger${isFeatureOpen ? ' is-open' : ''}`}
                  onClick={() => setIsFeatureOpen((v) => !v)}
                >
                  <span className={!startFeature ? 'is-placeholder' : ''}>
                    {startOptions.find((o) => o.value === startFeature)?.label ?? 'Select starting feature...'}
                  </span>
                  <ChevronDown size={16} />
                </button>

                {isFeatureOpen && (
                  <div className="data-select-menu">
                    {startOptions.map((opt) => (
                      <button
                        key={opt.value}
                        type="button"
                        className={`data-select-option${startFeature === opt.value ? ' is-active' : ''}`}
                        onClick={() => {
                          setStartFeature(opt.value);
                          setIsFeatureOpen(false);
                        }}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>
                )}
              </div>

              <div className="data-modal-actions">
                <button
                  type="button"
                  className="data-modal-btn data-modal-btn--ghost"
                  onClick={() => {
                    setShowStartModal(false);
                    setIsFeatureOpen(false);
                  }}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="data-modal-btn data-modal-btn--primary"
                  disabled={!projectName.trim() || !startFeature}
                >
                  Start Project
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      <div className="data-stats-row">
        {STATS_DATA.map((s, i) => (
          <div key={i} className="data-stat-card">
            <div className="data-stat-top">
              <div className="data-stat-icon"><TrendingUp size={16} /></div>
              <span className={`data-stat-badge ${s.positive ? 'positive' : 'negative'}`}>{s.change}</span>
            </div>
            <div className="data-stat-value">{s.value}</div>
            <div className="data-stat-label">{s.label}</div>
          </div>
        ))}
      </div>

      <div className="data-tools-grid">
        {dataTools.map(tool => (
          <DataToolCard key={tool.id} tool={tool} onOpen={setActiveTool} />
        ))}
      </div>

      {projects.length > 0 && (
        <div className="data-recent-projects" style={{ marginTop: '40px', marginBottom: '40px', width: '100%', maxWidth: '1400px', alignSelf: 'center' }}>
          <h2 style={{ fontSize: '1.25rem', color: 'var(--t-primary, #eef1ff)', marginBottom: '16px' }}>Recent Projects</h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '16px' }}>
            {projects.map((p: RecentProject) => (
              <div key={p.id} style={{
                background: 'var(--bg-surface, #0b1120)', border: '1px solid var(--border-mid, rgba(255,255,255,0.09))', borderRadius: '12px', padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <h3 style={{ fontSize: '1.1rem', margin: 0, color: 'var(--t-primary, #eef1ff)' }}>{p.name}</h3>
                  <button onClick={() => deleteProject(p.id)} style={{ background: 'transparent', border: 'none', color: 'var(--t-muted, #7e8ca8)', cursor: 'pointer' }} title="Delete">
                    <Trash2 size={16} />
                  </button>
                </div>
                <div style={{ color: 'var(--t-secondary, #7e8ca8)', fontSize: '0.9rem' }}>{getFeatureLabel(p.feature)}</div>
                <button 
                  onClick={() => {
                    if (p.feature === 'automl') {
                      navigate('/aisphere/model');
                    } else {
                      setActiveTool(p.feature as ToolId);
                    }
                  }} 
                  style={{ alignSelf: 'flex-start', display: 'flex', alignItems: 'center', gap: '6px', background: 'rgba(251,140,54,0.16)', color: 'var(--brand, #fb8c36)', border: '1px solid rgba(251,140,54,0.65)', padding: '6px 12px', borderRadius: '999px', cursor: 'pointer', fontSize: '0.85rem', fontWeight: 700 }}
                >
                  <span>Open Tool</span>
                  <ArrowRight size={14} />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Dashboard ── */}
      <div className="data-dashboard">

        {/* Left – Recent Activity */}
        <div className="data-activity-card">
          <div className="data-activity-header">
            <div className="data-activity-icon"><Activity size={18} /></div>
            <div>
              <h3>Recent Activity</h3>
              <p>Latest operations and data processing activities</p>
            </div>
          </div>
          <div className="data-activity-list">
            {ACTIVITY_DATA.map(item => (
              <div key={item.id} className="data-activity-item">
                <div className="data-activity-dot" />
                <div className="data-activity-info">
                  <span className="data-activity-type">{item.type}</span>
                  <span className="data-activity-desc">{item.desc}</span>
                  <span className="data-activity-meta">
                    <Users size={11} />{item.user}&nbsp;&nbsp;<span>{item.time}</span>
                  </span>
                </div>
                <span className="data-activity-status">{item.status}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Right col – Golden Datasets */}
        <div className="data-right-col">
          <div className="data-golden-card">
            <div className="data-golden-header">
              <div className="data-golden-icon"><Star size={18} /></div>
              <div>
                <h3>Golden Datasets</h3>
                <p>High-quality, curated datasets ready for immediate use</p>
              </div>
            </div>
            <div className="data-golden-list">
              {goldenDatasets.map(ds => (
                <div key={ds.id} className="data-golden-item">
                  <div className="data-golden-info">
                    
                    {/* Name Row (Removed the right-side category tag) */}
                    <div className="data-golden-title-row">
                      <span className="data-golden-name">{ds.title}</span>
                    </div>
                    
                    <p className="data-golden-desc">{ds.description}</p>
                    
                    {/* Bottom Tags */}
                    <div className="data-golden-tags">
                      {ds.tags.map((tag, idx) => (
                        <span key={idx} className={`data-golden-tag ${getTagStyleClass(tag)}`}>
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                  {/* DOWNLOAD BUTTON */}
                  <button 
                    className="data-golden-arrow" 
                    onClick={() => handleDownloadDataset(ds.filename)}
                    title={`Download ${ds.filename}`}
                    style={{ cursor: 'pointer' }}
                  >
                    <Download size={16} /> 
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

    </div>
  );
};

export default DataExchangeView;