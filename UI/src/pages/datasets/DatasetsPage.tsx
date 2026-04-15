import { useState, useEffect, useCallback, useMemo } from 'react';
import { Search, Download, Database, ChevronRight, ChevronLeft, Loader2, X } from 'lucide-react';
import { browseKaggleDatasets, getKaggleDownloadUrl, KaggleDataset, KaggleFilters } from './datasetsHubApi';
import './DatasetsPage.css';

const FILE_TYPES = ['CSV', 'JSON', 'SQLite', 'Parquet', 'BigQuery'];
const USABILITY_RATINGS = [
  { label: '8.00 or higher', value: 0.8 },
  { label: '9.00 or higher', value: 0.9 },
  { label: '10.00 (Perfect)', value: 1.0 }
];

const SORT_OPTIONS = [
  { label: 'Hotness', value: 'hottest' },
  { label: 'Most Downloads', value: 'downloadCount' },
  { label: 'Most Votes', value: 'votes' },
  { label: 'Newest', value: 'published' },
  { label: 'Recently Updated', value: 'updated' }
];

export default function DatasetsPage() {
  const [rawDatasets, setRawDatasets] = useState<KaggleDataset[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);

  // Server-Side Filters
  const [search, setSearch] = useState('');
  const [tags, setTags] = useState<string[]>([]);
  const [tagInput, setTagInput] = useState('');
  const [sortBy, setSortBy] = useState('hottest');
  const [activeFileType, setActiveFileType] = useState<string>('all');

  // Client-Side Filters (to prevent 400 errors from Kaggle)
  const [minSizeMB, setMinSizeMB] = useState<number>(0);
  const [maxSizeMB, setMaxSizeMB] = useState<number>(100000); // default large
  const [minUsability, setMinUsability] = useState<number | null>(null);

  const fetchWithServerFilters = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const filters: KaggleFilters = {
        search: [search, ...tags].filter(Boolean).join(' '),
        page: page,
        sort_by: sortBy,
        file_type: activeFileType !== 'all' ? activeFileType.toLowerCase() : 'all'
      };
      
      const response = await browseKaggleDatasets(filters);
      setRawDatasets(response.datasets);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to connect to Kaggle. Please check your API token.');
    } finally {
      setLoading(false);
    }
  }, [search, tags, page, sortBy, activeFileType]);

  useEffect(() => {
    fetchWithServerFilters();
  }, [fetchWithServerFilters]);

  // Apply Frontend filters on the retrieved data
  const filteredDatasets = useMemo(() => {
    return rawDatasets.filter(ds => {
      // Size filter (Kaggle totalBytes is in bytes)
      const sizeMB = ds.totalBytes / (1024 * 1024);
      if (minSizeMB > 0 && sizeMB < minSizeMB) return false;
      if (maxSizeMB < 100000 && sizeMB > maxSizeMB) return false;
      
      // Usability filter (typically 0.0 to 1.0)
      if (minUsability !== null && (ds.usabilityRating || 0) < minUsability) return false;
      
      return true;
    });
  }, [rawDatasets, minSizeMB, maxSizeMB, minUsability]);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setPage(1);
    fetchWithServerFilters();
  };

  const resetFilters = () => {
    setTags([]);
    setTagInput('');
    setMinSizeMB(0);
    setMaxSizeMB(100000);
    setActiveFileType('all');
    setMinUsability(null);
    setPage(1);
  };

  const handleDownload = (ref: string) => {
    const url = getKaggleDownloadUrl(ref);
    window.open(url, '_blank');
  };

  const handleTagKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault();
      const newTag = tagInput.trim();
      if (newTag && !tags.includes(newTag)) {
        setTags([...tags, newTag]);
      }
      setTagInput('');
    }
  };

  const removeTag = (tagToRemove: string) => {
    setTags(tags.filter(t => t !== tagToRemove));
  };

  return (
    <div className="dp-page fade-in">
      <div className="dp-container">
        
        <div className="dp-header-section">
          <h1 className="dp-title">Kaggle Datasets Hub</h1>
          <p className="dp-subtitle">Explore millions of datasets. Securely streamed directly to your browser.</p>
          
          <form className="dp-search-bar" onSubmit={handleSearchSubmit}>
            <input 
              type="text" 
              placeholder="Search by keyword..." 
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="dp-search-input"
            />
            <button type="submit" className="dp-search-btn">Search</button>
          </form>
        </div>

        <div className="dp-content-grid">
          
          {/* Advanced Sidebar */}
          <aside className="dp-sidebar">
            <div className="dp-filter-panel">
              
              <div className="df-section">
                <label className="df-label">Tags Search</label>
                <div className="df-search-wrapper" style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', padding: '6px 12px', background: 'var(--bg-elevated)', border: '1px solid var(--border-mid)', borderRadius: '8px', minHeight: '40px', alignItems: 'center' }}>
                  <Search size={14} style={{ color: 'var(--t-secondary)', flexShrink: 0 }} />
                  {tags.map(tag => (
                    <span key={tag} style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', background: 'var(--bg-surface)', border: '1px solid var(--border-mid)', padding: '2px 8px', borderRadius: '999px', fontSize: '13px', color: 'var(--t-primary)' }}>
                      {tag}
                      <X size={12} style={{ cursor: 'pointer', color: 'var(--t-secondary)' }} onClick={() => removeTag(tag)} />
                    </span>
                  ))}
                  <input 
                    type="text" 
                    placeholder={tags.length === 0 ? "Enter tags..." : ""}
                    value={tagInput}
                    onChange={(e) => setTagInput(e.target.value)}
                    onKeyDown={handleTagKeyDown}
                    style={{ flex: 1, minWidth: '80px', background: 'transparent', border: 'none', outline: 'none', color: 'var(--t-primary)', fontSize: '14px', padding: '0' }}
                  />
                </div>
              </div>

              <div className="df-section">
                <label className="df-label">File Size Range (MB)</label>
                <div className="df-range">
                  <input 
                    type="number" 
                    min="0"
                    step="any"
                    placeholder="Min" 
                    value={minSizeMB === 0 ? '' : minSizeMB}
                    onWheel={(e) => e.currentTarget.blur()}
                    onChange={(e) => {
                      const val = Number(e.target.value);
                      if (val >= 0) setMinSizeMB(val);
                    }}
                    className="df-num-input"
                  />
                  <span style={{ color: 'var(--t-secondary)' }}>to</span>
                  <input 
                    type="number" 
                    min="0"
                    step="any"
                    placeholder="Max" 
                    value={maxSizeMB === 100000 ? '' : maxSizeMB}
                    onWheel={(e) => e.currentTarget.blur()}
                    onChange={(e) => {
                      const val = Number(e.target.value);
                      if (val >= 0) setMaxSizeMB(val);
                    }}
                    className="df-num-input"
                  />
                </div>
              </div>

              <div className="df-section">
                <label className="df-label">File Types</label>
                <div className="df-pills">
                  {FILE_TYPES.map(type => (
                    <button 
                      key={type}
                      className={`df-pill ${activeFileType === type ? 'active' : ''}`}
                      onClick={() => setActiveFileType(prev => prev === type ? 'all' : type)}
                    >
                      {type}
                    </button>
                  ))}
                </div>
              </div>

              <div className="df-section">
                <label className="df-label">Usability Rating</label>
                <div className="df-pills">
                  {USABILITY_RATINGS.map(rate => (
                    <button 
                      key={rate.value}
                      className={`df-pill ${minUsability === rate.value ? 'active' : ''}`}
                      onClick={() => setMinUsability(prev => prev === rate.value ? null : rate.value)}
                    >
                      {rate.label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="df-actions">
                <button className="df-btn df-btn--ghost" onClick={resetFilters}>Clear All</button>
                <button className="df-btn df-btn--primary" onClick={() => { setPage(1); fetchWithServerFilters(); }}>Refresh Results</button>
              </div>
            </div>
          </aside>

          {/* Main List Area */}
          <main className="dp-main">
            
            <div className="dp-main-header">
              <div className="dp-results-count">
                {loading ? 'Fetching...' : `Showing page ${page} (${filteredDatasets.length} matches)`}
              </div>
              
              <div className="dp-sort-group">
                <span className="df-label" style={{ marginRight: '10px' }}>Sort By:</span>
                <select 
                  className="dp-sort-select"
                  value={sortBy}
                  onChange={(e) => { setSortBy(e.target.value); setPage(1); }}
                >
                  {SORT_OPTIONS.map(opt => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
              </div>
            </div>

            {error && <div className="dp-error-banner">{error}</div>}

            {loading ? (
              <div className="dp-loading">
                <Loader2 className="dp-spinner" size={48} />
                <p>Accessing Kaggle Hub...</p>
              </div>
            ) : (
              <>
                <div className="dp-dataset-grid">
                  {filteredDatasets.map(ds => (
                    <div key={ds.id} className="dp-dataset-card">
                      <div className="dp-card-header">
                        <Database className="dp-card-icon" size={24} />
                        <div className="dp-card-title-group">
                          <h3 className="dp-card-title" title={ds.title}>{ds.title}</h3>
                          <span className="dp-card-author">by {ds.ref?.split('/')[0]}</span>
                        </div>
                      </div>
                      
                      <p className="dp-card-desc">{ds.subtitle || 'Kaggle dataset available for secure streaming and download.'}</p>
                      
                      <div className="dp-card-footer">
                        <div className="dp-card-meta">
                          <span>{ds.size}</span>
                          <span>▲ {ds.voteCount}</span>
                          <span style={{ color: (ds.usabilityRating || 0) > 0.8 ? 'var(--brand)' : 'var(--t-secondary)' }}>
                            ⭐ {Math.round((ds.usabilityRating || 0) * 10)}/10
                          </span>
                        </div>
                        <button className="dp-download-btn" onClick={() => handleDownload(ds.ref)}>
                          <Download size={16} /> Download
                        </button>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="dp-pagination">
                  <button className="dp-page-btn" disabled={page === 1} onClick={() => setPage(p => Math.max(1, p - 1))}>
                    <ChevronLeft size={18} /> Previous
                  </button>
                  <span className="dp-page-indicator">Page {page}</span>
                  <button className="dp-page-btn" disabled={rawDatasets.length === 0} onClick={() => setPage(p => p + 1)}>
                    Next <ChevronRight size={18} />
                  </button>
                </div>
              </>
            )}
          </main>
        </div>

      </div>
    </div>
  );
}
