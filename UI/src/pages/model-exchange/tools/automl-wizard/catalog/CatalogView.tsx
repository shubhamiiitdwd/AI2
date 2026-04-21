import { useState, useEffect, useRef, useCallback } from 'react';
import { Search } from 'lucide-react';
import { fetchDatasets, type DatasetEntry } from './services';
import type { DatasetMetadata } from '../types';
import DatasetDetailsHub from './DatasetDetailsHub';

interface Props {
  onSelect?: (id: string) => void;
  /** Called when a catalog dataset is imported into the AutoML workspace */
  onDatasetImported?: (meta: DatasetMetadata) => void;
}

const PAGE_SIZE = 50;

const CatalogView = ({ onSelect, onDatasetImported }: Props) => {
  const [search, setSearch] = useState('');
  const [selected, setSelected] = useState<DatasetEntry | null>(null);
  const [datasets, setDatasets] = useState<DatasetEntry[]>([]);
  const [datasetsTotal, setDatasetsTotal] = useState(0);
  const [datasetsLoading, setDatasetsLoading] = useState(false);
  const [datasetsOffset, setDatasetsOffset] = useState(0);
  const [activeCategory, setActiveCategory] = useState<string | null>(null);
  const [hasMore, setHasMore] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const observer = useRef<IntersectionObserver | null>(null);
  const lastElementRef = useCallback(
    (node: HTMLDivElement | null) => {
      if (datasetsLoading) return;
      if (observer.current) observer.current.disconnect();
      observer.current = new IntersectionObserver((entries) => {
        if (entries[0]?.isIntersecting && hasMore) {
          setDatasetsOffset((prev) => prev + PAGE_SIZE);
        }
      });
      if (node) observer.current.observe(node);
    },
    [datasetsLoading, hasMore],
  );

  const categories = Array.from(new Set(datasets.map((d) => d.category))).slice(0, 10);

  const loadDatasets = async (query: string, category: string, offset: number) => {
    setDatasetsLoading(true);
    setError(null);
    try {
      const page = await fetchDatasets(query, offset, PAGE_SIZE, category);
      if (page.error) {
        setError(page.error);
        return;
      }
      setDatasetsTotal(page.total);
      setDatasets((prev) => (offset === 0 ? page.results : [...prev, ...page.results]));
      setHasMore(offset + PAGE_SIZE < page.total);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to connect to catalog';
      setError(msg);
    } finally {
      setDatasetsLoading(false);
    }
  };

  useEffect(() => {
    if (search.trim()) setActiveCategory(null);
    setDatasetsOffset(0);
    const t = setTimeout(() => loadDatasets(search, activeCategory || '', 0), search ? 400 : 0);
    return () => clearTimeout(t);
  }, [search, activeCategory]);

  useEffect(() => {
    if (datasetsOffset > 0) loadDatasets(search, activeCategory || '', datasetsOffset);
  }, [datasetsOffset]);

  return (
    <div className="aw-catalog-root">
      <div className="aw-catalog-main">
        <h2 className="aw-catalog-title">Government data catalog</h2>
        <p className="aw-catalog-sub">
          Browse Central Government datasets from{' '}
          <a href="https://data.gov.in" target="_blank" rel="noreferrer">
            data.gov.in
          </a>
          . Import one to continue in AutoML.
        </p>

        <div className="aw-catalog-search-wrap">
          <Search size={15} className="aw-catalog-search-icon" />
          <input
            className="aw-catalog-search-input"
            placeholder="Search datasets…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          {search && (
            <button type="button" className="aw-catalog-search-clear" onClick={() => setSearch('')}>
              ✕
            </button>
          )}
        </div>

        <div className="aw-catalog-chips">
          {['All Sectors', ...categories].map((cat) => (
            <button
              type="button"
              key={cat}
              className={`aw-catalog-chip ${
                (cat === 'All Sectors' ? !activeCategory : activeCategory === cat) ? 'aw-catalog-chip--on' : ''
              }`}
              onClick={() => {
                if (cat === 'All Sectors') {
                  setActiveCategory(null);
                } else {
                  setSearch('');
                  setActiveCategory(activeCategory === cat ? null : cat);
                }
              }}
            >
              {cat}
            </button>
          ))}
        </div>

        <p className="aw-catalog-meta">
          {datasetsLoading && datasets.length === 0
            ? 'Connecting to catalog…'
            : error
              ? `Error: ${error}`
              : `${datasetsTotal.toLocaleString()} datasets`}
        </p>

        <div className="aw-catalog-list">
          {datasets.length === 0 && datasetsLoading
            ? Array.from({ length: 6 }).map((_, i) => (
                <div key={i} className="aw-catalog-skeleton">
                  <div className="aw-catalog-skeleton-line aw-catalog-skeleton-line--lg" />
                  <div className="aw-catalog-skeleton-line" />
                </div>
              ))
            : datasets.map((d, i) => (
                <div
                  key={`${d.id || d.name}-${i}`}
                  ref={i === datasets.length - 1 ? lastElementRef : null}
                  className={`aw-catalog-row ${selected?.name === d.name ? 'aw-catalog-row--selected' : ''}`}
                  onClick={() => {
                    setSelected(d);
                    onSelect?.(d.id || d.name);
                  }}
                >
                  <div>
                    <div className="aw-catalog-row-title">{d.name}</div>
                    <div className="aw-catalog-row-desc">{d.desc}</div>
                  </div>
                  <span className="aw-catalog-row-badge">{d.category}</span>
                </div>
              ))}
          {datasetsLoading && datasets.length > 0 && (
            <div className="aw-catalog-loading-more">Loading more…</div>
          )}
        </div>
        {!hasMore && datasets.length > 0 && <p className="aw-catalog-end">— End of results —</p>}
      </div>

      {selected && (
        <DatasetDetailsHub
          dataset={selected}
          onClose={() => setSelected(null)}
          onImportComplete={onDatasetImported}
        />
      )}
    </div>
  );
};

export default CatalogView;
