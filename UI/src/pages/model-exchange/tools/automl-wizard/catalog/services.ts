// data.gov.in catalog — Team 1 API under /team1/data-gov/catalog

import type { DatasetMetadata } from '../types';

const DEFAULT_BACKEND_PORT = '8099';

const BASE =
  import.meta.env.VITE_API_URL ||
  (import.meta.env.DEV
    ? ''
    : `http://localhost:${import.meta.env.VITE_BACKEND_PORT || DEFAULT_BACKEND_PORT}`);

const CATALOG_BASE = `${BASE}/team1/data-gov/catalog`;

export interface DatasetEntry {
  name: string;
  desc: string;
  category: string;
  org: string;
  id?: string;
  fields?: Array<{ name: string; type?: string }>;
  updated?: string;
}

export interface DatasetPage {
  total: number;
  results: DatasetEntry[];
  error?: string;
}

export interface FetchProgressEvent {
  rows_fetched: number;
  total: number | null;
  pct: number;
}

export interface FetchForAnalysisResult {
  filename: string;
  rows_fetched: number;
  total_available: number | null;
  truncated: boolean;
}

export async function fetchDatasets(
  q: string,
  offset: number,
  limit: number,
  sector = '',
): Promise<DatasetPage> {
  const params = new URLSearchParams({ q, offset: String(offset), limit: String(limit), sector });
  const res = await fetch(`${CATALOG_BASE}/list?${params}`);
  if (!res.ok) throw new Error(`Catalog fetch failed: ${res.status}`);
  return res.json();
}

export async function fetchDatasetSample(
  indexId: string,
): Promise<{ records: unknown[]; field?: unknown[]; total?: number; error?: string }> {
  const res = await fetch(`${CATALOG_BASE}/sample/${encodeURIComponent(indexId)}`);
  if (!res.ok) throw new Error(`Sample fetch failed: ${res.status}`);
  return res.json();
}

export async function fetchDatasetForAnalysis(
  indexId: string,
  datasetName: string,
  onProgress?: (e: FetchProgressEvent) => void,
  maxRows = 100_000,
): Promise<FetchForAnalysisResult> {
  const res = await fetch(`${CATALOG_BASE}/fetch-for-analysis`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ index_id: indexId, dataset_name: datasetName, max_rows: maxRows }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' })) as { detail?: string };
    throw new Error(err.detail || `Fetch failed: ${res.status}`);
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error('No response stream');

  const decoder = new TextDecoder();
  let buffer = '';
  let result: FetchForAnalysisResult | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';

    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const event = JSON.parse(line) as {
          type: string;
          rows_fetched?: number;
          total?: number | null;
          pct?: number;
          filename?: string;
          total_available?: number | null;
          truncated?: boolean;
          message?: string;
        };
        if (event.type === 'progress') {
          onProgress?.({
            rows_fetched: event.rows_fetched ?? 0,
            total: event.total ?? null,
            pct: event.pct ?? 0,
          });
        } else if (event.type === 'complete') {
          result = {
            filename: event.filename ?? '',
            rows_fetched: event.rows_fetched ?? 0,
            total_available: event.total_available ?? null,
            truncated: Boolean(event.truncated),
          };
        } else if (event.type === 'error') {
          throw new Error(event.message || 'Unknown error');
        }
      } catch (e) {
        if (e instanceof SyntaxError) continue;
        throw e;
      }
    }
  }

  if (!result) throw new Error('Dataset fetch completed without a result event.');
  return result;
}

export function downloadDatasetCSV(indexId: string, datasetName: string): void {
  const safeName = encodeURIComponent(datasetName.slice(0, 60));
  const url = `${CATALOG_BASE}/download/${encodeURIComponent(indexId)}?filename=${safeName}`;
  const a = document.createElement('a');
  a.href = url;
  a.download = `${datasetName.slice(0, 60)}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

export async function chatAboutDataset(
  name: string,
  desc: string,
  fields: Array<{ name: string; type?: string }>,
  question: string,
  history: Array<{ role: string; content: string }> = [],
): Promise<{ answer: string; model?: string }> {
  const res = await fetch(`${CATALOG_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset_name: name, description: desc, fields, question, history }),
  });
  if (!res.ok) throw new Error(`Chat failed: ${res.status}`);
  return res.json();
}

/** Stream download from data.gov.in and register as a Team 1 dataset (NDJSON). */
export async function importDataGovForAutoml(
  indexId: string,
  datasetName: string,
  description: string,
  onProgress?: (e: FetchProgressEvent) => void,
  maxRows = 100_000,
): Promise<DatasetMetadata> {
  const res = await fetch(`${CATALOG_BASE}/import`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      index_id: indexId,
      dataset_name: datasetName,
      description,
      max_rows: maxRows,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' })) as { detail?: string };
    throw new Error(err.detail || `Import failed: ${res.status}`);
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error('No response stream');

  const decoder = new TextDecoder();
  let buffer = '';
  let dataset: DatasetMetadata | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';

    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const event = JSON.parse(line) as {
          type: string;
          rows_fetched?: number;
          total?: number | null;
          pct?: number;
          dataset?: DatasetMetadata;
          message?: string;
        };
        if (event.type === 'progress') {
          onProgress?.({
            rows_fetched: event.rows_fetched ?? 0,
            total: event.total ?? null,
            pct: event.pct ?? 0,
          });
        } else if (event.type === 'complete' && event.dataset) {
          dataset = event.dataset;
        } else if (event.type === 'error') {
          throw new Error(event.message || 'Import error');
        }
      } catch (e) {
        if (e instanceof SyntaxError) continue;
        throw e;
      }
    }
  }

  if (!dataset) throw new Error('Import finished without dataset metadata.');
  return dataset;
}
