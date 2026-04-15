import axios from 'axios';

const BASE = import.meta.env.VITE_API_URL || (import.meta.env.DEV ? '' : 'http://localhost:8765');
const api = axios.create({ baseURL: BASE });

export interface KaggleDataset {
  id: string;
  ref: string;
  title: string;
  subtitle: string;
  url: string;
  downloadCount: number;
  voteCount: number;
  size: string;
  totalBytes: number;
  usabilityRating: number;
  lastUpdated: string;
  tags: { name: string }[];
}

export interface BrowseDatasetsResponse {
  datasets: KaggleDataset[];
  page: number;
}

export interface KaggleFilters {
  search?: string;
  page?: number;
  sort_by?: string;
  file_type?: string;
}

export const browseKaggleDatasets = async (filters: KaggleFilters): Promise<BrowseDatasetsResponse> => {
  const { data } = await api.get('/api/kaggle/datasets/browse', { params: filters });
  return data;
};

export const getKaggleDownloadUrl = (ref: string): string => {
  return `${BASE}/api/kaggle/datasets/download/${ref}`;
};
