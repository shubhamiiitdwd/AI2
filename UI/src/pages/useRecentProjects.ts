import { useState, useEffect } from 'react';

export interface RecentProject {
  id: string;
  name: string;
  feature: string;
  createdAt: number;
}

const STORAGE_KEY = 'ai-sphere-recent-projects';

export function useRecentProjects() {
  const [projects, setProjects] = useState<RecentProject[]>([]);

  const loadProjects = () => {
    try {
      const data = localStorage.getItem(STORAGE_KEY);
      if (data) {
        setProjects(JSON.parse(data));
      } else {
        setProjects([]);
      }
    } catch(e) {
      console.error('Failed to load recent projects', e);
    }
  };

  useEffect(() => {
    loadProjects();
    const handleStorage = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY) {
        loadProjects();
      }
    };
    window.addEventListener('storage', handleStorage);
    window.addEventListener('projects-updated', loadProjects);
    return () => {
      window.removeEventListener('storage', handleStorage);
      window.removeEventListener('projects-updated', loadProjects);
    };
  }, []);

  const addProject = (name: string, feature: string) => {
    try {
      const newProject: RecentProject = {
        id: Math.random().toString(36).substring(2, 9),
        name,
        feature,
        createdAt: Date.now()
      };
      const prev = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
      const next = [newProject, ...prev];
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      window.dispatchEvent(new Event('projects-updated'));
    } catch (e) {
      console.error('Failed to add project', e);
    }
  };

  const deleteProject = (id: string) => {
    try {
      const prev: RecentProject[] = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
      const next = prev.filter(p => p.id !== id);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      window.dispatchEvent(new Event('projects-updated'));
    } catch (e) {
      console.error('Failed to delete project', e);
    }
  };

  return { projects, addProject, deleteProject };
}

export const getFeatureLabel = (feature: string) => {
  switch (feature) {
    case 'ocr': return 'OCR Conversion';
    case 'viz': return 'Data Visualization';
    case 'feat': return 'Feature Engineering';
    case 'anon': return 'Data Anonymization';
    case 'automl': return 'AutoML Pipeline';
    default: return feature;
  }
};
