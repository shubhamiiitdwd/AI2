/**
 * Model Exchange in-progress session — persisted so the dashboard can show "Session management"
 * and the user can resume the AutoML / clustering wizard where they left off.
 */
import type { MLTask, WizardStep } from './tools/automl-wizard/types';

const STORAGE_KEY = 'aikosh-model-exchange-wizard-v1';

export const MODEL_EXCHANGE_SESSION_KEY = STORAGE_KEY;

/** Must match the last `localStorage` string so `readModelExchangeSession` returns a stable object reference (required for `useSyncExternalStore`). */
let snapshotRaw: string | null | undefined;
let snapshotData: PersistedModelExchangeSession | null | undefined;

export type PersistedModelExchangeSession = {
  v: 1;
  at: string;
  step: WizardStep;
  completedSteps: number[];
  mlTask: MLTask;
  dataset: { id: string; filename: string } | null;
  targetColumn: string;
  featureColumns: string[];
  runId: string | null;
  clusteringRunId: string | null;
  trainingFinished: boolean;
  /** Short label for the dashboard (optional; recomputed if missing) */
  activity?: string;
};

function defaultActivity(s: Pick<PersistedModelExchangeSession, 'step' | 'mlTask'> & { isClustering: boolean }): string {
  const { step, isClustering } = s;
  if (isClustering) {
    if (step === 0) return 'Select dataset (clustering)';
    if (step === 1) return 'Configure clustering';
    if (step === 2) return 'Clustering run in progress';
    if (step === 3) return 'Review clustering results';
    return 'Clustering workflow';
  }
  if (step === 0) return 'Select dataset (AutoML)';
  if (step === 1) return 'Configure columns & task';
  if (step === 2) return 'Choose models & training settings';
  if (step === 3) return 'Training in progress';
  if (step === 4) return 'View results';
  return 'AutoML workflow';
}

export function getSessionActivityLabel(
  s: Pick<PersistedModelExchangeSession, 'step' | 'mlTask' | 'activity' | 'dataset'>,
): string {
  if (s.activity && s.activity.trim()) return s.activity.trim();
  const isClustering = s.mlTask === 'clustering';
  const base = defaultActivity({ step: s.step, mlTask: s.mlTask, isClustering });
  const name = s.dataset?.filename?.trim();
  return name ? `${base} — ${name}` : base;
}

export function readModelExchangeSession(): PersistedModelExchangeSession | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw === snapshotRaw) return snapshotData ?? null;
    snapshotRaw = raw;
    if (!raw) {
      snapshotData = null;
      return null;
    }
    const p = JSON.parse(raw) as PersistedModelExchangeSession;
    if (p.v !== 1 || typeof p.at !== 'string' || typeof p.step !== 'number') {
      snapshotData = null;
      return null;
    }
    snapshotData = p;
    return p;
  } catch {
    snapshotRaw = null;
    snapshotData = null;
    return null;
  }
}

export function writeModelExchangeSession(next: Partial<PersistedModelExchangeSession> & {
  step: WizardStep;
  mlTask: MLTask;
}): void {
  try {
    const prev = readModelExchangeSession();
    const at = new Date().toISOString();
    const merged: PersistedModelExchangeSession = {
      v: 1,
      at,
      step: next.step,
      completedSteps: next.completedSteps ?? prev?.completedSteps ?? [],
      mlTask: next.mlTask,
      dataset: next.dataset !== undefined ? next.dataset : (prev?.dataset ?? null),
      targetColumn: next.targetColumn ?? prev?.targetColumn ?? '',
      featureColumns: next.featureColumns ?? prev?.featureColumns ?? [],
      runId: next.runId !== undefined ? next.runId : (prev?.runId ?? null),
      clusteringRunId: next.clusteringRunId !== undefined ? next.clusteringRunId : (prev?.clusteringRunId ?? null),
      trainingFinished: next.trainingFinished ?? prev?.trainingFinished ?? false,
      activity: next.activity,
    };
    if (!merged.activity) {
      merged.activity = getSessionActivityLabel({
        step: merged.step,
        mlTask: merged.mlTask,
        activity: undefined,
        dataset: merged.dataset,
      });
    }
    const stored = JSON.stringify(merged);
    localStorage.setItem(STORAGE_KEY, stored);
    snapshotRaw = localStorage.getItem(STORAGE_KEY);
    snapshotData = merged;
  } catch {
    /* storage full or disabled */
  }
}

export function clearModelExchangeSession(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
    snapshotRaw = null;
    snapshotData = null;
  } catch {
    /* ignore */
  }
}

export function formatSessionTime(iso: string): string {
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return '—';
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(t);
}

export function formatSessionRelative(iso: string): string {
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return '—';
  const diffMs = Date.now() - t;
  const sec = Math.floor(diffMs / 1000);
  if (sec < 60) return 'Just now';
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m ago`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr}h ago`;
  return formatSessionTime(iso);
}
