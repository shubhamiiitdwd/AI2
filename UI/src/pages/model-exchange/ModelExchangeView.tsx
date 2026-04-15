import { useState, useEffect, useCallback } from 'react';
import Dashboard from './Dashboard';
import AutoMLWizard from './tools/automl-wizard/AutoMLWizard';
import { useModelExchangeNav } from './ModelExchangeNavContext';
import type { TrainingRunSummary } from './tools/automl-wizard/types';

export default function ModelExchangeView() {
  const [view, setView] = useState<'dashboard' | 'wizard'>('dashboard');
  const [wizardInitialRun, setWizardInitialRun] = useState<TrainingRunSummary | null>(null);
  const { setWizardNav } = useModelExchangeNav();

  const clearWizardInitial = useCallback(() => setWizardInitialRun(null), []);

  useEffect(() => {
    if (view === 'wizard') {
      setWizardNav(true, () => {
        setWizardInitialRun(null);
        setView('dashboard');
      });
    } else {
      setWizardNav(false, null);
    }
    return () => setWizardNav(false, null);
  }, [view, setWizardNav]);

  if (view === 'wizard') {
    return (
      <AutoMLWizard
        initialOpenRun={wizardInitialRun}
        onInitialOpenConsumed={clearWizardInitial}
      />
    );
  }

  return (
    <Dashboard
      onStartProject={() => {
        setWizardInitialRun(null);
        setView('wizard');
      }}
      onOpenRunResults={(run) => {
        setWizardInitialRun(run);
        setView('wizard');
      }}
    />
  );
}
