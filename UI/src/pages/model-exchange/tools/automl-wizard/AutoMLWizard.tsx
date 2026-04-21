import { useState, useCallback, useMemo, useEffect } from 'react';
import './AutoMLWizard.css';
import type {
  WizardStep, DatasetMetadata, ColumnInfo, MLTask, ModelType, TrainingStartRequest, ClusteringResultResponse,
  TrainingRunSummary,
} from './types';
import WizardStepper from './components/WizardStepper';
import StepSelectDataset from './components/StepSelectDataset';
import StepConfigureData from './components/StepConfigureData';
import StepConfiguration from './components/StepConfiguration';
import StepTraining from './components/StepTraining';
import StepResults from './components/StepResults';
import StepClustering from './components/StepClustering';
import * as api from './api';

interface AutoMLWizardProps {
  initialOpenRun?: TrainingRunSummary | null;
  onInitialOpenConsumed?: () => void;
}

const AutoMLWizard = ({ initialOpenRun, onInitialOpenConsumed }: AutoMLWizardProps) => {
  const [step, setStep] = useState<WizardStep>(0);
  const [completedSteps, setCompletedSteps] = useState<number[]>([]);

  const [dataset, setDataset] = useState<DatasetMetadata | null>(null);
  const [columns, setColumns] = useState<ColumnInfo[]>([]);
  const [targetColumn, setTargetColumn] = useState('');
  const [featureColumns, setFeatureColumns] = useState<string[]>([]);
  const [mlTask, setMlTask] = useState<MLTask>('classification');
  const [selectedModels, setSelectedModels] = useState<ModelType[]>(['DRF', 'GBM', 'XGBoost']);
  const [autoMode, setAutoMode] = useState(false);
  const [trainTestSplit, setTrainTestSplit] = useState(0.8);
  const [nfolds, setNfolds] = useState(5);
  const [maxModels, setMaxModels] = useState(20);
  const [maxRuntimeSecs, setMaxRuntimeSecs] = useState(300);
  const [runId, setRunId] = useState<string | null>(null);
  /** True after the current run has finished training (drives Training-step review UI; reset when a new run starts). */
  const [trainingFinished, setTrainingFinished] = useState(false);
  const [clusteringResult, setClusteringResult] = useState<ClusteringResultResponse | null>(null);
  const [clusteringRunId, setClusteringRunId] = useState<string | null>(null);

  const isClustering = mlTask === 'clustering';

  const clusteringDisplayStep = useMemo(() => {
    if (!isClustering) return 0;
    if (step === 0) return 0;
    if (step === 1) return 1;
    if (step === 2) return 2;
    if (step === 3) return 3;
    return 0;
  }, [isClustering, step]);

  const clusteringCompletedSteps = useMemo(() => {
    if (!isClustering) return completedSteps;
    const c: number[] = [];
    if (completedSteps.includes(0)) c.push(0);
    if (completedSteps.includes(1)) c.push(1);
    if (completedSteps.includes(2)) c.push(2);
    if (clusteringResult) c.push(3);
    return c;
  }, [isClustering, completedSteps, clusteringResult]);

  useEffect(() => {
    if (!initialOpenRun) return;
    let cancelled = false;
    const run = initialOpenRun;

    void (async () => {
      try {
        await api.loadPersistedResult(run.run_id);
      } catch {
        if (!cancelled) {
          window.alert('Could not load saved results for this run. It may have been removed from storage.');
          onInitialOpenConsumed?.();
        }
        return;
      }
      if (cancelled) return;

      const isCl = run.run_type === 'clustering' || run.ml_task === 'clustering';

      if (isCl) {
        setMlTask('clustering');
        try {
          const list = await api.listDatasets();
          const ds = list.find((d) => d.id === run.dataset_id);
          if (!ds) {
            if (!cancelled) {
              window.alert('Dataset for this run is no longer available.');
              onInitialOpenConsumed?.();
            }
            return;
          }
          setDataset(ds);
          const colData = await api.getDatasetColumns(ds.id);
          setColumns(colData.columns);
          const features = colData.columns
            .map((c) => c.name)
            .filter((name) => !/cluster_label|cluster_id|prediction_cluster|_cluster$/i.test(name));
          setFeatureColumns(features);
          setClusteringRunId(run.run_id);
          const result = await api.getClusteringResult(run.run_id);
          setClusteringResult(result);
          setCompletedSteps([0, 1, 2, 3]);
          setStep(3);
        } catch {
          if (!cancelled) {
            window.alert('Failed to open clustering results.');
            onInitialOpenConsumed?.();
          }
          return;
        }
      } else {
        const t = (run.ml_task as MLTask) || 'classification';
        setMlTask(t);
        setTargetColumn(run.target_column || '');
        setRunId(run.run_id);
        setTrainingFinished(true);
        setCompletedSteps([0, 1, 2, 3, 4]);
        setStep(4);
      }

      if (!cancelled) onInitialOpenConsumed?.();
    })();

    return () => {
      cancelled = true;
    };
  }, [initialOpenRun?.run_id, onInitialOpenConsumed]);

  const handleDatasetSelect = async (ds: DatasetMetadata) => {
    setDataset(ds);
    try {
      const colData = await api.getDatasetColumns(ds.id);
      setColumns(colData.columns);
      const features = colData.columns
        .map((c) => c.name)
        .filter((name) => !/cluster_label|cluster_id|prediction_cluster|_cluster$/i.test(name));
      setFeatureColumns(features);
    } catch { /* ignore */ }
  };

  /** After importing from data.gov.in: same column prep as upload, then advance to Configure Data (step 1). */
  const continueAfterCatalogImport = useCallback(async (ds: DatasetMetadata) => {
    setDataset(ds);
    try {
      const colData = await api.getDatasetColumns(ds.id);
      setColumns(colData.columns);
      const features = colData.columns
        .map((c) => c.name)
        .filter((name) => !/cluster_label|cluster_id|prediction_cluster|_cluster$/i.test(name));
      setFeatureColumns(features);
    } catch {
      /* ignore */
    }
    try {
      const detected = await api.autoDetectTask(ds.id);
      const t = detected.task as MLTask;
      setMlTask(t);
      if (t === 'clustering') {
        setClusteringRunId(null);
        setClusteringResult(null);
        setCompletedSteps((prev) => [...new Set([...prev, 0, 1])]);
        setStep(1);
        return;
      }
    } catch {
      /* fall through */
    }
    setCompletedSteps((prev) => [...new Set([...prev, 0])]);
    setStep(1);
  }, []);

  const handleClusteringDatasetSelect = async (ds: DatasetMetadata) => {
    setDataset(ds);
    setMlTask('clustering');
    setClusteringRunId(null);
    setClusteringResult(null);
    try {
      const colData = await api.getDatasetColumns(ds.id);
      setColumns(colData.columns);
      const features = colData.columns
        .map((c) => c.name)
        .filter((name) => !/cluster_label|cluster_id|prediction_cluster|_cluster$/i.test(name));
      setFeatureColumns(features);
    } catch { /* ignore */ }
    setCompletedSteps((prev) => [...new Set([...prev, 0, 1])]);
    setStep(1);
  };

  const handleClusteringDetected = useCallback(() => {
    setMlTask('clustering');
    setClusteringRunId(null);
    setClusteringResult(null);
    if (dataset) {
      setFeatureColumns(
        columns
          .map((c) => c.name)
          .filter((name) => !/cluster_label|cluster_id|prediction_cluster|_cluster$/i.test(name)),
      );
      setCompletedSteps((prev) => [...new Set([...prev, 0, 1])]);
      setStep(1);
    }
  }, [dataset, columns]);

  const handleConfigContinue = () => {
    setCompletedSteps((prev) => [...new Set([...prev, 1])]);
    setStep(2);
  };

  const handleClusteringStarted = useCallback(async (rid: string) => {
    setClusteringRunId(rid);
    setClusteringResult(null);
    setCompletedSteps((prev) => [...new Set([...prev, 1, 2])]);

    // Check if this is a cached/completed run — skip directly to results
    try {
      const status = await api.getClusteringStatus(rid);
      if (status.status === 'complete') {
        const result = await api.getClusteringResult(rid);
        setClusteringResult(result);
        setCompletedSteps((prev) => [...new Set([...prev, 1, 2, 3])]);
        setStep(3); // Jump straight to results
        return;
      }
    } catch { /* new run, proceed normally */ }

    setStep(2); // Normal: go to execution/logs view
  }, []);

  const handleStartTraining = async () => {
    setTrainingFinished(false);
    setCompletedSteps((prev) => [...new Set([...prev, 2])]);
    setStep(3);

    try {
      const req: TrainingStartRequest = {
        dataset_id: dataset!.id,
        target_column: targetColumn,
        feature_columns: featureColumns,
        ml_task: mlTask,
        models: autoMode ? [] : selectedModels,
        auto_mode: autoMode,
        train_test_split: trainTestSplit,
        nfolds,
        max_models: maxModels,
        max_runtime_secs: maxRuntimeSecs,
      };
      const res = await api.startTraining(req);
      setRunId(res.run_id);

      // Cache hit — backend returned previous results instantly
      if (res.status === 'complete') {
        setTrainingFinished(true);
        setCompletedSteps((prev) => [...new Set([...prev, 2, 3])]);
        setStep(4); // Jump directly to Results step
      }
    } catch (e) {
      alert('Failed to start training. Check backend connection.');
      setStep(2);
    }
  };

  const handleClusteringComplete = useCallback((result: ClusteringResultResponse) => {
    setClusteringResult(result);
    setCompletedSteps((prev) => [...new Set([...prev, 2])]);
  }, []);

  const goToStep = useCallback((s: WizardStep) => {
    setStep(s);
  }, []);

  const handleClusteringStepClick = useCallback(
    (displayIdx: number) => {
      if (displayIdx === 0) {
        setStep(0);
        return;
      }
      if (displayIdx === 1) {
        setStep(1);
        return;
      }
      if (displayIdx === 2) {
        setStep(2);
        return;
      }
      if (displayIdx === 3 && clusteringResult) {
        setStep(3);
      }
    },
    [clusteringResult],
  );

  const handleTrainingComplete = useCallback(() => {
    setTrainingFinished(true);
    setCompletedSteps((prev) => [...new Set([...prev, 3])]);
    setStep(4);
  }, []);

  return (
    <div className="aw-page">
      <div className="aw-container">
        <div className="aw-title-section">
          <h1 className="aw-title">{isClustering ? 'Clustering' : 'AutoML Wizard'}</h1>
          <p className="aw-subtitle">
            {isClustering
              ? 'Select data, configure features and search, then review logs and results.'
              : 'Train and optimize machine learning models automatically with our guided wizard'}
          </p>
        </div>

        {isClustering ? (
          <WizardStepper
            mode="clustering"
            currentStep={step}
            clusteringDisplayStep={clusteringDisplayStep}
            completedSteps={clusteringCompletedSteps}
            onClusteringStepClick={handleClusteringStepClick}
          />
        ) : (
          <WizardStepper
            mode="automl"
            currentStep={step}
            completedSteps={completedSteps}
            onStepClick={goToStep}
          />
        )}

        <div className="aw-content">
          {step === 0 && (
            <StepSelectDataset
              dataset={dataset}
              onSelect={handleDatasetSelect}
              onCatalogImportComplete={continueAfterCatalogImport}
              onClusteringSelect={handleClusteringDatasetSelect}
              onContinue={async (taskChoice) => {
                if (!dataset) return;
                if (taskChoice === 'auto') {
                  try {
                    const detected = await api.autoDetectTask(dataset.id);
                    const t = detected.task as MLTask;
                    setMlTask(t);
                    if (t === 'clustering') {
                      setClusteringRunId(null);
                      setClusteringResult(null);
                      const colData = await api.getDatasetColumns(dataset.id);
                      setColumns(colData.columns);
                      const features = colData.columns
                        .map((c) => c.name)
                        .filter((name) => !/cluster_label|cluster_id|prediction_cluster|_cluster$/i.test(name));
                      setFeatureColumns(features);
                      setCompletedSteps((prev) => [...new Set([...prev, 0, 1])]);
                      setStep(1);
                      return;
                    }
                  } catch {
                    /* fall through to generic configure step */
                  }
                }
                if (taskChoice && taskChoice !== 'auto') {
                  setMlTask(taskChoice as MLTask);
                }
                setCompletedSteps((prev) => [...new Set([...prev, 0])]);
                setStep(1);
              }}
              onClearDataset={() => {
                setDataset(null);
                setColumns([]);
                setTargetColumn('');
                setFeatureColumns([]);
                setCompletedSteps([]);
              }}
            />
          )}
          {step === 1 && dataset && (
            <StepConfigureData
              datasetId={dataset.id}
              dataset={dataset}
              columns={columns}
              targetColumn={targetColumn}
              featureColumns={featureColumns}
              onTargetChange={setTargetColumn}
              onFeaturesChange={setFeatureColumns}
              onTaskSuggest={setMlTask}
              onContinue={isClustering ? () => {} : handleConfigContinue}
              onClusteringDetected={handleClusteringDetected}
              onBack={() => setStep(0)}
              backLabel={'← Back to Select Dataset'}
              clusteringMode={isClustering}
              onClusteringStart={isClustering ? handleClusteringStarted : undefined}
            />
          )}
          {step === 2 && !isClustering && (
            <StepConfiguration
              mlTask={mlTask}
              selectedModels={selectedModels}
              autoMode={autoMode}
              trainTestSplit={trainTestSplit}
              nfolds={nfolds}
              maxModels={maxModels}
              maxRuntimeSecs={maxRuntimeSecs}
              onTaskChange={setMlTask}
              onModelsChange={setSelectedModels}
              onAutoModeChange={setAutoMode}
              onSplitChange={setTrainTestSplit}
              onNfoldsChange={setNfolds}
              onMaxModelsChange={setMaxModels}
              onMaxRuntimeChange={setMaxRuntimeSecs}
              onStartTraining={handleStartTraining}
              onBack={() => setStep(1)}
            />
          )}
          {/* Keep StepClustering mounted (hidden) on steps 0–1 when a run exists so logs/state survive Select dataset / Clustering configuration. */}
          {isClustering &&
            dataset &&
            (step === 2 ||
              step === 3 ||
              ((step === 0 || step === 1) && clusteringRunId)) && (
              <div
                className={step === 0 || step === 1 ? 'aw-clustering-keepalive' : undefined}
                style={step === 0 || step === 1 ? { display: 'none' } : undefined}
                aria-hidden={step === 0 || step === 1}
              >
                <StepClustering
                  datasetId={dataset.id}
                  featureColumns={featureColumns}
                  columns={columns}
                  wizardView={step === 3 ? 'results' : 'execution'}
                  runId={clusteringRunId}
                  clusteringResult={clusteringResult}
                  onComplete={handleClusteringComplete}
                  onViewResults={() => {
                    setCompletedSteps((prev) => [...new Set([...prev, 3])]);
                    setStep(3);
                  }}
                  onBackToLogs={() => setStep(2)}
                  onBack={() => setStep(1)}
                />
              </div>
            )}
          {step === 3 && !isClustering && runId && (
            <StepTraining runId={runId} onComplete={handleTrainingComplete} reviewMode={trainingFinished} onBack={() => setStep(2)} />
          )}
          {step === 4 && runId && (
            <StepResults runId={runId} onBack={() => setStep(3)} />
          )}
        </div>
      </div>
    </div>
  );
};

export default AutoMLWizard;
