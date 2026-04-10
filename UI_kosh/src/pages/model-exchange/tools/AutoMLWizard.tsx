import { useState, useCallback } from 'react';
import './AutoMLWizard.css';
import type { WizardStep, DatasetMetadata, ColumnInfo, MLTask, ModelType, TrainingStartRequest, ClusteringResultResponse } from './types';
import WizardStepper from './components/WizardStepper';
import StepSelectDataset from './components/StepSelectDataset';
import StepConfigureData from './components/StepConfigureData';
import StepConfiguration from './components/StepConfiguration';
import StepTraining from './components/StepTraining';
import StepResults from './components/StepResults';
import StepClustering from './components/StepClustering';
import * as api from './api';

interface Props {
  onBack: () => void;
}

const AutoMLWizard = ({ onBack }: Props) => {
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
  const [clusteringResult, setClusteringResult] = useState<ClusteringResultResponse | null>(null);
  const [clusteringDirect, setClusteringDirect] = useState(false);
  // Save pre-clustering state so user can go back to clustering results
  const [preClusteringDataset, setPreClusteringDataset] = useState<DatasetMetadata | null>(null);
  const [preClusteringColumns, setPreClusteringColumns] = useState<ColumnInfo[]>([]);
  const [preClusteringFeatures, setPreClusteringFeatures] = useState<string[]>([]);
  const [postClusteringActive, setPostClusteringActive] = useState(false);

  const isClustering = mlTask === 'clustering';
  const showClusteringInStepper = isClustering || (postClusteringActive && !!clusteringResult);

  const handleDatasetSelect = async (ds: DatasetMetadata) => {
    setDataset(ds);
    try {
      const colData = await api.getDatasetColumns(ds.id);
      setColumns(colData.columns);
      const features = colData.columns.map((c) => c.name);
      setFeatureColumns(features);
    } catch { /* ignore */ }
    // Stay on step 0 to show dataset info + task picker; user clicks "Continue" to proceed
  };

  const handleClusteringDatasetSelect = async (ds: DatasetMetadata) => {
    setDataset(ds);
    setMlTask('clustering');
    setClusteringDirect(true);
    try {
      const colData = await api.getDatasetColumns(ds.id);
      setColumns(colData.columns);
      const features = colData.columns.map((c) => c.name);
      setFeatureColumns(features);
    } catch { /* ignore */ }
    setCompletedSteps((prev) => [...new Set([...prev, 0, 1, 2])]);
    setStep(3);
  };

  const handleClusteringDetected = useCallback(() => {
    setMlTask('clustering');
    setClusteringDirect(true);
    if (dataset) {
      setFeatureColumns(columns.map((c) => c.name));
      setCompletedSteps((prev) => [...new Set([...prev, 0, 1, 2])]);
      setStep(3);
    }
  }, [dataset, columns]);

  const handleConfigContinue = () => {
    setCompletedSteps((prev) => [...new Set([...prev, 1])]);
    setStep(2);
  };

  const handleStartTraining = async () => {
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
    } catch (e) {
      alert('Failed to start training. Check backend connection.');
      setStep(2);
    }
  };

  const [clusteringRunId, setClusteringRunId] = useState<string | null>(null);

  const handleClusteringComplete = useCallback((result: ClusteringResultResponse) => {
    setClusteringResult(result);
    setClusteringRunId(result.run_id);
    setCompletedSteps((prev) => [...new Set([...prev, 3])]);
  }, []);

  const [postClusteringDataset, setPostClusteringDataset] = useState<DatasetMetadata | null>(null);
  const [postClusteringColumns, setPostClusteringColumns] = useState<ColumnInfo[]>([]);

  const handlePostClusteringContinue = useCallback(async () => {
    if (!clusteringRunId || !dataset) return;

    // If we already applied cluster labels before, reuse that dataset
    if (postClusteringDataset) {
      setPreClusteringDataset(dataset);
      setPreClusteringColumns(columns);
      setPreClusteringFeatures(featureColumns);
      setDataset(postClusteringDataset);
      setColumns(postClusteringColumns);
      setFeatureColumns(postClusteringColumns.map((c) => c.name));
      setMlTask('classification');
      setClusteringDirect(false);
      setPostClusteringActive(true);
      setCompletedSteps((prev) => [...new Set([...prev, 0, 3])]);
      setStep(1);
      return;
    }

    try {
      setPreClusteringDataset(dataset);
      setPreClusteringColumns(columns);
      setPreClusteringFeatures(featureColumns);

      const newDs = await api.applyClusterLabels(clusteringRunId, dataset.id);
      setDataset(newDs);
      const colData = await api.getDatasetColumns(newDs.id);
      setColumns(colData.columns);
      setFeatureColumns(colData.columns.map((c) => c.name));
      setPostClusteringDataset(newDs);
      setPostClusteringColumns(colData.columns);
      setTargetColumn('');
      setMlTask('classification');
      setClusteringDirect(false);
      setPostClusteringActive(true);
      setCompletedSteps((prev) => [...new Set([...prev, 0, 3])]);
      setStep(1);
    } catch {
      alert('Failed to apply cluster labels. Try again.');
    }
  }, [clusteringRunId, dataset, columns, featureColumns, postClusteringDataset, postClusteringColumns]);

  const handleBackToClusteringResults = useCallback(() => {
    if (preClusteringDataset) {
      setDataset(preClusteringDataset);
      setColumns(preClusteringColumns);
      setFeatureColumns(preClusteringFeatures);
    }
    setMlTask('clustering');
    setClusteringDirect(true);
    setPostClusteringActive(false);
    setCompletedSteps((prev) => [...new Set([...prev, 0, 1, 2, 3])]);
    setStep(3);
  }, [preClusteringDataset, preClusteringColumns, preClusteringFeatures]);

  const goToStep = useCallback((s: WizardStep) => {
    if (s === 3 && postClusteringActive && clusteringResult) {
      handleBackToClusteringResults();
      return;
    }
    setStep(s);
  }, [postClusteringActive, clusteringResult, handleBackToClusteringResults]);

  const handleTrainingComplete = useCallback(() => {
    setCompletedSteps((prev) => [...new Set([...prev, 3])]);
    setStep(4);
  }, []);

  const handleReset = () => {
    setStep(0);
    setCompletedSteps([]);
    setDataset(null);
    setColumns([]);
    setTargetColumn('');
    setFeatureColumns([]);
    setRunId(null);
    setAutoMode(false);
    setSelectedModels(['DRF', 'GBM', 'XGBoost']);
    setClusteringResult(null);
    setClusteringRunId(null);
    setClusteringDirect(false);
    setPreClusteringDataset(null);
    setPreClusteringColumns([]);
    setPreClusteringFeatures([]);
    setPostClusteringActive(false);
    setPostClusteringDataset(null);
    setPostClusteringColumns([]);
    setMlTask('classification');
  };

  return (
    <div className="aw-page">
      <div className="aw-header">
        <div className="aw-header-left">
          <span className="aw-logo">AI Kosh</span>
          <span className="aw-logo-sub">NTT DATA</span>
        </div>
        <div className="aw-header-right">
          <button className="aw-header-btn" onClick={handleReset}>↻ Reset</button>
          <button className="aw-header-btn" onClick={onBack}>← Back to Hub</button>
        </div>
      </div>

      <div className="aw-container">
        <div className="aw-title-section">
          <h1 className="aw-title">AutoML Wizard</h1>
          <p className="aw-subtitle">Train and optimize machine learning models automatically with our guided wizard</p>
        </div>

        <WizardStepper currentStep={step} onStepClick={goToStep} completedSteps={completedSteps} isClustering={showClusteringInStepper} />

        <div className="aw-content">
          {step === 0 && (
            <StepSelectDataset
              dataset={dataset}
              onSelect={handleDatasetSelect}
              onClusteringSelect={handleClusteringDatasetSelect}
              onContinue={(taskChoice) => {
                if (!dataset) return;
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
              onContinue={handleConfigContinue}
              onClusteringDetected={handleClusteringDetected}
              onBack={postClusteringActive ? handleBackToClusteringResults : () => setStep(0)}
              backLabel={postClusteringActive ? '← Back to Clustering Results' : '← Back to Select Dataset'}
            />
          )}
          {step === 2 && (
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
          {step === 3 && isClustering && dataset && (
            <StepClustering
              datasetId={dataset.id}
              featureColumns={featureColumns}
              columns={columns}
              onComplete={handleClusteringComplete}
              onContinueToSupervisedML={handlePostClusteringContinue}
              onBack={() => setStep(1)}
              existingResult={clusteringResult}
            />
          )}
          {step === 3 && !isClustering && runId && (
            <StepTraining runId={runId} onComplete={handleTrainingComplete} reviewMode={completedSteps.includes(3)} onBack={() => setStep(2)} />
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
