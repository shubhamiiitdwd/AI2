import type { WizardStep } from '../types';

const AUTOML_STEPS = [
  { label: 'Select Dataset', icon: '📋' },
  { label: 'Configure Data', icon: '⚙️' },
  { label: 'Configuration', icon: '🔧' },
  { label: 'Training', icon: '▶️' },
  { label: 'Results', icon: '📊' },
];

/** Four steps used only for the clustering workflow (no H2O AutoML configuration step). */
const CLUSTERING_STEPS = [
  { label: 'Select Dataset', icon: '📋' },
  { label: 'Clustering configuration', icon: '⚙️' },
  { label: 'Model search & evaluation', icon: '🔬' },
  { label: 'Clustering results', icon: '📊' },
];

interface Props {
  currentStep: WizardStep;
  /** Clustering UI uses indices 0–3 matching CLUSTERING_STEPS (not the 5-step AutoML path). */
  clusteringDisplayStep?: number;
  onStepClick?: (step: WizardStep) => void;
  /** When mode is clustering, maps visual step index (0–3) to wizard navigation. */
  onClusteringStepClick?: (displayIndex: number) => void;
  completedSteps?: number[];
  mode?: 'automl' | 'clustering';
}

export default function WizardStepper({
  currentStep,
  clusteringDisplayStep = 0,
  onStepClick,
  onClusteringStepClick,
  completedSteps = [],
  mode = 'automl',
}: Props) {
  const steps = mode === 'clustering' ? CLUSTERING_STEPS : AUTOML_STEPS;
  const activeIndex = mode === 'clustering' ? clusteringDisplayStep : currentStep;

  return (
    <div className={`aw-stepper ${mode === 'clustering' ? 'aw-stepper--clustering' : ''}`}>
      {steps.map((s, i) => {
        const isCompleted = completedSteps.includes(i);
        const isActive = i === activeIndex;
        const isClickable = isCompleted || i <= activeIndex;

        let cls = 'aw-step';
        if (isCompleted) cls += ' aw-step--completed';
        else if (isActive) cls += ' aw-step--active';
        else cls += ' aw-step--pending';
        if (isClickable) cls += ' aw-step--clickable';

        return (
          <div key={i} className="aw-step-wrapper">
            <div
              className={cls}
              onClick={() => {
                if (!isClickable) return;
                if (mode === 'clustering' && onClusteringStepClick) {
                  onClusteringStepClick(i);
                } else if (onStepClick) {
                  onStepClick(i as WizardStep);
                }
              }}
              style={{ cursor: isClickable ? 'pointer' : 'default' }}
            >
              <div className="aw-step-circle">
                {isCompleted ? '✓' : s.icon}
              </div>
              <span className="aw-step-label">{s.label}</span>
            </div>
            {i < steps.length - 1 && (
              <div className={`aw-step-connector ${i < activeIndex ? 'aw-step-connector--done' : ''}`} />
            )}
          </div>
        );
      })}
    </div>
  );
}
