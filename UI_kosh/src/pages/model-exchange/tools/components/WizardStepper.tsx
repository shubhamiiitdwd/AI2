import type { WizardStep } from '../types';

const STEPS = [
  { label: 'Select Dataset', icon: '📋' },
  { label: 'Configure Data', icon: '⚙️' },
  { label: 'Configuration', icon: '🔧' },
  { label: 'Training', icon: '▶️' },
  { label: 'Results', icon: '📊' },
];

const CLUSTERING_OVERRIDES: Record<number, { label: string; icon: string }> = {
  3: { label: 'Clustering', icon: '🔬' },
};

interface Props {
  currentStep: WizardStep;
  onStepClick?: (step: WizardStep) => void;
  completedSteps?: number[];
  isClustering?: boolean;
}

export default function WizardStepper({ currentStep, onStepClick, completedSteps = [], isClustering = false }: Props) {
  return (
    <div className="aw-stepper">
      {STEPS.map((s, i) => {
        const step = isClustering && CLUSTERING_OVERRIDES[i] ? CLUSTERING_OVERRIDES[i] : s;
        const isCompleted = completedSteps.includes(i);
        const isActive = i === currentStep;
        const isClickable = isCompleted || i <= currentStep;

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
                if (isClickable && onStepClick) onStepClick(i as WizardStep);
              }}
              style={{ cursor: isClickable ? 'pointer' : 'default' }}
            >
              <div className="aw-step-circle">
                {isCompleted ? '✓' : step.icon}
              </div>
              <span className="aw-step-label">{step.label}</span>
            </div>
            {i < STEPS.length - 1 && <div className={`aw-step-connector ${i < currentStep ? 'aw-step-connector--done' : ''}`} />}
          </div>
        );
      })}
    </div>
  );
}
