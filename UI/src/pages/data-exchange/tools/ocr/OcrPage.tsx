import { useState } from 'react';
import { ArrowLeft, Upload, Play, FileText, ScanLine, File, FileImage } from 'lucide-react';
import '../ToolPage.css';

type OcrStep = 'upload' | 'processing' | 'results';

const OCR_API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://localhost:8000';
const OCR_BACKEND_ENABLED = false;

const OCR_STEPS: { id: OcrStep; label: string; Icon: React.ElementType }[] = [
  { id: 'upload',     label: 'Upload',     Icon: Upload   },
  { id: 'processing', label: 'Processing', Icon: Play     },
  { id: 'results',    label: 'Results',    Icon: FileText },
];

const OcrPage = ({ onBack }: { onBack: () => void }) => {
  const [activeStep, setActiveStep] = useState<OcrStep>('upload');
  const activeIdx = OCR_STEPS.findIndex(s => s.id === activeStep);

  const syncOcrStepToBackend = async (step: OcrStep) => {
    if (!OCR_BACKEND_ENABLED) return;
    try {
      await fetch(`${OCR_API_BASE}/ocr/step`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ step }),
      });
    } catch {
      // Placeholder intentionally ignores network failures until backend is connected.
    }
  };

  const handleStepChange = (step: OcrStep) => {
    setActiveStep(step);
    void syncOcrStepToBackend(step);
  };

  // Build stepper elements as flat array to avoid Fragment-key issues
  const stepperNodes: React.ReactNode[] = [];
  OCR_STEPS.forEach((step, idx) => {
    const isActive = step.id === activeStep;
    const isDone   = activeIdx > idx;
    if (idx > 0) {
      stepperNodes.push(
        <div key={`conn-${idx}`} className={`xt-connector${isDone ? ' done' : ''}`} />
      );
    }
    stepperNodes.push(
      <div key={step.id} className="xt-step" onClick={() => handleStepChange(step.id)}>
        <div className={`xt-step-circle${isActive ? ' active' : isDone ? ' done' : ''}`}>
          <step.Icon size={20} />
        </div>
        <span className={`xt-step-label${isActive ? ' active' : ''}`}>{step.label}</span>
      </div>
    );
  });

  return (
    <div className="xt-page">
      <button className="xt-back-btn" onClick={onBack}>
        <ArrowLeft size={14} /> Back to Data Tools
      </button>

      {/* Header */}
      <div className="xt-header">
        <div className="xt-header-icon">
          <ScanLine size={26} />
        </div>
        <div>
          <h1 className="xt-header-title">Intelligent Document Extraction</h1>
          <p className="xt-header-subtitle">
            Transform documents into structured data with AI-powered vision
          </p>
        </div>
      </div>

      {/* Stepper */}
      <div className="xt-stepper-card">
        <div className="xt-stepper">{stepperNodes}</div>
      </div>

      {/* Dropzone */}
      <div className="xt-dropzone">
        <div className="xt-dropzone-icon">
          <Upload size={28} />
        </div>
        <h3 className="xt-dropzone-title">Drop files here or click to browse</h3>
        <p className="xt-dropzone-subtitle">
          Upload single files or drag a folder for bulk processing
        </p>
        <div className="xt-file-badges">
          <span className="xt-file-badge"><File size={13} /> PDF</span>
          <span className="xt-file-badge"><FileImage size={13} /> PNG, JPG</span>
          <span className="xt-max-size">• Max 50MB per file</span>
        </div>
      </div>
    </div>
  );
};

export default OcrPage;
