import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';

type Ctx = {
  wizardActive: boolean;
  setWizardNav: (active: boolean, onExit: (() => void) | null) => void;
  exitWizard: () => void;
};

const ModelExchangeNavContext = createContext<Ctx | null>(null);

export function ModelExchangeNavProvider({ children }: { children: ReactNode }) {
  const [wizardActive, setWizardActive] = useState(false);
  const exitRef = useRef<(() => void) | null>(null);

  const setWizardNav = useCallback((active: boolean, onExit: (() => void) | null) => {
    setWizardActive(active);
    exitRef.current = onExit;
  }, []);

  const exitWizard = useCallback(() => {
    exitRef.current?.();
  }, []);

  const value = useMemo(
    () => ({ wizardActive, setWizardNav, exitWizard }),
    [wizardActive, setWizardNav, exitWizard],
  );

  return (
    <ModelExchangeNavContext.Provider value={value}>{children}</ModelExchangeNavContext.Provider>
  );
}

export function useModelExchangeNavOptional(): Ctx | null {
  return useContext(ModelExchangeNavContext);
}

export function useModelExchangeNav(): Ctx {
  const ctx = useContext(ModelExchangeNavContext);
  if (!ctx) {
    throw new Error('useModelExchangeNav must be used within ModelExchangeNavProvider');
  }
  return ctx;
}
