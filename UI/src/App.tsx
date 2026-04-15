import { useState, useEffect, useRef, useCallback } from 'react';
import { Routes, Route, Navigate, useLocation, useNavigate, useParams } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'sonner';
import Navbar from './layouts/Navbar';
import LoginPage from './pages/LoginPage';
import HomeView from './pages/home/HomeView';
import DataExchangeView from './pages/data-exchange/DataExchangeView';
import ModelExchangeView from './pages/model-exchange/ModelExchangeView';
import { ModelExchangeNavProvider } from './pages/model-exchange/ModelExchangeNavContext';
import DatasetsPage from './pages/datasets/DatasetsPage';
import ChatButton from './layouts/ChatButton';
import { AuthContext, getStoredAuth, clearStoredAuth } from './auth/authContext';
import './App.css';

const queryClient = new QueryClient();

// ─── Auth Guard ───────────────────────────────────────────────────────────────
// Completely separate render tree — no router URLs exist until authenticated.
function AuthGuard({ children }: { children: React.ReactNode }) {
  const user = getStoredAuth();
  if (!user) return <LoginPage />;
  return <>{children}</>;
}

// ─── Zoom constants ──────────────────────────────────────────────────────────
const ZOOM_MIN  = 0.6;
const ZOOM_MAX  = 1.4;
const ZOOM_KEY  = 'ui-zoom';

// ─── Inner app (runs only after auth) ────────────────────────────────────────
function AppShell() {
  const { section } = useParams<{ section: string }>();
  const location = useLocation();
  const navigate = useNavigate();
  const user = getStoredAuth()!;

  useEffect(() => {
    const previous = window.history.scrollRestoration;
    window.history.scrollRestoration = 'manual';
    return () => {
      window.history.scrollRestoration = previous;
    };
  }, []);

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [location.pathname]);

  // ── Zoom ────────────────────────────────────────────────────────────────────
  // ── Theme ────────────────────────────────────────────────────────────────────
  const [theme, setTheme] = useState<'dark'|'light'>(() =>
    (localStorage.getItem('ui-theme') as 'dark'|'light') || 'dark'
  );

  useEffect(() => {
    document.documentElement.dataset.theme = theme === 'light' ? 'light' : '';
    localStorage.setItem('ui-theme', theme);
  }, [theme]);

  const toggleTheme = useCallback(() =>
    setTheme(t => t === 'dark' ? 'light' : 'dark'), []
  );

  // ── Page loading splash ───────────────────────────────────────────────────────
  const [splashDone, setSplashDone] = useState(false);
  useEffect(() => { const t = setTimeout(() => setSplashDone(true), 700); return () => clearTimeout(t); }, []);

  // ── Zoom ────────────────────────────────────────────────────────────────────
  const [zoom, setZoomState] = useState<number>(() => {
    const s = parseFloat(localStorage.getItem(ZOOM_KEY) ?? '');
    return isNaN(s) ? 1 : Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, s));
  });
  const zoomRef = useRef(zoom);

  const doZoom = (next: number) => {
    const c = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, next));
    zoomRef.current = c;
    document.documentElement.style.zoom = String(c);
    document.documentElement.style.setProperty('--ui-zoom', String(c));
    localStorage.setItem(ZOOM_KEY, String(c));
    setZoomState(c);
  };

  // Restore on mount
  useEffect(() => {
    document.documentElement.style.zoom = String(zoomRef.current);
    document.documentElement.style.setProperty('--ui-zoom', String(zoomRef.current));
  }, []);

  // Ctrl + Wheel → cursor-anchored zoom (Figma/Maps style)
  useEffect(() => {
    const handler = (e: WheelEvent) => {
      if (!e.ctrlKey && !e.metaKey) return;
      e.preventDefault();
      const prev   = zoomRef.current;
      const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
      const next   = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, prev * factor));
      if (next === prev) return;
      // Content-space cursor position before zoom
      const cx = (window.scrollX + e.clientX) / prev;
      const cy = (window.scrollY + e.clientY) / prev;
      doZoom(next);
      // Compensate scroll so cursor stays over same content
      requestAnimationFrame(() =>
        window.scrollTo(cx * next - e.clientX, cy * next - e.clientY)
      );
    };
    window.addEventListener('wheel', handler, { passive: false });
    return () => window.removeEventListener('wheel', handler);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Navigation ──────────────────────────────────────────────────────────────
  const handleLogout = (): void => {
    clearStoredAuth();
    // Full-page reload — tears down authenticated render tree cleanly
    window.location.reload();
  };

  const handleNavigate = (viewId: string): void => {
    if (viewId === 'solution') {
      window.open('https://ai-factory.nttsmartworld.com/asset-library?view=periodic', '_blank', 'noopener,noreferrer');
      return;
    }
    navigate(`/aisphere/${viewId}`);
    window.scrollTo(0, 0);
  };

  const renderSection = () => {
    switch (section) {
      case 'home':     return <HomeView onNavigate={handleNavigate} />;
      case 'data':     return <DataExchangeView />;
      case 'model':    return <ModelExchangeView />;
      case 'datasets': return <DatasetsPage />;
      default:         return <HomeView onNavigate={handleNavigate} />;
    }
  };

  return (
    <AuthContext.Provider value={{ user, logout: handleLogout }}>
      <ModelExchangeNavProvider>
        {!splashDone && (
          <div className="page-splash">
            <div className="page-splash-ring" />
          </div>
        )}
        <div className="app-container">
          <Navbar
            showBack={section !== 'home' && section !== undefined}
            onBack={() => navigate('/aisphere/home')}
            onLogout={handleLogout}
            theme={theme}
            onToggleTheme={toggleTheme}
          />
          <main className="main-content">
            {renderSection()}
          </main>
          <ChatButton />
        </div>
      </ModelExchangeNavProvider>
    </AuthContext.Provider>
  );
}

// ─── Root App: Providers + Auth Guard + Router ───────────────────────────────
function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthGuard>
        <Routes>
          {/* Canonical redirect: /aisphere → /aisphere/home */}
          <Route path="/aisphere" element={<Navigate to="/aisphere/home" replace />} />
          {/* URL-driven section routing — :section is the source of truth */}
          <Route path="/aisphere/:section" element={<AppShell />} />
          {/* Catch-all */}
          <Route path="*" element={<Navigate to="/aisphere/home" replace />} />
        </Routes>
      </AuthGuard>
      <Toaster richColors position="top-right" />
    </QueryClientProvider>
  );
}

export default App;
