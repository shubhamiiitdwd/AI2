import { RotateCcw, LogOut, ArrowLeft, Sun, Moon, PanelsTopLeft } from 'lucide-react';
import logo from '../assets/logo.png';
import './Navbar.css';
import { useModelExchangeNavOptional } from '../pages/model-exchange/ModelExchangeNavContext';

interface NavbarProps {
  showBack?: boolean;
  onBack?: () => void;
  onLogout?: () => void;
  theme?: 'dark' | 'light';
  onToggleTheme?: () => void;
}

const Navbar = ({ showBack = false, onBack, onLogout, theme = 'dark', onToggleTheme }: NavbarProps) => {
  const mxNav = useModelExchangeNavOptional();

  return (
    <nav className="navbar">
      <div className="navbar-glow-line" />
      <div className="navbar-container">
        {/* ── Brand ── */}
        <div
          className="navbar-brand"
          onClick={showBack ? onBack : undefined}
          style={{ cursor: showBack ? 'pointer' : 'default' }}
        >
          <div className="navbar-brand-text">
            <img src={logo} alt="NTT DATA" className="navbar-logo-img" />
            <span className="navbar-brand-sub">AI Innovation Hub</span>
          </div>
        </div>

        {/* ── Actions ── */}
        <div className="navbar-actions">
          <button
            type="button"
            className="nb-bubble"
            title="Reset"
            onClick={() => window.location.reload()}
          >
            <span className="nb-bubble-icon"><RotateCcw size={15} /></span>
            <span className="nb-bubble-label">Reset</span>
          </button>
          {mxNav?.wizardActive && (
            <button
              type="button"
              className="nb-bubble"
              onClick={mxNav.exitWizard}
              title="Back to Model Exchange"
            >
              <span className="nb-bubble-icon"><PanelsTopLeft size={15} /></span>
              <span className="nb-bubble-label">Exchange</span>
            </button>
          )}
          {showBack && (
            <button type="button" className="nb-bubble" onClick={onBack} title="Back to Home">
              <span className="nb-bubble-icon"><ArrowLeft size={15} /></span>
              <span className="nb-bubble-label">Home</span>
            </button>
          )}
          <button
            className="nb-bubble nb-bubble--theme"
            onClick={onToggleTheme}
            title={theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
          >
            <span className="nb-bubble-icon">
              {theme === 'dark' ? <Sun size={15} /> : <Moon size={15} />}
            </span>
            <span className="nb-bubble-label">{theme === 'dark' ? 'Light' : 'Dark'}</span>
          </button>
          <button className="nb-bubble nb-bubble--danger" onClick={onLogout} title="Logout">
            <span className="nb-bubble-icon"><LogOut size={15} /></span>
            <span className="nb-bubble-label">Logout</span>
          </button>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
