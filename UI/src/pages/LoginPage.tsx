import React, { useState } from 'react';
import { Eye, EyeOff, LogIn, UserPlus } from 'lucide-react';
import logo from '../assets/logo.png';
import { setStoredAuth } from '../auth/authContext';
import './LoginPage.css';
 
const LoginPage = () => {
  // State for toggling between Login and Signup modes
  const [isLogin, setIsLogin] = useState(true);
 
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [successMsg, setSuccessMsg] = useState('');
  const [loading, setLoading] = useState(false);
 
  // Your FastAPI backend URL (Adjust if running on a different port/host)
  const API_BASE_URL = 'http://127.0.0.1:8765/api/auth';
 
  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>): Promise<void> => {
    e.preventDefault();
    setError('');
    setSuccessMsg('');
 
    if (!username.trim() || !password.trim()) {
      setError('Please enter your username and password.');
      return;
    }
 
    setLoading(true);
 
    try {
      const endpoint = isLogin ? '/login' : '/signup';
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          username: username.trim(),
          password: password.trim()
        }),
      });
 
      const data = await response.json();
 
      if (!response.ok) {
        // The backend raises HTTPException with a "detail" key
        throw new Error(data.detail || 'An error occurred. Please try again.');
      }
 
      if (isLogin) {
        // --- LOGIN SUCCESS ---
        // Store the JWT token in sessionStorage (or localStorage)
        sessionStorage.setItem('access_token', data.access_token);
       
        // Call your existing auth context
        setStoredAuth({ username: username.trim() });
       
        // Full-page reload to mount authenticated app
        window.location.reload();
      } else {
        // --- SIGNUP SUCCESS ---
        setSuccessMsg('Account created successfully! Please sign in.');
        setIsLogin(true); // Switch back to login view
        setPassword('');  // Clear password for security
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };
 
  const toggleMode = () => {
    setIsLogin(!isLogin);
    setError('');
    setSuccessMsg('');
    setPassword('');
  };
 
  return (
    <div className="login-root">
      {/* ── Decorative left panel ── */}
      <div className="login-panel-left" aria-hidden="true">
        <div className="login-panel-orb login-panel-orb--a" />
        <div className="login-panel-orb login-panel-orb--b" />
        <div className="login-panel-orb login-panel-orb--c" />
        <div className="login-panel-brand">
          <img src={logo} alt="NTT DATA" className="login-panel-logo" />
          <h2 className="login-panel-title">AI Sphere</h2>
          <p className="login-panel-sub">Your unified platform for Data, Models &amp; Solutions</p>
        </div>
        <div className="login-panel-features">
          <span className="login-panel-chip">⚡ AutoML Pipeline</span>
          <span className="login-panel-chip">🔍 OCR &amp; Vision</span>
          <span className="login-panel-chip">📊 Data Visualization</span>
          <span className="login-panel-chip">🔒 Data Anonymization</span>
        </div>
      </div>

      {/* ── Right form panel ── */}
      <div className="login-panel-right">
        <div className="login-card">
          {/* Logo */}
          <div className="login-logo-wrap">
            <img src={logo} alt="NTT DATA Logo" className="login-logo" />
          </div>

          <div className="login-header">
            <h1 className="login-title">
              {isLogin ? 'Welcome back' : 'Create an Account'}
            </h1>
            <p className="login-subtitle">
              {isLogin ? 'Sign in to access your AI Innovation Hub' : 'Sign up to get started with AI Sphere'}
            </p>
          </div>
 
        <form className="login-form" onSubmit={handleSubmit} noValidate>
          <div className="login-field">
            <label htmlFor="username" className="login-label">Username</label>
            <input
              id="username"
              type="text"
              className="login-input"
              placeholder="Enter your username"
              value={username}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setUsername(e.target.value)}
              autoComplete="username"
              autoFocus
            />
          </div>
 
          <div className="login-field">
            <label htmlFor="password" className="login-label">Password</label>
            <div className="login-password-wrap">
              <input
                id="password"
                type={showPassword ? 'text' : 'password'}
                className="login-input"
                placeholder="Enter your password"
                value={password}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setPassword(e.target.value)}
                autoComplete={isLogin ? 'current-password' : 'new-password'}
              />
              <button
                type="button"
                className="login-eye-btn"
                onClick={() => setShowPassword(!showPassword)}
                aria-label={showPassword ? 'Hide password' : 'Show password'}
              >
                {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
              </button>
            </div>
          </div>
 
          {/* Messages */}
          {error && <p className="login-error">{error}</p>}
          {successMsg && <p className="login-success">{successMsg}</p>}
 
          <button type="submit" className="login-btn" disabled={loading}>
            {loading ? (
              <span className="login-spinner" />
            ) : (
              <>
                {isLogin ? <LogIn size={18} /> : <UserPlus size={18} />}
                <span>{isLogin ? 'Sign In' : 'Sign Up'}</span>
              </>
            )}
          </button>
        </form>
 
        {/* Toggle Login/Signup */}
        <div className="login-toggle">
          <span className="login-toggle-text">
            {isLogin ? "Don't have an account? " : "Already have an account? "}
          </span>
          <button type="button" className="login-toggle-btn" onClick={toggleMode}>
            {isLogin ? 'Sign Up' : 'Sign In'}
          </button>
        </div>

        <p className="login-footer">
          © {new Date().getFullYear()} NTT DATA. All rights reserved.
        </p>
      </div>
    </div>
    </div>
  );
};
 
export default LoginPage;