//Uncomment this code from line 2-41 to get the feature of Login, but for the time being, its commented
// import { createContext, useContext } from 'react';

// const AUTH_KEY = 'iagentops_auth';

// export interface AuthUser {
//   username: string;
// }

// // Persist auth to sessionStorage so a full-page reload restores it
// export function getStoredAuth(): AuthUser | null {
//   try {
//     const raw = sessionStorage.getItem(AUTH_KEY);
//     return raw ? (JSON.parse(raw) as AuthUser) : null;
//   } catch {
//     return null;
//   }
// }

// export function setStoredAuth(user: AuthUser): void {
//   sessionStorage.setItem(AUTH_KEY, JSON.stringify(user));
// }

// export function clearStoredAuth(): void {
//   sessionStorage.removeItem(AUTH_KEY);
// }

// // Context — consumers get the current user or null
// export interface AuthContextValue {
//   user: AuthUser | null;
//   logout: () => void;
// }

// export const AuthContext = createContext<AuthContextValue>({
//   user: null,
//   logout: () => {},
// });

// export function useAuth(): AuthContextValue {
//   return useContext(AuthContext);
// }


// This block of code just bypasses the login feature for the time being
import { createContext, useContext } from 'react';

const AUTH_KEY = 'iagentops_auth';

export interface AuthUser {
  username: string;
}

// Persist auth to sessionStorage so a full-page reload restores it
export function getStoredAuth(): AuthUser | null {
  // 🚀 LOGIN BYPASS: Always return a dummy user instantly
  return { username: 'dev_user' };
  
  /* Original logic:
  try {
    const raw = sessionStorage.getItem(AUTH_KEY);
    return raw ? (JSON.parse(raw) as AuthUser) : null;
  } catch {
    return null;
  }
  */
}

export function setStoredAuth(user: AuthUser): void {
  sessionStorage.setItem(AUTH_KEY, JSON.stringify(user));
}

export function clearStoredAuth(): void {
  sessionStorage.removeItem(AUTH_KEY);
}

// Context — consumers get the current user or null
export interface AuthContextValue {
  user: AuthUser | null;
  logout: () => void;
}

export const AuthContext = createContext<AuthContextValue>({
  // 🚀 LOGIN BYPASS: Set the default user in context
  user: { username: 'dev_user' },
  logout: () => {},
});

export function useAuth(): AuthContextValue {
  return useContext(AuthContext);
}
