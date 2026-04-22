/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Team1 AutoML API base (optional; default 127.0.0.1 + VITE_BACKEND_PORT / 8099). */
  readonly VITE_TEAM1_API_URL?: string;
}

declare module '*.png' {
  const src: string;
  export default src;
}
declare module '*.jpg' {
  const src: string;
  export default src;
}
declare module '*.svg' {
  const src: string;
  export default src;
}
