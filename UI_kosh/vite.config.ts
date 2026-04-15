import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'
import { defineConfig, loadEnv, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'

const __dirname = dirname(fileURLToPath(import.meta.url))

function backendHintPlugin(port: string): Plugin {
  return {
    name: 'ai-kosh-backend-hint',
    configureServer(server) {
      server.httpServer?.once('listening', () => {
        console.log('')
        console.log(`\x1b[33m[AI Kosh]\x1b[0m Proxy /team1 → http://127.0.0.1:${port}`)
        console.log(
          `\x1b[33m[AI Kosh]\x1b[0m \x1b[36mnpm run dev:all\x1b[0m waits for the API on port ${port} before starting Vite. Running \x1b[36mnpm run dev\x1b[0m alone needs the API already up.`,
        )
        console.log('')
      })
    },
  }
}

/** Single source of truth: `AI_Kosh_Project/.env` → `BACKEND_PORT` (same as `python -m modules.team1_automl.run_local`). */
const DEFAULT_BACKEND_PORT = '8099'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const backendProjectDir = resolve(__dirname, '../AI_Kosh_Project')
  const backendEnv = loadEnv(mode, backendProjectDir, '')
  const uiEnv = loadEnv(mode, process.cwd(), '')
  const backendPort =
    backendEnv.BACKEND_PORT ||
    uiEnv.VITE_BACKEND_PORT ||
    uiEnv.BACKEND_PORT ||
    DEFAULT_BACKEND_PORT

  return {
    plugins: [react(), backendHintPlugin(backendPort)],
    // Keep client `import.meta.env.VITE_BACKEND_PORT` in sync with the proxy (WS + prod-style URLs).
    define: {
      'import.meta.env.VITE_BACKEND_PORT': JSON.stringify(backendPort),
    },
    server: {
      port: 5173,
      proxy: {
        '/team1': {
          target: `http://127.0.0.1:${backendPort}`,
          changeOrigin: true,
          /** Required for clustering / training WebSocket when the UI uses same-origin ws: URLs. */
          ws: true,
        },
      },
    },
  }
})
