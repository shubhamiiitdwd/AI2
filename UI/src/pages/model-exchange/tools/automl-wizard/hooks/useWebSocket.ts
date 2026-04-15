import { useEffect, useRef, useState, useCallback } from 'react';

export interface WsMessage {
  status: string;
  progress: number;
  stage: string;
  message: string;
  timestamp: string;
}

const RECONNECT_DELAY_MS = 2000;
const MAX_RECONNECT_ATTEMPTS = 10;

export function useTrainingWebSocket(url: string | null) {
  const [messages, setMessages] = useState<WsMessage[]>([]);
  const [connected, setConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WsMessage | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectAttempts = useRef(0);
  const isClosed = useRef(false);

  const connect = useCallback(() => {
    if (!url || isClosed.current) return;

    // Don't reconnect if already connected
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      reconnectAttempts.current = 0;
    };

    ws.onclose = () => {
      setConnected(false);
      // Auto-reconnect unless intentionally closed or training is done
      if (!isClosed.current && reconnectAttempts.current < MAX_RECONNECT_ATTEMPTS) {
        reconnectAttempts.current += 1;
        reconnectTimer.current = setTimeout(() => {
          connect();
        }, RECONNECT_DELAY_MS);
      }
    };

    ws.onerror = () => {
      setConnected(false);
    };

    ws.onmessage = (event) => {
      try {
        const msg: WsMessage = JSON.parse(event.data);
        setMessages((prev) => [...prev, msg]);
        setLastMessage(msg);

        // Stop reconnecting if training is done
        if (msg.status === 'complete' || msg.status === 'failed' || msg.status === 'stopped') {
          isClosed.current = true;
        }
      } catch { /* ignore parse errors */ }
    };
  }, [url]);

  const disconnect = useCallback(() => {
    isClosed.current = true;
    if (reconnectTimer.current) {
      clearTimeout(reconnectTimer.current);
      reconnectTimer.current = null;
    }
    wsRef.current?.close();
    wsRef.current = null;
  }, []);

  useEffect(() => {
    isClosed.current = false;
    reconnectAttempts.current = 0;
    connect();
    return () => disconnect();
  }, [connect, disconnect]);

  const clearMessages = useCallback(() => setMessages([]), []);

  return { messages, lastMessage, connected, clearMessages };
}
