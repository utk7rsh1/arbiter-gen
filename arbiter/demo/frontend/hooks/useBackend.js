// useBackend.js — all fetch calls to FastAPI
// Note: server.py uses /sessions (plural), not /session

const BASE_URL = window.location.origin;

window.useBackend = function useBackend() {
  const { useState, useCallback } = React;
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const call = useCallback(async (method, path, body) => {
    setLoading(true);
    setError(null);
    try {
      const opts = {
        method,
        headers: { 'Content-Type': 'application/json' },
      };
      if (body !== undefined) opts.body = JSON.stringify(body);
      const res = await fetch(`${BASE_URL}${path}`, opts);
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      return await res.json();
    } catch (e) {
      setError(e.message);
      throw e;
    } finally {
      setLoading(false);
    }
  }, []);

  const createSession = useCallback(async (level = 1, seed = null, checkpoint = 'base') => {
    return call('POST', `/sessions?checkpoint=${checkpoint}`, { level, seed });
  }, [call]);

  const resetSession = useCallback(async (sessionId, seed = null) => {
    return call('POST', `/sessions/${sessionId}/reset`, { seed });
  }, [call]);

  const stepSession = useCallback(async (sessionId, action) => {
    return call('POST', `/sessions/${sessionId}/step`, { action });
  }, [call]);

  const renderSession = useCallback(async (sessionId) => {
    return call('GET', `/sessions/${sessionId}/render`);
  }, [call]);

  const getState = useCallback(async (sessionId) => {
    return call('GET', `/sessions/${sessionId}/metrics`);
  }, [call]);

  const getMetrics = useCallback(async () => {
    return call('GET', '/metrics');
  }, [call]);

  const getHealth = useCallback(async () => {
    return call('GET', '/health');
  }, [call]);

  return {
    loading, error,
    createSession, resetSession, stepSession,
    renderSession, getState, getMetrics, getHealth,
  };
};
