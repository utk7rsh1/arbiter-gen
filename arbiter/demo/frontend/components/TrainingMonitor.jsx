// TrainingMonitor.jsx — Full GRPO training control panel (replaces Gradio training tab)
(function () {
  const { useState, useEffect, useRef, useCallback } = React;

  function StatusBadge({ running, returncode }) {
    const { label, color, bg } = running
      ? { label: 'TRAINING', color: 'var(--green)', bg: 'rgba(16,185,129,0.1)' }
      : returncode === 0
        ? { label: 'COMPLETE', color: 'var(--cyan)', bg: 'rgba(0,196,224,0.1)' }
        : returncode != null
          ? { label: 'STOPPED', color: 'var(--red)', bg: 'rgba(239,68,68,0.1)' }
          : { label: 'IDLE', color: 'var(--text-muted)', bg: 'var(--bg-shell)' };
    return (
      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, padding: '4px 12px', background: bg, borderRadius: 999, fontFamily: 'var(--font-mono)', fontSize: 11, color, fontWeight: 600, letterSpacing: '0.06em' }}>
        {running && <span style={{ width: 7, height: 7, borderRadius: '50%', background: 'var(--green)', boxShadow: '0 0 6px var(--green)', display: 'inline-block', animation: 'pulse-dot 1.5s infinite' }} />}
        {label}
      </span>
    );
  }

  function StatPill({ label, value, color }) {
    return (
      <div style={{ background: 'var(--bg-card)', borderRadius: 12, padding: '12px 16px', border: '1px solid var(--border-light)', boxShadow: 'var(--shadow-card)', minWidth: 80, textAlign: 'center' }}>
        <div style={{ fontFamily: 'var(--font-body)', fontSize: 9, color: 'var(--text-muted)', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 4 }}>{label}</div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 22, fontWeight: 700, color: color || 'var(--text-primary)' }}>{value}</div>
      </div>
    );
  }

  window.TrainingMonitor = function TrainingMonitor() {
    // Form state
    const [checkpoint, setCheckpoint] = useState('lora_sft_v4');
    const [level,      setLevel]      = useState(3);
    const [episodes,   setEpisodes]   = useState(120);
    const [output,     setOutput]     = useState('lora_grpo_v2');
    const [klCoef,     setKlCoef]     = useState(0.1);
    const [lr,         setLr]         = useState('5e-6');

    // Status
    const [status, setStatus]   = useState(null);
    const [logLines, setLog]    = useState([]);
    const [running, setRunning] = useState(false);
    const [submitting, setSub]  = useState(false);

    const logRef = useRef(null);

    const fetchStatus = useCallback(async () => {
      try {
        const res = await fetch('/training/status');
        if (res.ok) {
          const d = await res.json();
          setStatus(d);
          setRunning(d.running);
        }
      } catch (_) {}
    }, []);

    const fetchLog = useCallback(async () => {
      try {
        const res = await fetch('/training-log?last=200');
        if (res.ok) {
          const d = await res.json();
          setLog(d.lines || []);
        }
      } catch (_) {}
    }, []);

    useEffect(() => {
      fetchStatus();
      fetchLog();
      const id = setInterval(() => { fetchStatus(); fetchLog(); }, 2000);
      return () => clearInterval(id);
    }, [fetchStatus, fetchLog]);

    // Auto-scroll log
    useEffect(() => {
      if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
    }, [logLines]);

    const handleStart = async () => {
      setSub(true);
      try {
        const res = await fetch('/training/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            checkpoint, level: +level, episodes: +episodes,
            output, kl_coef: +klCoef, lr: +lr,
          }),
        });
        const d = await res.json();
        if (!res.ok) throw new Error(d.detail || 'Failed');
        await fetchStatus();
      } catch (e) {
        alert('Start failed: ' + e.message);
      } finally {
        setSub(false);
      }
    };

    const handleAbort = async () => {
      if (!confirm('Abort training?')) return;
      await fetch('/training/abort', { method: 'POST' });
      await fetchStatus();
    };

    const last = status?.last_entry;
    const rc   = status?.returncode;

    return (
      <div style={{ height: '100%', display: 'flex', flexDirection: 'column', padding: '20px 24px', gap: 16, overflowY: 'auto', background: 'var(--bg-shell)' }}>
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <div style={{ fontFamily: 'var(--font-display)', fontSize: 20, fontWeight: 800, color: 'var(--text-primary)', marginBottom: 4 }}>
              GRPO Training Monitor
            </div>
            <div style={{ fontFamily: 'var(--font-body)', fontSize: 13, color: 'var(--text-muted)' }}>
              Launch and monitor adversarial GRPO fine-tuning from the browser
            </div>
          </div>
          <StatusBadge running={running} returncode={rc} />
        </div>

        {/* Stats bar */}
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          <StatPill label="Episode"     value={last?.episode ?? '—'}                     color="var(--cyan)" />
          <StatPill label="Mean Reward" value={last ? last.mean_reward?.toFixed(2) : '—'} color="var(--green)" />
          <StatPill label="GRPO Loss"   value={last ? last.grpo_loss?.toFixed(4) : '—'}   color="var(--red)" />
          <StatPill label="Evasion"     value={last ? (last.defender_evasion * 100).toFixed(1) + '%' : '—'} color="var(--amber)" />
          <StatPill label="Level"       value={last?.level ?? '—'}                        color="var(--purple)" />
        </div>

        <div style={{ display: 'flex', gap: 16, flex: 1, minHeight: 0 }}>
          {/* Left: config form */}
          <div style={{ flex: '0 0 300px', display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div style={{ background: 'var(--bg-card)', borderRadius: 14, padding: 20, border: '1px solid var(--border-light)', boxShadow: 'var(--shadow-card)' }}>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)', letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 16 }}>Training Config</div>

              {[
                { label: 'SFT Checkpoint', val: checkpoint, set: setCheckpoint, type: 'text' },
                { label: 'Output Name',    val: output,     set: setOutput,     type: 'text' },
                { label: 'Level',          val: level,      set: setLevel,      type: 'number', min: 1, max: 7 },
                { label: 'Episodes',       val: episodes,   set: setEpisodes,   type: 'number', min: 10, max: 5000 },
                { label: 'KL Coef',        val: klCoef,     set: setKlCoef,     type: 'number', step: 0.01 },
                { label: 'Learning Rate',  val: lr,         set: setLr,         type: 'text' },
              ].map(f => (
                <div key={f.label} style={{ marginBottom: 12 }}>
                  <div style={{ fontFamily: 'var(--font-body)', fontSize: 11, color: 'var(--text-secondary)', marginBottom: 4 }}>{f.label}</div>
                  <input
                    type={f.type}
                    value={f.val}
                    min={f.min}
                    max={f.max}
                    step={f.step}
                    onChange={e => f.set(e.target.value)}
                    disabled={running}
                    style={{ width: '100%', padding: '8px 12px', background: 'var(--bg-shell)', border: '1px solid var(--border-light)', borderRadius: 8, fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-primary)', outline: 'none' }}
                  />
                </div>
              ))}

              <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
                <button
                  onClick={handleStart}
                  disabled={running || submitting}
                  style={{ flex: 1, padding: '10px', background: running || submitting ? 'var(--border-light)' : 'var(--green)', color: running || submitting ? 'var(--text-muted)' : '#000', border: 'none', borderRadius: 8, fontFamily: 'var(--font-body)', fontWeight: 700, fontSize: 13, cursor: running || submitting ? 'not-allowed' : 'pointer' }}
                >
                  {submitting ? 'Starting...' : running ? '⏳ Running...' : '▶ Start Training'}
                </button>
                <button
                  onClick={handleAbort}
                  disabled={!running}
                  style={{ padding: '10px 16px', background: running ? 'rgba(239,68,68,0.1)' : 'var(--bg-shell)', color: running ? 'var(--red)' : 'var(--text-muted)', border: `1px solid ${running ? 'var(--red)' : 'var(--border-light)'}`, borderRadius: 8, fontFamily: 'var(--font-body)', fontWeight: 600, fontSize: 13, cursor: running ? 'pointer' : 'not-allowed' }}
                >
                  ■ Abort
                </button>
              </div>
            </div>
          </div>

          {/* Right: live log */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', background: 'var(--bg-panel-dark)', borderRadius: 14, overflow: 'hidden', border: '1px solid var(--border-dark)' }}>
            <div style={{ padding: '12px 16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid var(--border-dark)' }}>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted-dark)', letterSpacing: '0.08em' }}>STDOUT LOG</span>
              <button onClick={fetchLog} style={{ background: 'none', border: 'none', fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted-dark)', cursor: 'pointer' }}>↺ refresh</button>
            </div>
            <div ref={logRef} style={{ flex: 1, overflowY: 'auto', padding: '12px 16px', fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted-dark)', lineHeight: 1.7 }}>
              {logLines.length === 0
                ? <span style={{ color: 'var(--text-muted-dark)', fontStyle: 'italic' }}>No log output yet — start a training run.</span>
                : logLines.map((l, i) => {
                    const color = l.includes('error') || l.includes('Error') ? 'var(--red)'
                      : l.includes('GRPO') || l.includes('Episode') ? 'var(--cyan)'
                      : l.includes('reward') ? 'var(--green)'
                      : 'var(--text-muted-dark)';
                    return <div key={i} style={{ color, marginBottom: 1 }}>{l || ' '}</div>;
                  })
              }
            </div>
          </div>
        </div>
      </div>
    );
  };
})();
