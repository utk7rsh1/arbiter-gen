// EpisodeControls.jsx — Topbar (wordmark + pill nav) + Controls Strip (levels, model, seed, speed, step, buttons)

window.EpisodeControls = function EpisodeControls({
  level, onLevelChange,
  modelMode, onModelChange,
  seed, onSeedChange,
  speed, onSpeedChange,
  step, maxSteps, isRunning, isDone,
  onStep, onPause, onReset,
  activeTab, onTabChange,
  serverStatus,
  domainLabel,
  onBackToLanding,
  onChangeDomain,
}) {
  const levels = ['L1','L2','L3','L4','L5','L6','L7'];
  const models = ['UNTRAINED', 'SFT ONLY', 'FULL ARBITER'];
  const speeds = ['MANUAL','1×','2×','5×'];
  const tabs   = ['LIVE DEMO', 'ARMS RACE', 'COMPARISON', 'STATS', 'TRAINING'];
  const tabIds = ['LIVE', 'ARMS_RACE', 'COMPARISON', 'STATS', 'TRAINING'];

  const progressPct = maxSteps > 0 ? Math.round((step / maxSteps) * 100) : 0;

  const getModelActiveClass = (m) => {
    if (modelMode !== m) return '';
    if (m === 'UNTRAINED') return 'active-untrained';
    if (m === 'SFT ONLY') return 'active-sft';
    return 'active-full';
  };

  const dotColor = serverStatus === 'ok' ? 'var(--green)' : serverStatus === 'error' ? 'var(--red)' : 'var(--amber)';

  return (
    <div style={{ userSelect: 'none' }}>
      {/* ── TOPBAR ──────────────────────────────────────────── */}
      <div className="topbar">
        {/* Left: wordmark + status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginRight: '32px' }}>
          <span className="topbar-wordmark">ARBITER</span>
          <span className="status-dot" style={{ background: dotColor, boxShadow: `0 0 8px ${dotColor}` }} title={`Server: ${serverStatus}`} />
        </div>

        {/* Center: pill nav */}
        <div style={{ flex: 1, display: 'flex', justifyContent: 'center' }}>
          <div className="pill-nav">
            {tabs.map((t, i) => (
              <button
                key={t}
                id={`tab-${tabIds[i]}`}
                className={`pill-nav-item ${activeTab === tabIds[i] ? 'active' : ''}`}
                onClick={() => onTabChange(tabIds[i])}
              >{t}</button>
            ))}
          </div>
        </div>

        {/* Right: domain label + model badge */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          {domainLabel && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--text-muted)', letterSpacing: '0.08em' }}>DOMAIN:</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--cyan)', fontWeight: 500, maxWidth: 140, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{domainLabel}</span>
              {onChangeDomain && (
                <button onClick={onChangeDomain} style={{ background: 'none', border: '1px solid var(--border-light)', borderRadius: 999, padding: '2px 10px', fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--text-muted)', cursor: 'pointer' }}>change</button>
              )}
            </div>
          )}
          <span className="model-badge">
            {modelMode === 'FULL ARBITER' ? '⚡ ' : ''}{modelMode}
          </span>
          {onBackToLanding && (
            <button onClick={onBackToLanding} title="Back to Landing" style={{ background: 'none', border: '1px solid var(--border-light)', borderRadius: 999, padding: '4px 12px', fontFamily: 'var(--font-body)', fontSize: '12px', color: 'var(--text-secondary)', cursor: 'pointer' }}>← Home</button>
          )}
        </div>
      </div>

      {/* ── CONTROLS STRIP ──────────────────────────────────── */}
      <div className="controls-strip">
        {/* Level pills */}
        <div style={{ display: 'flex', gap: '4px', flexShrink: 0 }}>
          {levels.map(l => (
            <button
              key={l}
              className={`level-pill ${level === l ? 'active' : ''}`}
              onClick={() => onLevelChange(l)}
              id={`level-${l}`}
            >{l}</button>
          ))}
        </div>

        <div className="ctrl-separator" />

        {/* Model toggle */}
        <div className="model-toggle" style={{ flexShrink: 0 }}>
          {models.map(m => (
            <button
              key={m}
              className={`model-option ${getModelActiveClass(m)}`}
              onClick={() => onModelChange(m)}
              id={`model-${m.replace(/\s+/g,'-').toLowerCase()}`}
            >{m}</button>
          ))}
        </div>

        <div className="ctrl-separator" />

        {/* Seed */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexShrink: 0 }}>
          <span style={{ fontFamily: 'var(--font-body)', fontSize: '10px', color: 'var(--text-muted)', letterSpacing: '0.1em', fontWeight: 500 }}>SEED</span>
          <input
            id="seed-input"
            type="number"
            value={seed}
            onChange={e => onSeedChange(Number(e.target.value))}
            className="seed-input"
          />
        </div>

        <div className="ctrl-separator" />

        {/* Speed pills */}
        <div style={{ display: 'flex', gap: '4px', flexShrink: 0 }}>
          {speeds.map(s => (
            <button
              key={s}
              className={`speed-pill ${speed === s ? 'active' : ''}`}
              onClick={() => onSpeedChange(s)}
              id={`speed-${s}`}
            >{s}</button>
          ))}
        </div>

        {/* Spacer */}
        <div style={{ flex: 1 }} />

        {/* Step counter with progress bar */}
        <div className="step-counter" style={{ minWidth: '90px', flexShrink: 0 }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '4px' }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 500, fontSize: '12px', color: isDone ? 'var(--green)' : 'var(--text-primary)' }}>
              {isDone ? 'DONE' : `STEP ${step}`}
            </span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-muted)' }}>/ {maxSteps}</span>
          </div>
          <div className="step-progress-bar">
            <div className="step-progress-fill" style={{ width: `${progressPct}%` }} />
          </div>
        </div>

        <div className="ctrl-separator" />

        {/* Control buttons */}
        <div style={{ display: 'flex', gap: '6px', alignItems: 'center', flexShrink: 0 }}>
          <button id="btn-step" className="ctrl-btn" onClick={onStep} disabled={isDone} title="Step / Run">▶</button>
          <button id="btn-pause" className="ctrl-btn" onClick={onPause} title="Pause" style={{ color: isRunning ? 'var(--cyan)' : undefined }}>⏸</button>
          <button id="btn-reset" className="ctrl-btn" onClick={onReset} title="Reset Episode">↺</button>
        </div>
      </div>
    </div>
  );
};
