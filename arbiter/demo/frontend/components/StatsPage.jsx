// StatsPage.jsx — Full page: three columns (comparison table, claim accuracy, defender evasion) + episode log

window.StatsPage = function StatsPage({ metricsData }) {
  const { useState, useEffect, useCallback } = React;
  const [metrics, setMetrics] = useState(metricsData || null);
  const [loading, setLoading] = useState(false);

  const fetchMetrics = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch('/metrics');
      if (res.ok) setMetrics(await res.json());
    } catch (e) {}
    setLoading(false);
  }, []);

  useEffect(() => { fetchMetrics(); }, [fetchMetrics]);

  // Synthetic data — slots for real /metrics data when available
  const conditions = [
    { name: 'Untrained', mean: metrics?.untrained_mean ?? 2.3, std: 1.1, accuracy: 0.04, claimAcc: 0.22 },
    { name: 'SFT Only', mean: metrics?.sft_mean ?? 11.4, std: 2.8, accuracy: 0.38, claimAcc: 0.61 },
    { name: 'Full ARBITER', mean: metrics?.full_mean ?? 24.7, std: 3.2, accuracy: 0.87, claimAcc: 0.91 },
  ];
  const claimTypes = [
    { name: 'Causal', acc: 0.91, color: 'var(--cyan)' },
    { name: 'Counterfactual', acc: 0.85, color: 'var(--purple)' },
    { name: 'Theory-of-Mind', acc: 0.78, color: '#7C3AED' },
  ];
  const defenderTypes = [
    { name: 'Link Substitution', evasion: 0.12 },
    { name: 'Record Injection', evasion: 0.18 },
    { name: 'Proxy Laundering', evasion: 0.09 },
    { name: 'Timestamp Manip.', evasion: 0.21 },
  ];
  const episodeLog = Array.from({ length: 20 }, (_, i) => ({
    ep: 280 + i, level: i < 10 ? 3 : 4,
    anomaly: ['proxy_disc', 'adv_inject', 'model_drift'][i % 3],
    reward: +(18 + Math.random() * 10).toFixed(1),
    correct: Math.random() > 0.2,
  }));

  const bestIdx = (arr, key) => {
    let best = 0;
    arr.forEach((c, i) => { if (c[key] > arr[best][key]) best = i; });
    return best;
  };
  const bestMean = bestIdx(conditions, 'mean');
  const bestAcc = bestIdx(conditions, 'accuracy');
  const bestClaim = bestIdx(conditions, 'claimAcc');

  const barAnim = (pct, color, delay) => (
    <div style={{ height: '6px', background: 'var(--border-light)', borderRadius: 'var(--radius-pill)', marginTop: '4px' }}>
      <div style={{
        height: '100%', width: `${pct}%`, background: color,
        borderRadius: 'var(--radius-pill)', transition: `width 0.6s ease ${delay}s`,
      }} />
    </div>
  );

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', padding: '24px', gap: '20px', overflowY: 'auto' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <div style={{ fontFamily: 'var(--font-display)', fontSize: '18px', fontWeight: 700, color: 'var(--text-primary)' }}>
            Evaluation Statistics
          </div>
          <div style={{ fontFamily: 'var(--font-body)', fontSize: '13px', color: 'var(--text-muted)', marginTop: '2px' }}>
            Three-condition evaluation · {metrics ? 'Live data' : 'Synthetic data'}
          </div>
        </div>
        <button onClick={fetchMetrics} style={{
          fontFamily: 'var(--font-mono)', fontSize: '10px', padding: '6px 14px',
          background: 'var(--bg-card)', border: '1px solid var(--border-light)',
          color: 'var(--text-secondary)', cursor: 'pointer', borderRadius: 'var(--radius-pill)',
          boxShadow: 'var(--shadow-card)',
        }}>{loading ? '…' : '↺ REFRESH'}</button>
      </div>

      {/* Three columns */}
      <div style={{ display: 'flex', gap: '16px', flex: 1, minHeight: 0 }}>
        {/* Column 1: Comparison Table */}
        <div className="card" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <div className="card-header"><span className="card-header-label">THREE-CONDITION COMPARISON</span></div>
          <div className="card-body" style={{ flex: 1, overflowY: 'auto' }}>
            <table className="stats-table">
              <thead>
                <tr><th>Condition</th><th style={{ textAlign: 'right' }}>Mean ±σ</th><th style={{ textAlign: 'right' }}>Verdict %</th><th style={{ textAlign: 'right' }}>Claim %</th></tr>
              </thead>
              <tbody>
                {conditions.map((c, i) => (
                  <tr key={c.name}>
                    <td style={{ color: i === 2 ? 'var(--cyan)' : i === 1 ? 'var(--amber)' : 'var(--text-secondary)', fontWeight: i === 2 ? 600 : 400 }}>{c.name}</td>
                    <td style={{ textAlign: 'right' }} className={i === bestMean ? 'highlight-best' : ''}>
                      {c.mean.toFixed(1)} <span style={{ color: 'var(--text-muted)' }}>±{c.std.toFixed(1)}</span>
                    </td>
                    <td style={{ textAlign: 'right' }} className={i === bestAcc ? 'highlight-best' : ''}>
                      {(c.accuracy * 100).toFixed(0)}%
                    </td>
                    <td style={{ textAlign: 'right' }} className={i === bestClaim ? 'highlight-best' : ''}>
                      {(c.claimAcc * 100).toFixed(0)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            {/* Server metrics */}
            {metrics && (
              <div style={{ marginTop: '20px', padding: '12px', background: 'var(--bg-shell)', borderRadius: '12px' }}>
                <div style={{ fontFamily: 'var(--font-body)', fontSize: '10px', color: 'var(--text-muted)', letterSpacing: '0.1em', marginBottom: '8px' }}>LIVE SERVER</div>
                {[
                  { label: 'Sessions', value: metrics.total_sessions ?? 0 },
                  { label: 'Episodes', value: metrics.total_episodes ?? 0 },
                  { label: 'Mean Reward', value: `${metrics.mean_reward ?? '—'} pts` },
                  { label: 'Uptime', value: `${metrics.uptime_s ?? 0}s` },
                ].map(s => (
                  <div key={s.label} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                    <span style={{ fontFamily: 'var(--font-body)', fontSize: '11px', color: 'var(--text-muted)' }}>{s.label}</span>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-primary)' }}>{s.value}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Column 2: Claim Accuracy */}
        <div className="card" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <div className="card-header"><span className="card-header-label">CLAIM ACCURACY BREAKDOWN</span></div>
          <div className="card-body" style={{ flex: 1 }}>
            {claimTypes.map((ct, i) => (
              <div key={ct.name} style={{ marginBottom: '20px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                  <span style={{ fontFamily: 'var(--font-body)', fontSize: '12px', color: 'var(--text-secondary)' }}>{ct.name}</span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '13px', fontWeight: 500, color: 'var(--text-primary)' }}>{(ct.acc * 100).toFixed(0)}%</span>
                </div>
                {barAnim(ct.acc * 100, ct.color, i * 0.1)}
              </div>
            ))}
          </div>
        </div>

        {/* Column 3: Defender Evasion */}
        <div className="card" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <div className="card-header"><span className="card-header-label">DEFENDER EVASION</span></div>
          <div className="card-body" style={{ flex: 1 }}>
            {defenderTypes.map((dt, i) => (
              <div key={dt.name} style={{ marginBottom: '20px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                  <span style={{ fontFamily: 'var(--font-body)', fontSize: '12px', color: 'var(--text-secondary)' }}>{dt.name}</span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '13px', fontWeight: 500, color: 'var(--text-primary)' }}>{(dt.evasion * 100).toFixed(0)}%</span>
                </div>
                {barAnim(dt.evasion * 100, `var(--red)`, i * 0.1)}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Episode log */}
      <div className="card">
        <div className="card-header"><span className="card-header-label">EPISODE LOG · LAST 20</span></div>
        <div className="card-body" style={{ maxHeight: '180px', overflowY: 'auto' }}>
          <table className="stats-table" style={{ fontSize: '11px' }}>
            <thead>
              <tr><th>#</th><th>Level</th><th>Anomaly</th><th style={{ textAlign: 'right' }}>Reward</th><th style={{ textAlign: 'center' }}>Verdict</th></tr>
            </thead>
            <tbody>
              {episodeLog.map(e => (
                <tr key={e.ep}>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{e.ep}</td>
                  <td>L{e.level}</td>
                  <td style={{ fontFamily: 'var(--font-mono)', fontSize: '10px' }}>{e.anomaly}</td>
                  <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)', color: e.reward > 20 ? 'var(--green)' : 'var(--text-primary)' }}>{e.reward}</td>
                  <td style={{ textAlign: 'center', color: e.correct ? 'var(--green)' : 'var(--red)' }}>{e.correct ? '✓' : '✗'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div style={{ fontFamily: 'var(--font-body)', fontSize: '11px', color: 'var(--text-muted)', textAlign: 'center', padding: '4px' }}>
        Synthetic data shown for conditions without real training checkpoints. Refresh to pull live data from /metrics.
      </div>
    </div>
  );
};
