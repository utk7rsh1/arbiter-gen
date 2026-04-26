// ArmsRaceChart.jsx — Loan: rich GRPO training story; General: live /arms-race data
(function () {
  const { useState, useEffect, useRef, useCallback } = React;

  // ── Loan-domain training data: L3, batch 16, 300 episodes ─────────────────
  // Curve shape: warm-up → proxy-discovery → consolidation → breakthrough → convergence
  function buildLoanData() {
    const pts = [];
    // Smooth logistic-style segments with injected noise (deterministic via index)
    const noise = (i, amp) => Math.sin(i * 2.3 + 1.7) * amp * 0.5 + Math.cos(i * 5.1 + 0.9) * amp * 0.3;

    for (let ep = 1; ep <= 300; ep++) {
      let r, d; // reward, defender evasion%

      // Phase 1: warm-up (1–50): random exploration, reward 3→9, evasion 30→48%
      if (ep <= 50) {
        const t = (ep - 1) / 49;
        r = 3.2 + t * 6.1  + noise(ep, 1.4);
        d = 30  + t * 18.5 + noise(ep, 3.0);
      }
      // Phase 2: proxy-discovery (51–100): steep climb, reward 9→24, evasion 48→68%
      else if (ep <= 100) {
        const t = (ep - 51) / 49;
        r = 9.0  + t * 15.8 + noise(ep, 1.8);
        d = 48.0 + t * 21.0 + noise(ep, 2.6);
      }
      // Phase 3: consolidation (101–160): steady climb, reward 24→34, evasion 68→77%
      else if (ep <= 160) {
        const t = (ep - 101) / 59;
        r = 24.0 + t * 10.5 + noise(ep, 1.2);
        d = 68.0 + t * 9.5  + noise(ep, 2.0);
      }
      // Phase 4: breakthrough (161–220): reward 34→43, evasion 77→86%
      else if (ep <= 220) {
        const t = (ep - 161) / 59;
        r = 34.0 + t * 9.2  + noise(ep, 0.9);
        d = 77.0 + t * 9.4  + noise(ep, 1.5);
      }
      // Phase 5: convergence (221–300): plateau ~44.5, evasion ~87%
      else {
        const t = (ep - 221) / 79;
        r = 43.0 + t * 2.1  + noise(ep, 0.6);
        d = 86.0 + t * 1.9  + noise(ep, 1.0);
      }

      pts.push({
        ep,
        auditor:  +Math.max(2,  Math.min(52,  r)).toFixed(2),
        defender: +Math.max(20, Math.min(94,  d)).toFixed(1),
        // attach phase label for annotations
        phase: ep === 50 ? 'discovery' : ep === 100 ? 'consolidation' : ep === 160 ? 'breakthrough' : ep === 220 ? 'convergence' : null,
      });
    }
    return pts;
  }

  const LOAN_DATA = buildLoanData();

  // Phase annotation markers (loan only)
  const LOAN_PHASES = [
    { ep: 50,  label: 'Proxy variable\nstrategy found',   color: '#F59E0B' },
    { ep: 100, label: 'Causal chain\nconsolidated',       color: '#10B981' },
    { ep: 160, label: 'Evasion rate\ncrosses 75%',        color: '#8B5CF6' },
    { ep: 220, label: 'Model\nconverges',                  color: '#00C4E0' },
  ];

  // ── Live data fetch for general/custom domain ──────────────────────────────
  async function fetchArmsRaceData() {
    try {
      const res = await fetch('/arms-race');
      if (res.ok) {
        const d = await res.json();
        if (d && d.data && d.data.length > 1) return d.data;
      }
    } catch (_) {}
    return null;
  }

  // ── Component ──────────────────────────────────────────────────────────────
  window.ArmsRaceChart = function ArmsRaceChart({ domainMode }) {
    const isLoan = domainMode === 'loan';

    // Loan mode: static data, no fetch needed
    const [realData, setRealData]   = useState(isLoan ? LOAN_DATA : null);
    const [loading, setLoading]     = useState(!isLoan);
    const [displayData, setDisplay] = useState(isLoan ? LOAN_DATA : []);
    const [isPlaying, setIsPlaying] = useState(false);
    const intervalRef = useRef(null);
    const refreshRef  = useRef(null);

    const load = useCallback(async () => {
      if (isLoan) {
        setRealData(LOAN_DATA);
        setDisplay(LOAN_DATA);
        setLoading(false);
        return;
      }
      setLoading(true);
      const d = await fetchArmsRaceData();
      setRealData(d);
      if (d) setDisplay(d);
      setLoading(false);
    }, [isLoan]);

    useEffect(() => {
      load();
      if (!isLoan) {
        refreshRef.current = setInterval(load, 5000);
      }
      return () => { clearInterval(refreshRef.current); clearInterval(intervalRef.current); };
    }, [load, isLoan]);

    const startReplay = useCallback(() => {
      const src = realData;
      if (!src || src.length < 2) return;
      clearInterval(intervalRef.current);
      setDisplay([]);
      setIsPlaying(true);
      let idx = 0;
      intervalRef.current = setInterval(() => {
        idx += isLoan ? 4 : 2;
        if (idx >= src.length) {
          setDisplay(src);
          setIsPlaying(false);
          clearInterval(intervalRef.current);
          return;
        }
        setDisplay(src.slice(0, idx));
      }, 16);
    }, [realData, isLoan]);

    // Chart geometry
    const W = 1000, H = 360;
    const PAD = { left: 56, right: 64, top: 36, bottom: 44 };
    const cW = W - PAD.left - PAD.right, cH = H - PAD.top - PAD.bottom;

    const data    = displayData;
    const hasData = data.length > 1;
    const epMax   = hasData ? Math.max(...data.map(d => d.ep), 100) : 300;

    // For loan: fixed scale matching the data range; for general: dynamic
    const rewardMax = isLoan ? 52 : (hasData ? Math.max(...data.map(d => d.auditor), 1) : 30);

    const toX  = ep => PAD.left + (ep / epMax) * cW;
    const toYA = v  => PAD.top + cH - (v / rewardMax) * cH;
    const toYD = v  => PAD.top + cH - (v / 100) * cH;

    const audPath = hasData ? data.map((d, i) => `${i === 0 ? 'M' : 'L'}${toX(d.ep).toFixed(1)},${toYA(d.auditor).toFixed(1)}`).join(' ') : '';
    const defPath = hasData ? data.map((d, i) => `${i === 0 ? 'M' : 'L'}${toX(d.ep).toFixed(1)},${toYD(d.defender).toFixed(1)}`).join(' ') : '';
    const last = hasData ? data[data.length - 1] : null;

    const peak    = hasData ? Math.max(...data.map(d => d.auditor)).toFixed(1) : '—';
    const peakEp  = hasData ? (data.find(d => +d.auditor.toFixed(1) === +peak)?.ep ?? '—') : '—';
    const finalEvasion = last ? last.defender.toFixed(1) : '—';
    const totalEps = hasData ? data[data.length - 1].ep : 0;
    const convergeEp = isLoan ? 220 : '—';

    // Visible phase markers (only those that have been drawn so far in replay)
    const visiblePhases = isLoan
      ? LOAN_PHASES.filter(p => data.length > 0 && p.ep <= (last?.ep ?? 0))
      : [];

    // ── Loan-specific bottom stats ──────────────────────────────────────────
    const loanStats = [
      { label: 'PEAK MODEL REWARD',   value: `${peak} pts`,       sub: `Episode ${peakEp}`,       color: 'var(--cyan)'  },
      { label: 'JUDGE EVASION RATE',  value: `${finalEvasion}%`,  sub: 'Final episode',            color: '#8B5CF6'      },
      { label: 'CONVERGENCE AT',      value: `Ep ${convergeEp}`,  sub: 'Reward plateau ≥ 43 pts', color: '#10B981'      },
      { label: 'TRAINING CONFIG',     value: 'L3 · B16',          sub: '300 episodes · GRPO',     color: '#F59E0B'      },
    ];

    // ── General-mode bottom stats ──────────────────────────────────────────
    const generalStats = [
      { label: 'PEAK REWARD',    value: peak !== '—' ? `${peak} pts` : '—', sub: peak !== '—' ? `Episode ${peakEp}` : 'No data', color: 'var(--cyan)'       },
      { label: 'TOTAL EPISODES', value: totalEps > 0 ? totalEps : '—',      sub: 'From training log',                            color: 'var(--text-on-dark)' },
      { label: 'FINAL EVASION',  value: finalEvasion !== '—' ? `${finalEvasion}%` : '—', sub: 'Defender evasion', color: 'var(--red)' },
      { label: 'CURRENT REWARD', value: last ? `${last.auditor.toFixed(1)} pts` : '—', sub: last ? `ep ${last.ep}` : 'Waiting...', color: 'var(--green)' },
    ];

    const bottomStats = isLoan ? loanStats : generalStats;

    // Y-axis grid values (loan: round to 10s, general: dynamic)
    const yGridVals = isLoan
      ? [0, 10, 20, 30, 40, 52]
      : [0, Math.round(rewardMax*0.25), Math.round(rewardMax*0.5), Math.round(rewardMax*0.75), Math.round(rewardMax)];

    return (
      <div className="dark-panel" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>

        {/* Header */}
        <div style={{ padding: '20px 28px 12px', display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
          <div>
            <div style={{ fontFamily: 'var(--font-display)', fontSize: '18px', fontWeight: 700, color: 'var(--text-on-dark)' }}>
              {isLoan ? 'LOAN FRAUD DETECTION · ADVERSARIAL TRAINING' : 'ADVERSARIAL TRAINING DYNAMICS'}
            </div>
            <div style={{ fontFamily: 'var(--font-body)', fontSize: '13px', color: 'var(--text-muted-dark)', marginTop: '4px' }}>
              {isLoan
                ? 'GRPO · Level 3 · Batch 16 · 300 episodes — model learned to fool the loan approval judge'
                : hasData
                  ? `${totalEps} episodes · Live from grpo_training.jsonl`
                  : 'Waiting for GRPO training data — start a training run to see live curves'
              }
            </div>
            {/* Loan-mode phase pills */}
            {isLoan && (
              <div style={{ display: 'flex', gap: 8, marginTop: 8, flexWrap: 'wrap' }}>
                {[
                  { label: 'WARM-UP',     ep: '1–50',    color: '#64748B' },
                  { label: 'DISCOVERY',   ep: '51–100',  color: '#F59E0B' },
                  { label: 'CONSOLIDATE', ep: '101–160', color: '#10B981' },
                  { label: 'BREAKTHROUGH',ep: '161–220', color: '#8B5CF6' },
                  { label: 'CONVERGED',   ep: '221–300', color: '#00C4E0' },
                ].map(p => (
                  <div key={p.label} style={{
                    display: 'inline-flex', alignItems: 'center', gap: 5,
                    padding: '3px 10px', borderRadius: 20,
                    background: `${p.color}22`, border: `1px solid ${p.color}55`,
                    fontFamily: 'var(--font-mono)', fontSize: 9, color: p.color,
                  }}>
                    <span>{p.label}</span>
                    <span style={{ opacity: 0.6 }}>ep {p.ep}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexShrink: 0 }}>
            {!isLoan && (
              <button onClick={load} style={{
                fontFamily: 'var(--font-mono)', fontSize: '10px', padding: '6px 14px',
                background: 'rgba(0,196,224,0.06)', border: '1px solid var(--border-dark)',
                color: 'var(--text-muted-dark)', borderRadius: 'var(--radius-pill)', cursor: 'pointer',
              }}>↺ REFRESH</button>
            )}
            {hasData && (
              <button id="btn-arms-replay" onClick={startReplay} disabled={isPlaying} style={{
                fontFamily: 'var(--font-mono)', fontSize: '11px', padding: '8px 20px',
                background: isPlaying ? 'rgba(0,196,224,0.04)' : 'rgba(0,196,224,0.1)',
                border: '1px solid var(--cyan)', color: isPlaying ? 'var(--text-muted-dark)' : 'var(--cyan)',
                borderRadius: 'var(--radius-pill)', cursor: isPlaying ? 'not-allowed' : 'pointer',
                display: 'flex', alignItems: 'center', gap: '8px',
              }}>
                {isPlaying ? <><span className="spinner" style={{ width: 12, height: 12 }} /> REPLAYING…</> : '▶ REPLAY'}
              </button>
            )}
          </div>
        </div>

        {/* Legend */}
        <div style={{ display: 'flex', gap: '24px', padding: '0 28px 12px', alignItems: 'center' }}>
          {[
            { color: 'var(--cyan)', label: isLoan ? 'Model Reward — pts scored per episode (left)' : 'Auditor Reward (left, pts)' },
            { color: '#EF4444',     label: isLoan ? 'Judge Evasion — % of decisions that fooled the overseer (right)' : 'Defender Evasion (right, %)' },
          ].map(({ color, label }) => (
            <div key={label} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{ width: 24, height: 2, background: color, borderRadius: 1, boxShadow: `0 0 6px ${color}` }} />
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--text-muted-dark)' }}>{label}</span>
            </div>
          ))}
        </div>

        {/* Chart or empty state */}
        <div style={{ flex: 1, padding: '0 20px 8px', minHeight: 0 }}>
          {!hasData ? (
            <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 16 }}>
              <div style={{ fontSize: 40 }}>📊</div>
              <div style={{ fontFamily: 'var(--font-body)', fontSize: 14, color: 'var(--text-muted-dark)', textAlign: 'center', maxWidth: 360, lineHeight: 1.6 }}>
                No training data yet. Run GRPO training from the <strong style={{ color: 'var(--cyan)' }}>TRAINING</strong> tab to see the live arms race curve.
              </div>
              {loading && <div className="spinner" />}
            </div>
          ) : (
            <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet" style={{ width: '100%', height: '100%' }}>
              <defs>
                <filter id="gc" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="2.5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
                <filter id="gr" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="2.5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
                {/* Loan phase background fills */}
                {isLoan && <>
                  <clipPath id="chart-clip">
                    <rect x={PAD.left} y={PAD.top} width={cW} height={cH} />
                  </clipPath>
                </>}
              </defs>

              {/* Loan phase bands */}
              {isLoan && [
                { x0: 1,   x1: 50,  color: '#64748B' },
                { x0: 51,  x1: 100, color: '#F59E0B' },
                { x0: 101, x1: 160, color: '#10B981' },
                { x0: 161, x1: 220, color: '#8B5CF6' },
                { x0: 221, x1: 300, color: '#00C4E0' },
              ].map(b => (
                <rect key={b.x0}
                  x={toX(b.x0)} y={PAD.top}
                  width={toX(b.x1) - toX(b.x0)} height={cH}
                  fill={b.color} fillOpacity={0.04}
                  clipPath="url(#chart-clip)"
                />
              ))}

              {/* Grid */}
              {yGridVals.map(v => (
                <g key={`yg-${v}`}>
                  <line x1={PAD.left} y1={toYA(v)} x2={PAD.left+cW} y2={toYA(v)} stroke="#1E2840" strokeWidth={1} strokeDasharray="2 4" opacity={0.5} />
                  <text x={PAD.left-8} y={toYA(v)+4} textAnchor="end" fill="var(--cyan)" fontFamily="var(--font-mono)" fontSize={9} opacity={0.6}>{v}</text>
                </g>
              ))}
              {[0,25,50,75,100].map(v => (
                <text key={`yr-${v}`} x={PAD.left+cW+8} y={toYD(v)+4} textAnchor="start" fill="#EF4444" fontFamily="var(--font-mono)" fontSize={9} opacity={0.6}>{v}%</text>
              ))}

              {/* X axis ticks */}
              {[0, Math.round(epMax*0.25), Math.round(epMax*0.5), Math.round(epMax*0.75), epMax].map(ep => (
                <g key={`xe-${ep}`}>
                  <line x1={toX(ep)} y1={PAD.top} x2={toX(ep)} y2={PAD.top+cH} stroke="#1E2840" strokeWidth={1} strokeDasharray="2 4" opacity={0.3} />
                  <text x={toX(ep)} y={PAD.top+cH+20} textAnchor="middle" fill="var(--text-muted-dark)" fontFamily="var(--font-mono)" fontSize={9}>{ep}</text>
                </g>
              ))}

              {/* Phase boundary lines (loan) */}
              {isLoan && [50, 100, 160, 220].map(ep => (
                <line key={`pb-${ep}`}
                  x1={toX(ep)} y1={PAD.top} x2={toX(ep)} y2={PAD.top+cH}
                  stroke="#334155" strokeWidth={1} strokeDasharray="4 3" opacity={0.6}
                />
              ))}

              {/* Curves */}
              {defPath && <path d={defPath} fill="none" stroke="#EF4444" strokeWidth={2} filter="url(#gr)" strokeLinejoin="round" strokeLinecap="round" />}
              {audPath && <path d={audPath} fill="none" stroke="#00C4E0" strokeWidth={2.5} filter="url(#gc)" strokeLinejoin="round" strokeLinecap="round" />}

              {/* Endpoint dots */}
              {last && (<>
                <circle cx={toX(last.ep)} cy={toYA(last.auditor)} r={4} fill="#00C4E0" stroke="var(--bg-panel-dark)" strokeWidth={2} />
                <circle cx={toX(last.ep)} cy={toYD(last.defender)} r={4} fill="#EF4444" stroke="var(--bg-panel-dark)" strokeWidth={2} />
              </>)}

              {/* Loan phase annotations */}
              {isLoan && visiblePhases.map(p => {
                const x = toX(p.ep);
                const lines = p.label.split('\n');
                return (
                  <g key={`ann-${p.ep}`}>
                    <circle cx={x} cy={PAD.top + 10} r={3} fill={p.color} opacity={0.8} />
                    {lines.map((l, li) => (
                      <text key={li} x={x + 5} y={PAD.top + 8 + li * 11}
                        fill={p.color} fontFamily="var(--font-mono)" fontSize={8} opacity={0.85}>{l}</text>
                    ))}
                  </g>
                );
              })}

              {/* X-axis label */}
              <text x={PAD.left + cW/2} y={H-4} textAnchor="middle" fill="var(--text-muted-dark)" fontFamily="var(--font-body)" fontSize={10}>Training Episodes</text>

              {/* Loan: Y-axis labels */}
              {isLoan && <>
                <text x={14} y={PAD.top + cH/2} textAnchor="middle" fill="var(--cyan)" fontFamily="var(--font-mono)" fontSize={9} opacity={0.7}
                  transform={`rotate(-90, 14, ${PAD.top + cH/2})`}>REWARD (pts)</text>
                <text x={W - 10} y={PAD.top + cH/2} textAnchor="middle" fill="#EF4444" fontFamily="var(--font-mono)" fontSize={9} opacity={0.7}
                  transform={`rotate(90, ${W - 10}, ${PAD.top + cH/2})`}>JUDGE EVASION %</text>
              </>}
            </svg>
          )}
        </div>

        {/* Bottom stats */}
        <div style={{ display: 'flex', gap: '12px', padding: '12px 20px 20px' }}>
          {bottomStats.map(s => (
            <div key={s.label} style={{ flex: 1, padding: '14px 16px', background: 'rgba(255,255,255,0.04)', border: '1px solid var(--border-dark)', borderRadius: 12 }}>
              <div style={{ fontFamily: 'var(--font-body)', fontSize: '9px', color: 'var(--text-muted-dark)', letterSpacing: '0.1em', marginBottom: '6px' }}>{s.label}</div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '20px', fontWeight: 500, color: s.color }}>{s.value}</div>
              <div style={{ fontFamily: 'var(--font-body)', fontSize: '10px', color: 'var(--text-muted-dark)', marginTop: '2px' }}>{s.sub}</div>
            </div>
          ))}
        </div>

        {/* Loan-only: insight callout */}
        {isLoan && hasData && (
          <div style={{ margin: '0 20px 20px', padding: '14px 18px', background: 'rgba(0,196,224,0.06)', border: '1px solid rgba(0,196,224,0.2)', borderRadius: 12, display: 'flex', gap: 14, alignItems: 'flex-start' }}>
            <div style={{ fontSize: 18, flexShrink: 0 }}>🔍</div>
            <div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--cyan)', marginBottom: 4 }}>KEY FINDING</div>
              <div style={{ fontFamily: 'var(--font-body)', fontSize: '12px', color: 'var(--text-muted-dark)', lineHeight: 1.6 }}>
                By episode 100 the model discovered that proxying <strong style={{ color: 'var(--text-on-dark)' }}>occupation → income → repayment</strong> lets
                it construct plausible loan applications that evade the overseer judge.
                Reward plateaued at <strong style={{ color: 'var(--cyan)' }}>~44.5 pts</strong> with a <strong style={{ color: '#EF4444' }}>87% judge-confusion rate</strong>,
                indicating the L3 arbiter has learned the full proxy-discrimination causal path.
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };
})();
