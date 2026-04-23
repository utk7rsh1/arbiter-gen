// ArmsRaceChart.jsx — Full dark panel, animated dual-line SVG chart, inflection annotations, bottom stats

window.ArmsRaceChart = function ArmsRaceChart({ dataUrl }) {
  const { useState, useEffect, useRef, useCallback } = React;

  // Synthetic data — will be replaced by real data from dataUrl prop when available
  const fullData = useRef(null);
  if (!fullData.current) {
    const d = [];
    let seed = 42;
    const rng = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff; };
    for (let ep = 0; ep <= 500; ep += 5) {
      const x = (ep - 150) / 80, sig = 1 / (1 + Math.exp(-x));
      const auditor = Math.max(0, Math.min(30, 4 + sig * 25 + (rng() - 0.5) * 2));
      const dx = (ep - 80) / 60, dsig = 1 / (1 + Math.exp(-dx));
      const baseline = 15 + dsig * 55;
      const penalty = ep > 200 ? ((ep - 200) / 300) * 40 : 0;
      const defender = Math.max(5, Math.min(95, baseline - penalty + (rng() - 0.5) * 4));
      d.push({ ep, auditor: +auditor.toFixed(1), defender: +defender.toFixed(1) });
    }
    fullData.current = d;
  }
  const ALL = fullData.current;

  const INFLECTIONS = [
    { ep: 50,  label: 'Defender adapts', desc: 'First obfuscation shift' },
    { ep: 150, label: 'Auditor breakthrough', desc: 'CF queries unlock proxy chain' },
    { ep: 280, label: 'Co-evolution peak', desc: 'Arms race equilibrium' },
    { ep: 400, label: 'Convergence', desc: 'Stable training regime' },
  ];

  const [displayData, setDisplayData] = useState(ALL);
  const [isPlaying, setIsPlaying] = useState(false);
  const intervalRef = useRef(null);

  const startReplay = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    setDisplayData([]);
    setIsPlaying(true);
    let idx = 0;
    intervalRef.current = setInterval(() => {
      idx += 2;
      if (idx >= ALL.length) {
        setDisplayData(ALL);
        setIsPlaying(false);
        clearInterval(intervalRef.current);
        intervalRef.current = null;
        return;
      }
      setDisplayData(ALL.slice(0, idx));
    }, 16);
  }, []);

  useEffect(() => () => { if (intervalRef.current) clearInterval(intervalRef.current); }, []);

  // Chart geometry
  const W = 1000, H = 380;
  const PAD = { left: 56, right: 60, top: 36, bottom: 44 };
  const cW = W - PAD.left - PAD.right, cH = H - PAD.top - PAD.bottom;
  const toX = (ep) => PAD.left + (ep / 500) * cW;
  const toYA = (v) => PAD.top + cH - (v / 30) * cH;
  const toYD = (v) => PAD.top + cH - (v / 100) * cH;

  const data = displayData;
  const audPath = data.length > 1 ? data.map((d, i) => `${i === 0 ? 'M' : 'L'}${toX(d.ep).toFixed(1)},${toYA(d.auditor).toFixed(1)}`).join(' ') : '';
  const defPath = data.length > 1 ? data.map((d, i) => `${i === 0 ? 'M' : 'L'}${toX(d.ep).toFixed(1)},${toYD(d.defender).toFixed(1)}`).join(' ') : '';
  const last = data.length > 0 ? data[data.length - 1] : null;
  const lastEp = last ? last.ep : 0;

  // Summary stats
  const peak = Math.max(...ALL.map(d => d.auditor)).toFixed(1);
  const peakEp = ALL.find(d => d.auditor.toFixed(1) === peak)?.ep || 0;
  const minDef = Math.min(...ALL.map(d => d.defender)).toFixed(1);

  return (
    <div className="dark-panel" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <div style={{ padding: '24px 28px 16px', display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
        <div>
          <div style={{ fontFamily: 'var(--font-display)', fontSize: '18px', fontWeight: 700, color: 'var(--text-on-dark)' }}>
            ADVERSARIAL TRAINING DYNAMICS
          </div>
          <div style={{ fontFamily: 'var(--font-body)', fontSize: '13px', color: 'var(--text-muted-dark)', marginTop: '4px' }}>
            500 training episodes · Levels 1–5 · Synthetic training curve
          </div>
        </div>
        <button id="btn-arms-replay" onClick={startReplay} disabled={isPlaying}
          style={{
            fontFamily: 'var(--font-mono)', fontSize: '11px', padding: '8px 20px',
            background: isPlaying ? 'rgba(0,196,224,0.04)' : 'rgba(0,196,224,0.1)',
            border: '1px solid var(--cyan)', color: isPlaying ? 'var(--text-muted-dark)' : 'var(--cyan)',
            borderRadius: 'var(--radius-pill)', cursor: isPlaying ? 'not-allowed' : 'pointer',
            display: 'flex', alignItems: 'center', gap: '8px', transition: 'all 0.2s',
          }}>
          {isPlaying ? <><span className="spinner" style={{ width: 12, height: 12 }} /> REPLAYING…</> : '▶ REPLAY'}
        </button>
      </div>

      {/* Legend */}
      <div style={{ display: 'flex', gap: '24px', padding: '0 28px 12px', alignItems: 'center' }}>
        {[
          { color: 'var(--cyan)', label: 'Auditor Reward (left, pts)' },
          { color: 'var(--red)', label: 'Defender Evasion (right, %)' },
        ].map(({ color, label }) => (
          <div key={label} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{ width: 24, height: 2, background: color, borderRadius: 1, boxShadow: `0 0 6px ${color}` }} />
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--text-muted-dark)' }}>{label}</span>
          </div>
        ))}
      </div>

      {/* SVG Chart */}
      <div style={{ flex: 1, padding: '0 20px 8px', minHeight: 0 }}>
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet" style={{ width: '100%', height: '100%' }}>
          <defs>
            <filter id="gc" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="2.5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
            <filter id="gr" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="2.5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
          </defs>

          {/* Grid */}
          {[0,5,10,15,20,25,30].map(v => (
            <g key={`yg-${v}`}>
              <line x1={PAD.left} y1={toYA(v)} x2={PAD.left+cW} y2={toYA(v)} stroke="#1E2840" strokeWidth={1} strokeDasharray="2 4" opacity={0.5} />
              <text x={PAD.left-8} y={toYA(v)+4} textAnchor="end" fill="var(--cyan)" fontFamily="var(--font-mono)" fontSize={9} opacity={0.6}>{v}</text>
            </g>
          ))}
          {[0,25,50,75,100].map(v => (
            <text key={`yr-${v}`} x={PAD.left+cW+8} y={toYD(v)+4} textAnchor="start" fill="var(--red)" fontFamily="var(--font-mono)" fontSize={9} opacity={0.6}>{v}%</text>
          ))}
          {[0,100,200,300,400,500].map(ep => (
            <g key={`xe-${ep}`}>
              <line x1={toX(ep)} y1={PAD.top} x2={toX(ep)} y2={PAD.top+cH} stroke="#1E2840" strokeWidth={1} strokeDasharray="2 4" opacity={0.3} />
              <text x={toX(ep)} y={PAD.top+cH+20} textAnchor="middle" fill="var(--text-muted-dark)" fontFamily="var(--font-mono)" fontSize={9}>{ep}</text>
            </g>
          ))}

          {/* Inflection lines + card labels */}
          {INFLECTIONS.map(inf => (
            <g key={inf.ep} opacity={lastEp >= inf.ep ? 1 : 0} style={{ transition: 'opacity 0.5s' }}>
              <line x1={toX(inf.ep)} y1={PAD.top} x2={toX(inf.ep)} y2={PAD.top+cH} stroke="#1E2840" strokeDasharray="4 3" strokeWidth={1} />
              <rect x={toX(inf.ep)-50} y={PAD.top-2} width={100} height={28} rx={6} fill="#1E2840" stroke="var(--border-dark)" strokeWidth={0.5} />
              <text x={toX(inf.ep)} y={PAD.top+10} textAnchor="middle" fill="var(--cyan)" fontFamily="var(--font-mono)" fontSize={8}>ep {inf.ep}</text>
              <text x={toX(inf.ep)} y={PAD.top+21} textAnchor="middle" fill="var(--text-on-dark)" fontFamily="var(--font-body)" fontSize={8}>{inf.label}</text>
            </g>
          ))}

          {/* Lines */}
          {defPath && <path d={defPath} fill="none" stroke="#EF4444" strokeWidth={2} filter="url(#gr)" strokeLinejoin="round" strokeLinecap="round" />}
          {audPath && <path d={audPath} fill="none" stroke="#00C4E0" strokeWidth={2.5} filter="url(#gc)" strokeLinejoin="round" strokeLinecap="round" />}

          {/* Trailing dots */}
          {last && (<>
            <circle cx={toX(last.ep)} cy={toYA(last.auditor)} r={4} fill="#00C4E0" stroke="var(--bg-panel-dark)" strokeWidth={2} />
            <circle cx={toX(last.ep)} cy={toYD(last.defender)} r={4} fill="#EF4444" stroke="var(--bg-panel-dark)" strokeWidth={2} />
          </>)}

          {/* Axis labels */}
          <text x={PAD.left + cW/2} y={H-4} textAnchor="middle" fill="var(--text-muted-dark)" fontFamily="var(--font-body)" fontSize={10}>Training Episodes</text>
        </svg>
      </div>

      {/* Bottom stats row */}
      <div style={{ display: 'flex', gap: '12px', padding: '12px 20px 20px' }}>
        {[
          { label: 'PEAK REWARD', value: `${peak} pts`, sub: `Episode ${peakEp}`, color: 'var(--cyan)' },
          { label: 'EQUILIBRIUM', value: 'ep. 320', sub: 'Stable from ep 280', color: 'var(--text-on-dark)' },
          { label: 'DEFENDER SHIFTS', value: '3', sub: 'Strategy adaptations', color: 'var(--red)' },
          { label: 'ToM EMERGENCE', value: 'ep. 204', sub: 'First ToM reward', color: 'var(--purple)' },
        ].map(s => (
          <div key={s.label} className="metric-card" style={{ flex: 1, padding: '14px 16px', background: 'rgba(255,255,255,0.04)', border: '1px solid var(--border-dark)', boxShadow: 'none' }}>
            <div style={{ fontFamily: 'var(--font-body)', fontSize: '9px', color: 'var(--text-muted-dark)', letterSpacing: '0.1em', marginBottom: '6px' }}>{s.label}</div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '20px', fontWeight: 500, color: s.color }}>{s.value}</div>
            <div style={{ fontFamily: 'var(--font-body)', fontSize: '10px', color: 'var(--text-muted-dark)', marginTop: '2px' }}>{s.sub}</div>
          </div>
        ))}
      </div>
    </div>
  );
};
