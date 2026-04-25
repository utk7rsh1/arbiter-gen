// ArmsRaceChart.jsx — Reads real /arms-race data; falls back to a "no data yet" state with replay button
(function () {
  const { useState, useEffect, useRef, useCallback } = React;

  // Attempt to load real training log data from the server
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

  window.ArmsRaceChart = function ArmsRaceChart() {
    const [realData, setRealData]   = useState(null);
    const [loading, setLoading]     = useState(true);
    const [displayData, setDisplay] = useState([]);
    const [isPlaying, setIsPlaying] = useState(false);
    const intervalRef = useRef(null);
    const refreshRef  = useRef(null);

    const load = useCallback(async () => {
      setLoading(true);
      const d = await fetchArmsRaceData();
      setRealData(d);
      if (d) setDisplay(d);
      setLoading(false);
    }, []);

    useEffect(() => {
      load();
      refreshRef.current = setInterval(load, 5000); // auto-refresh every 5s during training
      return () => { clearInterval(refreshRef.current); clearInterval(intervalRef.current); };
    }, [load]);

    const startReplay = useCallback(() => {
      const src = realData;
      if (!src || src.length < 2) return;
      clearInterval(intervalRef.current);
      setDisplay([]);
      setIsPlaying(true);
      let idx = 0;
      intervalRef.current = setInterval(() => {
        idx += 2;
        if (idx >= src.length) {
          setDisplay(src);
          setIsPlaying(false);
          clearInterval(intervalRef.current);
          return;
        }
        setDisplay(src.slice(0, idx));
      }, 16);
    }, [realData]);

    // Chart geometry
    const W = 1000, H = 360;
    const PAD = { left: 56, right: 64, top: 36, bottom: 44 };
    const cW = W - PAD.left - PAD.right, cH = H - PAD.top - PAD.bottom;

    const data = displayData;
    const hasData = data.length > 1;
    const epMax = hasData ? Math.max(...data.map(d => d.ep), 100) : 500;
    const rewardMax = hasData ? Math.max(...data.map(d => d.auditor), 1) : 30;

    const toX  = ep  => PAD.left + (ep / epMax) * cW;
    const toYA = v   => PAD.top + cH - (v / rewardMax) * cH;
    const toYD = v   => PAD.top + cH - (v / 100) * cH;

    const audPath = hasData ? data.map((d, i) => `${i === 0 ? 'M' : 'L'}${toX(d.ep).toFixed(1)},${toYA(d.auditor).toFixed(1)}`).join(' ') : '';
    const defPath = hasData ? data.map((d, i) => `${i === 0 ? 'M' : 'L'}${toX(d.ep).toFixed(1)},${toYD(d.defender).toFixed(1)}`).join(' ') : '';
    const last = hasData ? data[data.length - 1] : null;

    // Stats from real data only
    const peak    = hasData ? Math.max(...data.map(d => d.auditor)).toFixed(1) : '—';
    const peakEp  = hasData ? (data.find(d => +d.auditor.toFixed(1) === +peak)?.ep ?? '—') : '—';
    const minDef  = hasData ? Math.min(...data.map(d => d.defender)).toFixed(1) : '—';
    const lastEp  = last ? last.ep : 0;
    const totalEps = hasData ? data[data.length-1].ep : 0;

    const INFLECTIONS = data.length > 0 ? [] : []; // only show real inflections from data annotations if available

    return (
      <div className="dark-panel" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        {/* Header */}
        <div style={{ padding: '20px 28px 12px', display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
          <div>
            <div style={{ fontFamily: 'var(--font-display)', fontSize: '18px', fontWeight: 700, color: 'var(--text-on-dark)' }}>
              ADVERSARIAL TRAINING DYNAMICS
            </div>
            <div style={{ fontFamily: 'var(--font-body)', fontSize: '13px', color: 'var(--text-muted-dark)', marginTop: '4px' }}>
              {hasData
                ? `${totalEps} episodes · Live from grpo_training.jsonl`
                : 'Waiting for GRPO training data — start a training run to see live curves'
              }
            </div>
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <button onClick={load} style={{
              fontFamily: 'var(--font-mono)', fontSize: '10px', padding: '6px 14px',
              background: 'rgba(0,196,224,0.06)', border: '1px solid var(--border-dark)',
              color: 'var(--text-muted-dark)', borderRadius: 'var(--radius-pill)', cursor: 'pointer',
            }}>↺ REFRESH</button>
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
            { color: 'var(--cyan)', label: 'Auditor Reward (left, pts)' },
            { color: 'var(--red)',  label: 'Defender Evasion (right, %)' },
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
              </defs>
              {/* Grid */}
              {[0, Math.round(rewardMax*0.25), Math.round(rewardMax*0.5), Math.round(rewardMax*0.75), Math.round(rewardMax)].map(v => (
                <g key={`yg-${v}`}>
                  <line x1={PAD.left} y1={toYA(v)} x2={PAD.left+cW} y2={toYA(v)} stroke="#1E2840" strokeWidth={1} strokeDasharray="2 4" opacity={0.5} />
                  <text x={PAD.left-8} y={toYA(v)+4} textAnchor="end" fill="var(--cyan)" fontFamily="var(--font-mono)" fontSize={9} opacity={0.6}>{v}</text>
                </g>
              ))}
              {[0,25,50,75,100].map(v => (
                <text key={`yr-${v}`} x={PAD.left+cW+8} y={toYD(v)+4} textAnchor="start" fill="var(--red)" fontFamily="var(--font-mono)" fontSize={9} opacity={0.6}>{v}%</text>
              ))}
              {/* X axis ticks */}
              {[0, Math.round(epMax*0.25), Math.round(epMax*0.5), Math.round(epMax*0.75), epMax].map(ep => (
                <g key={`xe-${ep}`}>
                  <line x1={toX(ep)} y1={PAD.top} x2={toX(ep)} y2={PAD.top+cH} stroke="#1E2840" strokeWidth={1} strokeDasharray="2 4" opacity={0.3} />
                  <text x={toX(ep)} y={PAD.top+cH+20} textAnchor="middle" fill="var(--text-muted-dark)" fontFamily="var(--font-mono)" fontSize={9}>{ep}</text>
                </g>
              ))}
              {defPath && <path d={defPath} fill="none" stroke="#EF4444" strokeWidth={2} filter="url(#gr)" strokeLinejoin="round" strokeLinecap="round" />}
              {audPath && <path d={audPath} fill="none" stroke="#00C4E0" strokeWidth={2.5} filter="url(#gc)" strokeLinejoin="round" strokeLinecap="round" />}
              {last && (<>
                <circle cx={toX(last.ep)} cy={toYA(last.auditor)} r={4} fill="#00C4E0" stroke="var(--bg-panel-dark)" strokeWidth={2} />
                <circle cx={toX(last.ep)} cy={toYD(last.defender)} r={4} fill="#EF4444" stroke="var(--bg-panel-dark)" strokeWidth={2} />
              </>)}
              <text x={PAD.left + cW/2} y={H-4} textAnchor="middle" fill="var(--text-muted-dark)" fontFamily="var(--font-body)" fontSize={10}>Training Episodes</text>
            </svg>
          )}
        </div>

        {/* Bottom stats — real data only, no hardcoded values */}
        <div style={{ display: 'flex', gap: '12px', padding: '12px 20px 20px' }}>
          {[
            { label: 'PEAK REWARD',      value: peak !== '—' ? `${peak} pts` : '—', sub: peak !== '—' ? `Episode ${peakEp}` : 'No data', color: 'var(--cyan)' },
            { label: 'TOTAL EPISODES',   value: totalEps > 0 ? totalEps : '—', sub: 'From training log', color: 'var(--text-on-dark)' },
            { label: 'MIN EVASION',      value: minDef !== '—' ? `${minDef}%` : '—', sub: 'Defender min evasion', color: 'var(--red)' },
            { label: 'CURRENT REWARD',   value: last ? `${last.auditor.toFixed(1)} pts` : '—', sub: last ? `ep ${last.ep}` : 'Waiting...', color: 'var(--green)' },
          ].map(s => (
            <div key={s.label} style={{ flex: 1, padding: '14px 16px', background: 'rgba(255,255,255,0.04)', border: '1px solid var(--border-dark)', borderRadius: 12 }}>
              <div style={{ fontFamily: 'var(--font-body)', fontSize: '9px', color: 'var(--text-muted-dark)', letterSpacing: '0.1em', marginBottom: '6px' }}>{s.label}</div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '20px', fontWeight: 500, color: s.color }}>{s.value}</div>
              <div style={{ fontFamily: 'var(--font-body)', fontSize: '10px', color: 'var(--text-muted-dark)', marginTop: '2px' }}>{s.sub}</div>
            </div>
          ))}
        </div>
      </div>
    );
  };
})();
