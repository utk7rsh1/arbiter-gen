// BottomMetrics.jsx — Three metric cards: Episode Progress (sparkline), Claim Accuracy (mini bars), Budget Gauge (SVG arc)

window.BottomMetrics = function BottomMetrics({ step, maxSteps, claims, reward, budget, maxBudget, level, episodeHistory }) {
  const { useMemo } = React;

  // Compute claim accuracy from props (forward-compatible: backend will supply real data)
  const claimStats = useMemo(() => {
    const all = claims || [];
    const byType = { causal: { total: 0, correct: 0 }, counterfactual: { total: 0, correct: 0 }, tom: { total: 0, correct: 0 } };
    all.forEach(c => {
      const t = (c.claim_type || '').toLowerCase();
      const key = t.includes('counterfactual') ? 'counterfactual' : t.includes('theory') || t.includes('tom') ? 'tom' : 'causal';
      byType[key].total++;
      if (c.correct) byType[key].correct++;
    });
    const totalCorrect = all.filter(c => c.correct).length;
    const totalAll = all.length;
    return { overall: totalAll > 0 ? Math.round((totalCorrect / totalAll) * 100) : 0, byType };
  }, [claims]);

  // Budget gauge
  const budgetVal = budget ?? (maxBudget || 20) - step;
  const budgetMax = maxBudget || 20;
  const budgetPct = Math.max(0, Math.min(1, budgetVal / budgetMax));

  // Gauge SVG arc
  const renderGauge = () => {
    const r = 38, cx = 50, cy = 50;
    const startAngle = -135, endAngle = 135;
    const totalArc = endAngle - startAngle; // 270 degrees
    const sweepAngle = totalArc * budgetPct;
    const currentAngle = startAngle + sweepAngle;

    const toRad = (deg) => (deg * Math.PI) / 180;
    const startX = cx + r * Math.cos(toRad(startAngle));
    const startY = cy + r * Math.sin(toRad(startAngle));
    const endX = cx + r * Math.cos(toRad(currentAngle));
    const endY = cy + r * Math.sin(toRad(currentAngle));
    const largeArc = sweepAngle > 180 ? 1 : 0;

    // Color based on budget
    const gaugeColor = budgetPct > 0.6 ? 'var(--green)' : budgetPct > 0.3 ? 'var(--amber)' : 'var(--red)';

    return (
      <svg viewBox="0 0 100 100" width="100" height="100">
        {/* Track */}
        <path
          d={`M ${cx + r * Math.cos(toRad(startAngle))},${cy + r * Math.sin(toRad(startAngle))} A ${r},${r} 0 1,1 ${cx + r * Math.cos(toRad(endAngle))},${cy + r * Math.sin(toRad(endAngle))}`}
          fill="none" stroke="var(--border-light)" strokeWidth="6" strokeLinecap="round"
        />
        {/* Value arc */}
        {budgetPct > 0.01 && (
          <path
            d={`M ${startX},${startY} A ${r},${r} 0 ${largeArc},1 ${endX},${endY}`}
            fill="none" stroke={gaugeColor} strokeWidth="6" strokeLinecap="round"
            style={{ transition: 'all 0.3s ease' }}
          />
        )}
        {/* Center number */}
        <text x={cx} y={cy - 2} textAnchor="middle" dominantBaseline="central"
          style={{ fontFamily: 'var(--font-mono)', fontSize: '20px', fontWeight: 500, fill: 'var(--text-primary)' }}>
          {Math.max(0, Math.round(budgetVal))}
        </text>
        <text x={cx} y={cy + 14} textAnchor="middle"
          style={{ fontFamily: 'var(--font-body)', fontSize: '7px', fill: 'var(--text-muted)', letterSpacing: '0.1em' }}>
          REMAINING
        </text>
      </svg>
    );
  };

  // Mini sparkline from episode history
  const renderSparkline = () => {
    const history = episodeHistory || [];
    if (history.length < 2) return null;
    const h = 40, w = 120;
    const maxVal = Math.max(...history.map(d => d.reward), 1);
    const points = history.map((d, i) => {
      const x = (i / (history.length - 1)) * w;
      const y = h - (d.reward / maxVal) * (h - 4);
      return `${x},${y}`;
    }).join(' ');
    const areaPath = `M0,${h} L${points.split(' ').map((p, i) => (i === 0 ? p : ` L${p}`)).join('')} L${w},${h} Z`;

    return (
      <svg width={w} height={h} style={{ marginTop: '8px' }}>
        <polyline points={points} fill="none" stroke="var(--cyan)" strokeWidth="1.5" />
        <polygon points={`0,${h} ${points} ${w},${h}`} fill="rgba(0,196,224,0.1)" />
      </svg>
    );
  };

  const accBar = (label, pct, color, delay) => (
    <div style={{ marginBottom: '6px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '2px' }}>
        <span style={{ fontFamily: 'var(--font-body)', fontSize: '10px', color: 'var(--text-muted)', letterSpacing: '0.06em' }}>{label}</span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--text-primary)' }}>{pct}%</span>
      </div>
      <div style={{ height: '4px', background: 'var(--border-light)', borderRadius: 'var(--radius-pill)' }}>
        <div style={{
          height: '100%', width: `${pct}%`, background: color,
          borderRadius: 'var(--radius-pill)', transition: `width 0.6s ease ${delay}s`,
        }} />
      </div>
    </div>
  );

  return (
    <div style={{ display: 'flex', gap: '16px' }}>
      {/* Card 1: Episode Progress */}
      <div className="metric-card" style={{ flex: 1 }}>
        <div className="metric-label" style={{ marginBottom: '8px' }}>EPISODE PROGRESS</div>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '4px' }}>
          <span className="metric-number" style={{ fontSize: '32px' }}>{step}</span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '16px', color: 'var(--text-muted)' }}>/ {maxSteps}</span>
        </div>
        {renderSparkline()}
        <div style={{
          marginTop: '8px', display: 'inline-flex', alignItems: 'center', gap: '6px',
          fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--cyan)',
        }}>
          <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--cyan)', display: 'inline-block' }} />
          {level || 'L1'} ACTIVE
        </div>
      </div>

      {/* Card 2: Claim Accuracy */}
      <div className="metric-card" style={{ flex: 1 }}>
        <div className="metric-label" style={{ marginBottom: '8px' }}>CLAIM ACCURACY</div>
        <div className="metric-number" style={{ fontSize: '32px', marginBottom: '12px' }}>
          {claimStats.overall}<span style={{ fontSize: '18px', color: 'var(--text-muted)' }}>%</span>
        </div>
        {accBar('CAUSAL', claimStats.byType.causal.total > 0 ? Math.round((claimStats.byType.causal.correct / claimStats.byType.causal.total) * 100) : 0, 'var(--cyan)', 0)}
        {accBar('CF', claimStats.byType.counterfactual.total > 0 ? Math.round((claimStats.byType.counterfactual.correct / claimStats.byType.counterfactual.total) * 100) : 0, 'var(--purple)', 0.1)}
        {accBar('TOM', claimStats.byType.tom.total > 0 ? Math.round((claimStats.byType.tom.correct / claimStats.byType.tom.total) * 100) : 0, '#7C3AED', 0.2)}
      </div>

      {/* Card 3: Budget Remaining */}
      <div className="metric-card" style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <div className="metric-label" style={{ marginBottom: '8px', alignSelf: 'flex-start' }}>BUDGET</div>
        {renderGauge()}
      </div>
    </div>
  );
};
