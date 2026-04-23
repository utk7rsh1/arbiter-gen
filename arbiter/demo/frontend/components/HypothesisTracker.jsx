// HypothesisTracker.jsx — Three hypothesis status cards in a horizontal strip (white cards)

window.HypothesisTracker = function HypothesisTracker({ hypotheses }) {
  const getCard = (key, hyp) => {
    const status = (hyp.status || 'ACTIVE').toUpperCase();
    const isActive     = status === 'ACTIVE';
    const isWeakened   = status === 'WEAKENED';
    const isEliminated = status === 'ELIMINATED';
    const isConfirmed  = status === 'CONFIRMED';

    const dotColor = isConfirmed ? 'var(--green)' : isActive ? 'var(--amber)' : isWeakened ? 'var(--orange)' : 'var(--text-muted)';
    const statusLabel = isConfirmed ? 'Confirmed' : isActive ? 'Active' : isWeakened ? 'Weakened' : 'Eliminated';

    return (
      <div
        key={key}
        className={`hyp-card ${isActive ? 'active' : isWeakened ? 'weakened' : isEliminated ? 'eliminated' : 'confirmed'}`}
        style={{ flex: 1 }}
      >
        <div style={{
          fontFamily: 'var(--font-body)', fontSize: '10px',
          color: 'var(--text-muted)', letterSpacing: '0.1em',
          marginBottom: '6px', fontWeight: 500,
        }}>{key.toUpperCase()}</div>

        <div style={{
          fontFamily: 'var(--font-body)', fontSize: '13px',
          color: 'var(--text-primary)', fontWeight: 600,
          marginBottom: '10px',
          textDecoration: isEliminated ? 'line-through' : 'none',
        }}>{hyp.label || hyp.type || key}</div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{
            width: 8, height: 8, borderRadius: '50%',
            background: dotColor,
            boxShadow: isActive ? `0 0 8px ${dotColor}` : 'none',
            animation: isActive ? 'pulse-dot 2s ease-in-out infinite' : 'none',
            display: 'inline-block',
          }} />
          <span style={{
            fontFamily: 'var(--font-body)', fontSize: '11px',
            color: dotColor, fontWeight: isActive || isConfirmed ? 600 : 400,
          }}>{statusLabel}</span>
        </div>
      </div>
    );
  };

  return (
    <div style={{ display: 'flex', gap: '12px' }}>
      {Object.entries(hypotheses || {}).map(([key, hyp]) => getCard(key, hyp))}
    </div>
  );
};
