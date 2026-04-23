// RewardPanel.jsx — White card: large total score (left) + component bars (right) with delta animations

window.RewardPanel = function RewardPanel({ reward, rewardDeltas, episodeReward }) {
  const { useRef } = React;

  const components = [
    { key: 'claim',          label: 'Claim Reward',   max: 12, color: 'var(--cyan)'   },
    { key: 'counterfactual', label: 'Counterfactual', max: 6,  color: 'var(--purple)'  },
    { key: 'tom',            label: 'Theory of Mind', max: 3,  color: 'var(--purple)'  },
    { key: 'chain',          label: 'Chain Bonus',    max: 8,  color: 'var(--green)'   },
    { key: 'consistency',    label: 'Consistency',    max: 3,  color: 'var(--red)'     },
    { key: 'budget',         label: 'Budget Eff.',    max: 4,  color: 'var(--cyan)'    },
  ];

  const total = reward?.total ?? 0;
  const totalMax = 35;

  const renderBar = (comp) => {
    const val = reward?.[comp.key] ?? 0;
    const max = comp.max || 1;
    const pct = max > 0 ? Math.min(100, Math.abs(val / max) * 100) : 0;
    const isNegative = val < 0;

    return (
      <div key={comp.key} style={{ marginBottom: '12px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
          <span style={{ fontFamily: 'var(--font-body)', fontSize: '11px', color: 'var(--text-secondary)' }}>{comp.label}</span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: isNegative ? 'var(--red)' : 'var(--text-primary)' }}>
            {isNegative ? '' : '+'}{val.toFixed(1)}
            <span style={{ color: 'var(--text-muted)' }}> / {max.toFixed(1)}</span>
          </span>
        </div>
        <div className="reward-bar-track" style={{ position: 'relative' }}>
          {isNegative ? (
            <div style={{
              position: 'absolute', right: 0, top: 0, height: '100%',
              width: `${pct}%`, background: 'var(--red)',
              borderRadius: 'var(--radius-pill)', transition: 'width 0.6s ease',
            }} />
          ) : (
            <div className="reward-bar-fill" style={{
              width: `${pct}%`, background: comp.color,
            }} />
          )}
          {/* Delta popup */}
          {rewardDeltas?.filter(d => d.component === comp.key).map(d => (
            <div key={d.id} className="delta-popup" style={{
              left: `${pct}%`, top: '-20px',
              color: d.value > 0 ? 'var(--green)' : 'var(--red)',
            }}>
              {d.value > 0 ? '+' : ''}{d.value.toFixed(1)}
            </div>
          ))}
        </div>
      </div>
    );
  };

  const isVerdictCorrect = episodeReward?.terminal?.verdict_correct;
  const scoreColor = episodeReward
    ? (isVerdictCorrect ? 'var(--green)' : 'var(--red)')
    : (total > 15 ? 'var(--cyan)' : 'var(--text-primary)');
  const scoreGlow = total > 10 ? `0 0 24px var(--cyan-glow)` : 'none';

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div className="card-header">
        <span className="card-header-label">REWARD BREAKDOWN</span>
      </div>

      <div className="card-body" style={{ flex: 1, overflowY: 'auto', display: 'flex', gap: '24px' }}>
        {/* Left: Total score */}
        <div style={{ flex: '0 0 120px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', position: 'relative' }}>
          <div style={{ fontFamily: 'var(--font-body)', fontSize: '10px', color: 'var(--text-muted)', letterSpacing: '0.12em', marginBottom: '8px' }}>TOTAL SCORE</div>
          <div style={{
            fontFamily: 'var(--font-mono)', fontSize: '48px', fontWeight: 500,
            color: scoreColor, lineHeight: 1, textShadow: scoreGlow,
          }}>
            {total.toFixed(1)}
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '14px', color: 'var(--text-muted)', marginTop: '4px' }}>
            / {totalMax}.0 MAX
          </div>

          {episodeReward && (
            <div style={{
              marginTop: '12px', padding: '4px 14px',
              borderRadius: 'var(--radius-pill)',
              background: isVerdictCorrect ? 'rgba(16,185,129,0.08)' : 'rgba(239,68,68,0.08)',
              border: `1px solid ${isVerdictCorrect ? 'var(--green)' : 'var(--red)'}`,
              fontFamily: 'var(--font-mono)', fontSize: '10px',
              color: isVerdictCorrect ? 'var(--green)' : 'var(--red)',
              letterSpacing: '0.08em',
            }}>
              {isVerdictCorrect ? '✓ VERDICT CORRECT' : '✗ VERDICT WRONG'}
            </div>
          )}
        </div>

        {/* Separator */}
        <div style={{ width: '1px', background: 'var(--border-light)', alignSelf: 'stretch', margin: '0 4px' }} />

        {/* Right: Component bars */}
        <div style={{ flex: 1 }}>
          {components.map(renderBar)}
        </div>
      </div>
    </div>
  );
};
