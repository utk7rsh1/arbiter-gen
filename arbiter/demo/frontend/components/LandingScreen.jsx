// LandingScreen.jsx — Mode selector: Loan (classic) vs Custom Domain (Groq)
(function () {
  const { useState } = React;

  window.LandingScreen = function LandingScreen({ onSelectLoan, onSelectCustom }) {
    const [hovered, setHovered] = useState(null);

    const cards = [
      {
        id: 'loan',
        icon: '🏦',
        title: 'Loan Approval AI',
        subtitle: 'Classic domain — pre-configured',
        description: 'Audit a synthetic loan approval system. Uses the battle-tested loan domain with zip_code proxies, credit_score thresholds, and three embedded anomaly types.',
        badge: 'READY',
        badgeColor: 'var(--green)',
        badgeBg: 'rgba(16,185,129,0.12)',
        accent: 'var(--green)',
        accentGlow: 'var(--green-glow)',
        onClick: onSelectLoan,
      },
      {
        id: 'custom',
        icon: '⚡',
        title: 'Any AI System',
        subtitle: 'Groq-powered domain generation',
        description: 'Describe any AI decision system in plain English. Groq generates the full domain config — features, proxies, causal chains, drift rules — and you can tune it before running.',
        badge: 'GROQ POWERED',
        badgeColor: 'var(--cyan)',
        badgeBg: 'rgba(0,196,224,0.12)',
        accent: 'var(--cyan)',
        accentGlow: 'var(--cyan-glow)',
        onClick: onSelectCustom,
      },
    ];

    return (
      <div style={{
        height: '100vh',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'var(--bg-shell)',
        padding: '40px 24px',
        gap: '0',
      }}>
        {/* Logo + tagline */}
        <div style={{ textAlign: 'center', marginBottom: '56px' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '12px',
            marginBottom: '16px',
          }}>
            <div style={{
              width: '48px', height: '48px',
              background: 'var(--bg-panel-dark)',
              borderRadius: '14px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '22px',
              boxShadow: '0 4px 24px rgba(0,196,224,0.15)',
              border: '1px solid var(--border-dark)',
            }}>🔍</div>
            <span style={{
              fontFamily: 'var(--font-display)',
              fontWeight: 800,
              fontSize: '32px',
              letterSpacing: '0.08em',
              color: 'var(--text-primary)',
            }}>ARBITER</span>
          </div>
          <p style={{
            fontFamily: 'var(--font-body)',
            fontSize: '15px',
            color: 'var(--text-secondary)',
            maxWidth: '420px',
            lineHeight: 1.6,
            margin: '0 auto',
          }}>
            AI Oversight Intelligence Platform — choose the domain you want to audit
          </p>
        </div>

        {/* Mode cards */}
        <div style={{
          display: 'flex',
          gap: '24px',
          width: '100%',
          maxWidth: '820px',
        }}>
          {cards.map(card => (
            <button
              key={card.id}
              onClick={card.onClick}
              onMouseEnter={() => setHovered(card.id)}
              onMouseLeave={() => setHovered(null)}
              style={{
                flex: 1,
                background: 'var(--bg-card)',
                border: `1.5px solid ${hovered === card.id ? card.accent : 'var(--border-light)'}`,
                borderRadius: '20px',
                padding: '32px',
                cursor: 'pointer',
                textAlign: 'left',
                boxShadow: hovered === card.id
                  ? `0 12px 40px ${card.accentGlow}, 0 4px 12px rgba(0,0,0,0.08)`
                  : 'var(--shadow-card)',
                transform: hovered === card.id ? 'translateY(-4px)' : 'translateY(0)',
                transition: 'all 0.22s cubic-bezier(0.16,1,0.3,1)',
              }}
            >
              {/* Badge */}
              <div style={{ marginBottom: '20px' }}>
                <span style={{
                  display: 'inline-block',
                  padding: '4px 12px',
                  background: card.badgeBg,
                  color: card.badgeColor,
                  borderRadius: '999px',
                  fontFamily: 'var(--font-mono)',
                  fontSize: '10px',
                  fontWeight: 600,
                  letterSpacing: '0.08em',
                }}>
                  {card.badge}
                </span>
              </div>

              {/* Icon + title */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '14px', marginBottom: '12px' }}>
                <div style={{
                  width: '52px', height: '52px',
                  background: 'var(--bg-shell)',
                  borderRadius: '14px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '24px',
                  border: '1px solid var(--border-light)',
                }}>
                  {card.icon}
                </div>
                <div>
                  <div style={{
                    fontFamily: 'var(--font-display)',
                    fontWeight: 700,
                    fontSize: '20px',
                    color: 'var(--text-primary)',
                    marginBottom: '4px',
                  }}>
                    {card.title}
                  </div>
                  <div style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: '11px',
                    color: card.accent,
                    letterSpacing: '0.04em',
                  }}>
                    {card.subtitle}
                  </div>
                </div>
              </div>

              {/* Description */}
              <p style={{
                fontFamily: 'var(--font-body)',
                fontSize: '13px',
                color: 'var(--text-secondary)',
                lineHeight: 1.65,
                marginBottom: '24px',
                margin: '0 0 24px 0',
              }}>
                {card.description}
              </p>

              {/* CTA */}
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                fontFamily: 'var(--font-body)',
                fontSize: '13px',
                fontWeight: 600,
                color: card.accent,
              }}>
                {card.id === 'loan' ? 'Start auditing →' : 'Configure domain →'}
              </div>
            </button>
          ))}
        </div>

        {/* Keyboard hint */}
        <div style={{
          marginTop: '40px',
          fontFamily: 'var(--font-mono)',
          fontSize: '11px',
          color: 'var(--text-muted)',
          display: 'flex',
          gap: '24px',
        }}>
          <span>↑↓ navigate</span>
          <span>⏎ select</span>
          <span>J toggle judge view</span>
        </div>
      </div>
    );
  };
})();
